"""One attention layer of the pthfa verify/decode kernel, timed as the CUSTOM backend calls it.

What it measures (docs/research/deepinfer/attention-kv.md: 'Roofline per engine step', attn-1 GPU validation
step 2, open question 1): microseconds per full-attention layer of the target for a verify of T query tokens per
request at a context depth L, on the int8_per_token_head paged cache pthfa reads, with the call the serving
backend makes (pthfa/python/pth_flash_attn/vllm_backend.py, PthFlashAttnImpl._run_slice):

  * cache: one packed [blocks, block, Hkv, 2 * (D + 4)] int8 tensor per layer (NHD; --layout HND), K / V /
    k_scale / v_scale cut out of it by PthFlashAttnImpl._caches, exactly as the backend does; block size 816
    (the hybrid group's), block table a random permutation of the pool (blocks scattered as in a used pool);
  * metadata: seq_lens = L (the KV length, query tokens included), query_start_loc = 0, T, 2T, ...,
    max_query_len = T; the plan from planner.plan_launch(is_prefill=False) with the workspace clamp the builder
    uses (workspace_token_splits(max_num_seqs=--max-num-seqs, q_dec_max=T)), i.e. a verify of 1 + k = T tokens;
  * timing: a CUDA graph holding one launch per layer for --layers distinct caches (16 = the target's
    full-attention layers, so one replay is the attention of one engine step and no layer's KV is in L2 when it
    is read), replayed --inner times between CUDA events, --samples times. us/layer = sample / (inner * layers);
    median, p10 and p90 are over the samples. --no-graph times eager launches the same way.

Bytes read per layer: batch * L * Hkv * (2 * D + 8) (int8 K and V plus one fp32 scale each per token and head;
2,080 B per token at Hq/Hkv/D = 24/4/256). Effective DRAM GB/s = bytes / time; the fraction is of 936 GB/s (the
3090's peak, --peak-gbs). The 'kernel' column mirrors the dispatch (launch.cu tile_tokens, instantiate_d256.cu):
'dec' is the swap-AB decode kernel (rows <= 32, n-tiles of 8 rows; rows <= 48 with the attn-1 tile48 knob),
'tile16' the 16-row-tile kernel (rows padded to 64 per row tile).

attn-1 arms (docs/research/deepinfer/attention-kv.md, attn-1; pth_flash_attn.ops attn-1 mode bits): by default the
mode is the one the environment set at import (PTHFA_TILE48, PTHFA_MMA), recorded in the JSON meta and in every
row's 'arm'. --arms base,tile48,tile48+pv16,tile48+s8pv16 times several arms in one process on the same caches
(set_attn1_mode before each arm's graph is captured) and adds a speed-up column against the first arm; arms are
names joined with '+' (ops.ATTN1_ARMS: base, tile48, pv16, s8pv16; pv16 and s8pv16 exclude each other).

The model shape comes from --model-config (Hq = num_attention_heads, Hkv = num_key_value_heads, D = head_dim),
or --shape Hq,Hkv,D.

  python bench/pthfa_verify_bench.py [--T 1,2,4,5,6,8,10,12,16] [--ctx 8k,25k,64k,112k,128k,150k] [--batch 1]
      [--arms base,tile48] [--out PREFIX]   -> PREFIX.csv, PREFIX.json and a table on stdout
  python bench/pthfa_verify_bench.py --kernel-only 8,150000 [--iters 5] [--nvtx-range pthfa_kernel] [--arms tile48]
      runs --iters eager launches of one config (the first arm) inside an NVTX push/pop range (for ncu:
      bench/ncu_pthfa.sh; select the arm there with PTHFA_TILE48 in the environment or BENCH_ARGS="--arms ...")
  python bench/pthfa_verify_bench.py --dry-run
      CPU only: per ctx builds --dry-run-layers real-size layer caches as the GPU run does and checks every
      launch's exact arguments against the op's shape contract (ops._check_metadata = launch.cu check_metadata,
      plus per-layer view / block-table checks; exit 2 on a violation); prints plans and byte counts

Memory: --layers caches of batch * ceil(L / 816) + 1 blocks: about 5.0 GB at 150k, batch 1, 16 layers.
Exit codes (bench/README.md): 0 = measured (a measurement exits 0 whatever the numbers); 2 = no run: pthfa not
importable or its op library missing, no CUDA device / not sm8x, no model shape, or the caches do not fit.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

PEAK_GBS = 936.0
FULL_ATTN_LAYERS = 16  # the target's full-attention layers (64 layers, full_attention_interval 4)
DEFAULT_CONFIG = "/app/models/Qwen3.8-27B-W4A16-AutoRound-fast/config.json"
DEC_MAX_ROWS = 32  # kDecMaxRows: row tiles of <= 32 rows take the swap-AB decode kernel
DEC_TILE48_MAX_ROWS = 48  # kDecTile48MaxRows: with the attn-1 tile48 bit, D = 256 tiles of <= 48 rows do too


def no_run(msg):
    print(f"pthfa_verify_bench: NO RUN: {msg}", flush=True)
    sys.exit(2)


def parse_tokens(s):
    s = s.strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(s)


def parse_list(s, conv=int):
    return [conv(x) for x in s.split(",") if x.strip()]


def model_shape(args):
    if args.shape:
        hq, hkv, d = parse_list(args.shape)
        return hq, hkv, d, "--shape"
    if not os.path.isfile(args.model_config):
        no_run(f"no model config at {args.model_config} (mount -v /mnt/g/models:/app/models:ro, or --shape Hq,Hkv,D)")
    with open(args.model_config) as f:
        cfg = json.load(f)
    t = cfg.get("text_config", cfg)
    hq, hkv = int(t["num_attention_heads"]), int(t["num_key_value_heads"])
    d = int(t.get("head_dim") or t["hidden_size"] // hq)
    return hq, hkv, d, args.model_config


def tile_tokens(wm, g, q_max):
    """launch.cu tile_tokens(): the row tile the library actually runs."""
    wm_for = lambda rows: next(w for w in (1, 2, 4) if rows <= 16 * w) if rows <= 64 else 4
    cap = 16 * wm // g
    return q_max if 1 <= q_max < cap and wm_for(q_max * g) == wm else cap


def kernel_desc(plan, g, q_max, d, tile48=False):
    bq = tile_tokens(plan.wm, g, q_max)
    rows = bq * g
    if rows <= DEC_MAX_ROWS or (tile48 and d == 256 and rows <= DEC_TILE48_MAX_ROWS):
        return "dec", rows, 8 * math.ceil(rows / 8)
    return "tile16", rows, 64


def kv_bytes(batch, ctx, hkv, d):
    return batch * ctx * hkv * (2 * d + 8)


def make_plan(planner, *, T, batch, g, hkv, d, num_sms, max_num_seqs):
    w_ts = planner.workspace_token_splits(max_num_seqs=max_num_seqs, q_dec_max=T, Hkv=hkv, D=d, num_sms=num_sms)
    return planner.plan_launch(num_tokens=batch * T, num_reqs=batch, max_query_len=T, G=g, Hkv=hkv, D=d,
                               num_sms=num_sms, is_prefill=False, ws_token_splits=w_ts, window=0)


def gpu_state():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,power.draw,power.limit,"
                            "temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else f"nvidia-smi exit {r.returncode}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"nvidia-smi unavailable ({e.__class__.__name__})"


def lib_sha(ops):
    try:
        with open(ops.LIBRARY_PATH, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "unreadable"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--T", default="1,2,4,5,6,8,10,12,16", help="query tokens per request (verify length 1+k)")
    ap.add_argument("--ctx", default="8k,25k,64k,112k,128k,150k", help="KV lengths (seq_lens); k = 1000")
    ap.add_argument("--batch", type=int, default=1, help="requests, each T tokens at the same depth")
    ap.add_argument("--model-config", default=DEFAULT_CONFIG)
    ap.add_argument("--shape", help="Hq,Hkv,D instead of --model-config")
    ap.add_argument("--block-size", type=int, default=816, help="tokens per block (the hybrid group's 816)")
    ap.add_argument("--layout", choices=("NHD", "HND"), default="NHD")
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16", help="query/output dtype")
    ap.add_argument("--layers", type=int, default=FULL_ATTN_LAYERS,
                    help="distinct layer caches cycled per graph replay (16: one engine step, KV not in L2)")
    ap.add_argument("--max-num-seqs", type=int, default=64,
                    help="sizes the workspace clamp as the builder does (start_qwen.sh MAX_SEQS default 64)")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--inner", type=int, default=4, help="graph replays per timed sample")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-graph", action="store_true", help="time eager launches instead of a CUDA graph")
    ap.add_argument("--peak-gbs", type=float, default=PEAK_GBS)
    ap.add_argument("--tag", default="", help="label stored in every row (e.g. the PTHFA_MMA arm)")
    ap.add_argument("--arms", default="",
                    help="comma list of attn-1 arms timed in one process, e.g. base,tile48,tile48+pv16 (default: "
                         "the mode the environment set, PTHFA_TILE48 / PTHFA_MMA)")
    ap.add_argument("--rounds", type=int, default=1,
                    help="with several arms: alternate the arms this many times per config, --samples each time")
    ap.add_argument("--out", default="pthfa_verify_bench", help="output prefix: PREFIX.csv and PREFIX.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="CPU only: plans, views and byte counts; no GPU")
    ap.add_argument("--kernel-only", metavar="T,CTX", help="one config, --iters eager launches in an NVTX range")
    ap.add_argument("--iters", type=int, default=5, help="launches inside the NVTX range of --kernel-only")
    ap.add_argument("--nvtx-range", default="pthfa_kernel")
    ap.add_argument("--num-sms", type=int, default=82, help="SM count for --dry-run (the 3090 has 82)")
    ap.add_argument("--dry-run-layers", type=int, default=2,
                    help="layers built per ctx in --dry-run's shape-contract check (>= 2: catches views shared "
                         "across layers; real size, CPU RAM ~0.3 GB per layer at 150k)")
    args = ap.parse_args()

    if args.kernel_only:
        kt, kc = args.kernel_only.split(",")
        Ts, ctxs = [int(kt)], [parse_tokens(kc)]
    else:
        Ts, ctxs = parse_list(args.T), parse_list(args.ctx, parse_tokens)
    hq, hkv, d, shape_src = model_shape(args)
    if hq % hkv:
        no_run(f"Hq={hq} is not a multiple of Hkv={hkv}")
    g = hq // hkv

    try:
        import torch
        import pth_flash_attn.ops as ops
        from pth_flash_attn import planner
        from pth_flash_attn.vllm_backend import PthFlashAttnImpl
    except ImportError as e:
        no_run(f"pth_flash_attn is not importable here ({e}); run in the image or after pthfa/install.sh")

    scale = d ** -0.5
    pack = 2 * (d + 4)

    # attn-1 arms: (name, mode bits). A pthfa without the attn-1 API has the base arm only.
    has_attn1 = hasattr(ops, "set_attn1_mode")
    if args.arms:
        if not has_attn1:
            no_run(f"--arms needs a pth_flash_attn with the attn-1 knobs (this one, {ops.__file__}, has none)")
        try:
            modes = [ops.attn1_mode_from_arm(a) for a in args.arms.split(",") if a.strip()]
        except ValueError as e:
            no_run(str(e))
    else:
        modes = [ops.attn1_mode_from_env() if has_attn1 else 0]
    arms = [(ops.attn1_arm_name(m) if has_attn1 else "base", m) for m in modes]
    tile48_bit = getattr(ops, "ATTN1_TILE48", 0)
    env_knobs = {k: os.environ.get(k, "") for k in ("PTHFA_TILE48", "PTHFA_MMA")}

    def packed(nb, device):
        shape = (nb, args.block_size, hkv, pack) if args.layout == "NHD" else (nb, hkv, args.block_size, pack)
        kv = torch.empty(shape, dtype=torch.int8, device=device)
        return kv.transpose(1, 2) if args.layout == "NHD" else kv  # the backend's logical [nb, Hkv, bs, 2(D+4)]

    def new_layer(nb, nbr, device, fill):
        """One layer as served: its own packed cache AND its own Impl.

        vLLM builds one PthFlashAttnImpl per layer, and TritonAttentionImpl._ensure_scale_caches makes the
        k/v scale views once per Impl (it returns early once _k_scale_cache is set). One Impl shared across
        caches would hand every later cache the FIRST cache's scale views: wrong NB at a new depth (the
        check_size failure), and silently another layer's scales at the same depth."""
        kv = packed(nb, device)
        impl = PthFlashAttnImpl(num_heads=hq, head_size=d, scale=scale, num_kv_heads=hkv, alibi_slopes=None,
                                sliding_window=None, kv_cache_dtype="int8_per_token_head")
        k, v, ks, vs = impl._caches(kv)
        if fill:
            k.copy_(torch.randint_like(k, -127, 128)); v.copy_(torch.randint_like(v, -127, 128))
            ks.copy_(torch.rand_like(ks) * 0.5 / 127 + 0.5 / 127); vs.copy_(torch.rand_like(vs) * 0.5 / 127 + 0.5 / 127)
        bt = torch.randperm(nb, device=device, dtype=torch.int32)[: args.batch * nbr].view(args.batch, nbr).contiguous()
        return kv, (k, v, ks, vs), bt

    check_metadata = getattr(ops, "_check_metadata", None)

    def contract_check(kv, caches, bt, out, q, sl, qsl, T, plan, ws_o, ws_lse, ctx):
        """The op's shape contract on the exact launch arguments, before any launch.

        ops._check_metadata mirrors check_metadata() in csrc/pth_attn/launch.cu (same order and messages:
        every check_dim / check_dtype / check_size / stride / workspace check). Added here, because the C++
        sizes alone cannot see them: every view lies in its own layer's pack, the block table covers ctx and
        points inside the pool. Raises ValueError (via ops) or AssertionError."""
        if check_metadata is None:
            raise AssertionError(f"{ops.__file__} has no _check_metadata (the shape-contract mirror)")
        check_metadata(out, q, *caches, bt, sl, qsl, T, plan.wm, plan.num_splits, scale, ws_o, ws_lse, 0, True)
        storage = kv.untyped_storage().data_ptr()
        for t, name in zip(caches, ("k_cache", "v_cache", "k_scale", "v_scale")):
            assert t.untyped_storage().data_ptr() == storage, f"{name} is not a view of this layer's cache"
            assert t.shape[0] == kv.shape[0], f"{name}.size(0) = {t.shape[0]}, the cache has {kv.shape[0]} blocks"
        assert bt.shape[1] * args.block_size >= ctx, f"block table covers {bt.shape[1] * args.block_size} < {ctx}"
        assert int(bt.max()) < kv.shape[0] and int(bt.min()) >= 0, "block table points outside the pool"


    print(f"== pthfa verify bench: Hq/Hkv/D = {hq}/{hkv}/{d} (G={g}, from {shape_src}), batch {args.batch}, "
          f"block {args.block_size} {args.layout}, q {args.dtype}, {args.layers} layer caches", flush=True)

    if args.dry_run:
        # Shape contract on CPU: per ctx, --dry-run-layers real-size layers (as the GPU run builds them, minus the
        # random fill) and, per T, the exact launch arguments, through ops._check_metadata and the extras above.
        n_checks = 0
        for ctx in ctxs:
            nbr = math.ceil(ctx / args.block_size)
            nb = args.batch * nbr + 1
            built = [new_layer(nb, nbr, "cpu", False) for _ in range(max(2, args.dry_run_layers))]
            sl = torch.full((args.batch,), ctx, dtype=torch.int32)
            for T in Ts:
                plan = make_plan(planner, T=T, batch=args.batch, g=g, hkv=hkv, d=d, num_sms=args.num_sms,
                                 max_num_seqs=args.max_num_seqs)
                q = torch.empty(args.batch * T, hq, d, dtype=torch.bfloat16)
                qsl = torch.arange(0, (args.batch + 1) * T, T, dtype=torch.int32)
                ws_o = torch.empty(plan.ws_o_shape, dtype=torch.float32)
                ws_lse = torch.empty(plan.ws_lse_shape, dtype=torch.float32)
                for kv, caches, bt in built:
                    try:
                        contract_check(kv, caches, bt, torch.empty_like(q), q, sl, qsl, T, plan, ws_o, ws_lse, ctx)
                    except (AssertionError, ValueError, RuntimeError) as e:
                        print(f"DRY RUN FAIL: shape contract at T={T} ctx={ctx}: {e}", flush=True)
                        return 2
                    n_checks += 1
            del built
        k, v, ks, vs = new_layer(3, 1, "cpu", False)[1]
        print(f"   shape contract OK: {n_checks} launches (ops._check_metadata + view/pool checks), "
              f"{max(2, args.dry_run_layers)} layers per ctx")
        print(f"   cache views (CPU, 3 blocks): k {tuple(k.shape)} stride {k.stride()} {k.dtype}; "
              f"k_scale {tuple(ks.shape)} {ks.dtype}; v/v_scale alias the same pack", flush=True)
        print(f"   op library: {ops.LIBRARY_PATH} (built: {ops.is_available()}); num_sms = {args.num_sms} (--num-sms)")
        print(f"   arms: {', '.join(f'{n} ({m:#x})' for n, m in arms)}; env {env_knobs}")
        print(f"{'T':>3} {'ctx':>7} {'arm':>14} {'wm':>2} {'bq':>3} {'rows':>4} {'kernel':>6} {'mma_rows':>8} "
              f"{'splits':>6} {'grid':>14} {'blocks':>6} {'KV MB/layer':>11} {'floor us':>9} {'GB pool':>8}")
        for ctx in ctxs:
            nb = args.batch * math.ceil(ctx / args.block_size) + 1
            pool_gb = args.layers * nb * args.block_size * hkv * pack / 1e9
            for T in Ts:
                plan = make_plan(planner, T=T, batch=args.batch, g=g, hkv=hkv, d=d, num_sms=args.num_sms,
                                 max_num_seqs=args.max_num_seqs)
                b = kv_bytes(args.batch, ctx, hkv, d)
                for name, mode in arms:
                    kern, rows, mrows = kernel_desc(plan, g, T, d, bool(mode & tile48_bit))
                    print(f"{T:>3} {ctx:>7} {name:>14} {plan.wm:>2} {tile_tokens(plan.wm, g, T):>3} {rows:>4} "
                          f"{kern:>6} {mrows:>8} {plan.num_splits:>6} {str(plan.grid):>14} {nb:>6} "
                          f"{b / 1e6:>11.1f} {b / (args.peak_gbs * 1e3):>9.1f} {pool_gb:>8.2f}")
        print("DRY RUN OK (no GPU used)")
        return 0

    if not torch.cuda.is_available():
        no_run("no CUDA device")
    cap = torch.cuda.get_device_capability()
    if cap[0] != 8:
        no_run(f"sm{cap[0]}{cap[1]}: the library has SASS for sm80/86/89 only")
    if not ops.is_available():
        no_run(f"the package is installed but its op library is not ({ops.LIBRARY_PATH})")
    dev = "cuda"
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    qdtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    meta = dict(device=torch.cuda.get_device_name(0), sm=f"{cap[0]}{cap[1]}", num_sms=num_sms,
                lib=str(ops.LIBRARY_PATH), lib_sha256_16=lib_sha(ops), pthfa_mma=env_knobs["PTHFA_MMA"],
                pthfa_tile48=env_knobs["PTHFA_TILE48"], arms=[dict(arm=n, mode=m) for n, m in arms],
                rounds=args.rounds,
                Hq=hq, Hkv=hkv, D=d, shape_src=shape_src, batch=args.batch, block_size=args.block_size,
                layout=args.layout, dtype=args.dtype, layers=args.layers, max_num_seqs=args.max_num_seqs,
                samples=args.samples, inner=args.inner, graph=not args.no_graph, peak_gbs=args.peak_gbs,
                tag=args.tag, gpu_state_start=gpu_state(), started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"   {meta['device']} sm{meta['sm']} {num_sms} SMs; lib sha256 {meta['lib_sha256_16']}; env {env_knobs}; "
          f"arms {', '.join(f'{n} ({m:#x})' for n, m in arms)}; {meta['gpu_state_start']}", flush=True)

    def use_arm(mode):
        if has_attn1 and ops.attn1_mode() != mode:
            ops.set_attn1_mode(mode)

    rows_out = []
    for ctx in ctxs:
        nbr = math.ceil(ctx / args.block_size)
        nb = args.batch * nbr + 1
        need = args.layers * nb * args.block_size * hkv * pack
        free, _ = torch.cuda.mem_get_info()
        if need > free - (1 << 30):
            no_run(f"ctx {ctx}: {args.layers} layer caches need {need / 1e9:.2f} GB, {free / 1e9:.2f} GB free "
                   f"(lower --layers or --batch)")
        layers = []
        for _ in range(args.layers):
            kv, caches, bt = new_layer(nb, nbr, dev, True)
            layers.append((kv, caches, bt))
        sl = torch.full((args.batch,), ctx, dtype=torch.int32, device=dev)
        for T in Ts:
            plan = make_plan(planner, T=T, batch=args.batch, g=g, hkv=hkv, d=d, num_sms=num_sms,
                             max_num_seqs=args.max_num_seqs)
            q = torch.randn(args.batch * T, hq, d, dtype=qdtype, device=dev)
            out = torch.empty_like(q)
            qsl = torch.arange(0, (args.batch + 1) * T, T, dtype=torch.int32, device=dev)
            ws_o = torch.empty(plan.ws_o_shape, dtype=torch.float32, device=dev)
            ws_lse = torch.empty(plan.ws_lse_shape, dtype=torch.float32, device=dev)

            for kv, caches, bt in layers:  # the op's contract on every layer's exact arguments, before any launch
                contract_check(kv, caches, bt, out, q, sl, qsl, T, plan, ws_o, ws_lse, ctx)

            def launch(i):
                _, caches, bt = layers[i % len(layers)]
                ops.pth_flash_attn_fwd(out, q, *caches, bt, sl, qsl, T, plan.wm, plan.num_splits, scale,
                                       ws_o, ws_lse, 0, True)

            def step():
                for i in range(len(layers)):
                    launch(i)

            if args.kernel_only:
                name, mode = arms[0]
                use_arm(mode)
                kern, rows, mrows = kernel_desc(plan, g, T, d, bool(mode & tile48_bit))
                for _ in range(args.warmup):
                    step()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(args.nvtx_range)
                for i in range(args.iters):
                    launch(i)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                print(f"KERNEL-ONLY T={T} ctx={ctx} arm={name} ({mode:#x}): {args.iters} launches in NVTX range "
                      f"'{args.nvtx_range}' (kernel {kern}, rows {rows}, splits {plan.num_splits}, grid {plan.grid})",
                      flush=True)
                return 0

            # One graph (or the eager step) per arm, captured under that arm's mode; the arms then alternate
            # --rounds times so that clock drift hits them alike.
            runs = []
            for name, mode in arms:
                use_arm(mode)
                for _ in range(args.warmup):
                    step()
                torch.cuda.synchronize()
                if args.no_graph:
                    runs.append(step)
                else:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        step()
                    for _ in range(args.warmup):
                        graph.replay()
                    runs.append(graph.replay)
                torch.cuda.synchronize()
            samples = [[] for _ in arms]
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(max(1, args.rounds)):
                for a, (name, mode) in enumerate(arms):
                    use_arm(mode)  # eager runs launch under the current mode; graphs keep their own
                    run = runs[a]
                    run()
                    torch.cuda.synchronize()
                    for _ in range(args.samples):
                        e0.record()
                        for _ in range(args.inner):
                            run()
                        e1.record()
                        e1.synchronize()
                        samples[a].append(e0.elapsed_time(e1) * 1e3 / (args.inner * len(layers)))
                    if not torch.isfinite(out).all():
                        print(f"   WARNING: non-finite output at T={T} ctx={ctx} arm={name}", flush=True)
            b = kv_bytes(args.batch, ctx, hkv, d)
            med0 = None
            for a, (name, mode) in enumerate(arms):
                kern, rows, mrows = kernel_desc(plan, g, T, d, bool(mode & tile48_bit))
                s = sorted(samples[a])
                q10 = statistics.quantiles(s, n=10, method="inclusive") if len(s) > 1 else s * 9
                med = statistics.median(s)
                med0 = med if med0 is None else med0
                gbs = b / (med * 1e3)
                row = dict(tag=args.tag, arm=name, attn1_mode=mode, T=T, ctx=ctx, batch=args.batch, kernel=kern,
                           rows=rows, mma_rows=mrows, wm=plan.wm, splits=plan.num_splits, us_med=round(med, 2),
                           us_p10=round(q10[0], 2), us_p90=round(q10[-1], 2), kv_mb=round(b / 1e6, 2),
                           gbs=round(gbs, 1), frac_peak=round(gbs / args.peak_gbs, 3),
                           floor_us=round(b / (args.peak_gbs * 1e3), 1),
                           step_ms_16=round(med * FULL_ATTN_LAYERS / 1e3, 3), x_vs_first=round(med0 / med, 3))
                rows_out.append(row)
                print(f"   T={T:>2} ctx={ctx:>6} {name:>14} {kern:>6} splits={plan.num_splits:>2}: {med:8.1f} us/layer "
                      f"(p10 {q10[0]:.1f}, p90 {q10[-1]:.1f}), {gbs:6.1f} GB/s = {gbs / args.peak_gbs:.1%}, "
                      f"x{med0 / med:.3f} vs {arms[0][0]}", flush=True)
            del q, out, ws_o, ws_lse, runs
            if not args.no_graph:
                del graph
        del layers
        torch.cuda.empty_cache()
    use_arm(modes[0])

    meta["gpu_state_end"] = gpu_state()
    cols = list(rows_out[0])
    with open(args.out + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows_out:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    with open(args.out + ".json", "w") as f:
        json.dump(dict(meta=meta, rows=rows_out), f, indent=1)
    print(f"\n{'T':>3} {'ctx':>7} {'arm':>14} {'kernel':>6} {'spl':>3} {'us med':>8} {'p10':>8} {'p90':>8} {'GB/s':>7} "
          f"{'%936':>5} {'floor us':>8} {'ms x16':>7} {'x first':>7}")
    for r in rows_out:
        print(f"{r['T']:>3} {r['ctx']:>7} {r['arm']:>14} {r['kernel']:>6} {r['splits']:>3} {r['us_med']:>8.1f} "
              f"{r['us_p10']:>8.1f} {r['us_p90']:>8.1f} {r['gbs']:>7.1f} {r['frac_peak'] * 100:>5.1f} "
              f"{r['floor_us']:>8.1f} {r['step_ms_16']:>7.3f} {r['x_vs_first']:>7.3f}")
    print(f"wrote {args.out}.csv, {args.out}.json; {meta['gpu_state_end']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
