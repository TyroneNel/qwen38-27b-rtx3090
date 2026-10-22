"""Microbench: KVarN verify attention, µs per layer, stock per-token path vs KVARN_VERIFY_SHARED (attn-2).

docs/research/deepinfer/attention-kv.md, attn-2. Needs the image's vLLM with
docs/research/deepinfer/experiments/patches/attn2-kvarn-verify-shared.patch applied.

Mode E's shape: Hq 24 / Hk 4, head_dim 256, KVarN k4v2 g128 (26,880 B per block and kv head), one request,
the deployment's block bound cdiv(245760, 128) = 1920 (so the stock path runs its 64 splits, as in serving).
The cache is bench/test_kvarn_verify.py's synthetic one (permuted block table, sink / mid / tail blocks in the
fp16 pool). Each arm is one layer's whole verify call (q rotation, stage 1, stage 2, output rotation), captured
in a CUDA graph and replayed (serving replays FULL graphs), timed with CUDA events.

  base    kvarn_verify_attention with the route off: the per-token VQ_INDIRECT split-K path the E profile
          measured (_kvarn_fused_decode_stage1). Its BLOCK_N / warps come from the autotuner; by default the
          bench first tunes it on the server's warmup shape (8 requests x 4 blocks, VQ_INDIRECT off, 64 splits,
          kvarn_attn._warm_decode_kernels), as the server does, and prints the pick. --tune-on-real tunes on the
          first timed shape instead. --base-sweep also times every autotune config of stage 1 alone.
  shared  kvarn_verify_shared_launch (what the route runs), default config and splits unless --configs /
          --splits sweep them; --pv16 adds the KVARN_VERIFY_SHARED_PV16 arm. T=1 is the same kernel at QLEN 1
          (reference only: serving's T=1 decode goes through kvarn_decode_attention, untouched by the patch).
  old_shared  (--old-shared) the older KVARN_SHARED_VERIFY kernel the stock module already has (T in 2/4/8),
          autotuned on the timed shape: the knob-only alternative, minus its known serving corruption.

DRAM fraction = cache bytes the call must read (blocks x Hk x 26,880) / time / 936 GB/s.
  /app/venv/bin/python bench/kvarn_verify_bench.py --json /tmp/kvb.json
  /app/venv/bin/python bench/kvarn_verify_bench.py --ctx 240000 --qlen 8 --configs 16,8,1 32,8,1 64,8,1 32,4,1 \\
      --splits 20 41 64 82 128 --pv16 --base-sweep
--dry-run: import everything, build a tiny case on the CPU, print the plan, exit 0.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_kvarn_verify import D, G, HK, HQ, Case, load_kernels  # noqa: E402

PEAK_BW = 936e9
MAX_CTX_BLOCKS = -(-245760 // G)     # serving's deployment-constant bound (CTX=huge)


def parse_ctx(s):
    s = s.lower()
    return int(float(s[:-1]) * 1024) if s.endswith("k") else int(s)


def time_graph(fn, iters):
    fn()                                                    # compile / autotune outside the capture
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / iters           # µs


def server_like_autotune(tkd, cfg, impl, case):
    """kvarn_attn._warm_decode_kernels step 2: stage 1 on 8 requests x 4 blocks, VQ_INDIRECT off."""
    B, nb = 8, 4
    dev = case.kv.device
    cache = torch.zeros(B * nb, HK, cfg.tile_bytes_aligned, dtype=torch.uint8, device=dev)
    pool = torch.zeros(1, G, HK, D, dtype=torch.float16, device=dev)
    b2s = torch.full((B * nb,), -1, dtype=torch.int32, device=dev)
    bt = torch.zeros(B, MAX_CTX_BLOCKS, dtype=torch.int32, device=dev)
    bt[:, :nb] = torch.arange(B * nb, dtype=torch.int32, device=dev).view(B, nb)
    sl = torch.full((B,), nb * G, dtype=torch.int32, device=dev)
    q = torch.zeros(B, HQ, D, dtype=torch.float16, device=dev)
    splits = tkd.adaptive_num_kv_splits(MAX_CTX_BLOCKS)
    mid_o = torch.zeros(B * HQ, splits, D, dtype=torch.float32, device=dev)
    mid_lse = torch.zeros(B * HQ, splits, dtype=torch.float32, device=dev)
    tkd._kvarn_fused_decode_stage1[(B, HK, splits)](
        q, sl, bt, sl, b2s, cache, pool, pool, mid_o, mid_lse, case.scale,
        HQ * D, D, bt.stride(0), cache.stride(0), cache.stride(1),
        pool.stride(0), pool.stride(1), pool.stride(2),
        mid_o.stride(0), mid_o.stride(1), mid_lse.stride(0),
        NUM_KV_SPLITS=splits, HQ=HQ, **base_consts(tkd, cfg, impl, vq=False))
    torch.cuda.synchronize()


def base_consts(tkd, cfg, impl, vq=True):
    import vllm.envs as envs
    return dict(D=D, GROUP=G, Q_PER_KV=HQ // HK, Q_PER_KV_PAD=8, SLIDING_WINDOW=0,
                K_BITS=cfg.key_bits, V_BITS=cfg.value_bits, NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
                K_PACKED_OFFSET=cfg.k_packed_offset, K_S_COL_OFFSET=cfg.k_s_col_offset,
                K_ZP_OFFSET=cfg.k_zp_offset, K_S_ROW_OFFSET=cfg.k_s_row_offset,
                V_PACKED_OFFSET=cfg.v_packed_offset, V_S_COL_OFFSET=cfg.v_s_col_offset,
                V_S_ROW_OFFSET=cfg.v_s_row_offset, V_ZP_OFFSET=cfg.v_zp_offset,
                VQ_INDIRECT=vq, F16=envs.KVARN_FP16_DEQUANT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ctx", nargs="+", default=["64k", "128k", "240000"])
    ap.add_argument("--qlen", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--configs", nargs="*", default=[], help="shared-kernel BLOCK_N,num_warps,num_stages to sweep")
    ap.add_argument("--splits", nargs="*", type=int, default=[], help="shared-kernel split counts to sweep")
    ap.add_argument("--pv16", action="store_true", help="add the KVARN_VERIFY_SHARED_PV16 arm")
    ap.add_argument("--base-sweep", action="store_true", help="time every stage-1 autotune config alone")
    ap.add_argument("--old-shared", action="store_true", help="add the stock KVARN_SHARED_VERIFY kernel's arm")
    ap.add_argument("--tune-on-real", action="store_true", help="let the base autotune on the first real shape")
    ap.add_argument("--json", default="")
    ap.add_argument("--module", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ctxs = [parse_ctx(c) for c in a.ctx]
    qmax = max(a.qlen)

    if a.dry_run:
        tkd, cfg = load_kernels(a.module)
        case = Case(cfg, [300], qmax, "cpu")
        print(f"kvarn_verify_bench: dry run OK (module {tkd.__name__}, launch "
              f"{'present' if hasattr(tkd, 'kvarn_verify_shared_launch') else 'MISSING'}, "
              f"tile {cfg.tile_bytes_aligned} B, case {case.total} blocks)")
        for c in ctxs:
            nb = -(-c // G)
            print(f"  ctx {c}: {nb} blocks, {nb * HK * cfg.tile_bytes_aligned / 1e6:.1f} MB/layer, "
                  f"DRAM floor {nb * HK * cfg.tile_bytes_aligned / PEAK_BW * 1e6:.0f} µs/layer; T in {a.qlen}")
        return 0
    if not torch.cuda.is_available():
        print("kvarn_verify_bench: no CUDA device (--dry-run checks the import)")
        return 0
    tkd, cfg = load_kernels(a.module)
    if not hasattr(tkd, "kvarn_verify_shared_launch"):
        print("kvarn_verify_bench: the attn2-kvarn-verify-shared patch is not installed in this vLLM")
        return 2

    print(f"building a {max(ctxs)}-token cache ...", flush=True)
    case = Case(cfg, [max(ctxs)], qmax, "cuda", seed=1)
    impl = case.impl
    if not a.tune_on_real:
        server_like_autotune(tkd, cfg, impl, case)
        print(f"base stage-1 autotune pick (server warmup shape): {tkd._kvarn_fused_decode_stage1.best_config}")
    dev = case.kv.device
    tb = cfg.tile_bytes_aligned
    rows = []
    configs = [tuple(int(x) for x in c.split(",")) for c in a.configs] or [None]
    splits_list = a.splits or [None]

    for ctx in ctxs:
        nb = -(-ctx // G)
        bytes_ = nb * HK * tb
        floor = bytes_ / PEAK_BW * 1e6
        for T in a.qlen:
            L = ctx + T
            vq_seqlen = torch.arange(L - T + 1, L + 1, dtype=torch.int32, device=dev)
            vq_req = torch.zeros(T, dtype=torch.int32, device=dev)
            q = torch.randn(T, HQ, D, device=dev).half()
            q_rot = torch.mm(q.reshape(-1, D), case.H16).view(T, HQ, D)
            out = torch.empty_like(q_rot)

            def base():
                tkd._VERIFY_SHARED = False
                return tkd.kvarn_verify_attention(q, case.kv, case.bt, case.scale, cfg, impl, vq_req, vq_seqlen,
                                                  MAX_CTX_BLOCKS)
            t_base = time_graph(base, a.iters)
            pick = str(getattr(tkd._kvarn_fused_decode_stage1, "best_config", ""))
            rows.append(dict(ctx=ctx, T=T, arm="base", config=pick, splits=tkd.adaptive_num_kv_splits(MAX_CTX_BLOCKS),
                             us=t_base, floor_us=floor, dram_frac=floor / t_base, speedup=1.0))
            print(f"ctx {ctx:>7} T {T}: base   {t_base:9.1f} µs  ({floor / t_base:6.1%} of DRAM, floor {floor:.0f} µs)"
                  f"  [{pick}]", flush=True)

            if a.old_shared and T >= 2 and (T * 8) & (T * 8 - 1) == 0:
                seq_lens = torch.tensor([L], dtype=torch.int32, device=dev)

                def old_shared():                       # KVARN_SHARED_VERIFY's kernel, autotuned on this shape
                    os.environ["KVARN_SHARED_VERIFY"] = "1"
                    try:
                        return tkd.kvarn_verify_attention(q, case.kv, case.bt, case.scale, cfg, impl, vq_req,
                                                          vq_seqlen, MAX_CTX_BLOCKS, qlen=T, seq_lens=seq_lens)
                    finally:
                        os.environ.pop("KVARN_SHARED_VERIFY", None)
                try:
                    t = time_graph(old_shared, a.iters)
                    pick_o = str(getattr(tkd._kvarn_fused_verify_stage1, "best_config", ""))
                    rows.append(dict(ctx=ctx, T=T, arm="old_shared", config=pick_o, splits=None, us=t,
                                     floor_us=floor, dram_frac=floor / t, speedup=t_base / t))
                    print(f"ctx {ctx:>7} T {T}: old KVARN_SHARED_VERIFY {t:9.1f} µs  ({floor / t:6.1%} of DRAM)"
                          f"  x{t_base / t:5.2f} vs base  [{pick_o}]", flush=True)
                except Exception as e:
                    print(f"ctx {ctx:>7} T {T}: old KVARN_SHARED_VERIFY: {type(e).__name__}: {str(e)[:120]}")

            for pv16 in ((False, True) if a.pv16 else (False,)):
                for c in configs:
                    for s in splits_list:
                        def shared(c=c, s=s, pv16=pv16):
                            if T == 1:
                                return tkd.kvarn_verify_shared_launch(q_rot, out, case.kv, case.bt, vq_seqlen,
                                                                      case.scale, cfg, impl, 1, MAX_CTX_BLOCKS,
                                                                      n_splits=s, config=c, pv16=pv16)
                            if c is None and s is None and pv16 == tkd._VERIFY_SHARED_PV16:
                                tkd._VERIFY_SHARED = True       # the routed driver call, as serving runs it
                                try:
                                    return tkd.kvarn_verify_attention(q, case.kv, case.bt, case.scale, cfg, impl,
                                                                      vq_req, vq_seqlen, MAX_CTX_BLOCKS, qlen=T)
                                finally:
                                    tkd._VERIFY_SHARED = False
                            o = tkd.kvarn_verify_shared_launch(q_rot, out, case.kv, case.bt, vq_seqlen, case.scale,
                                                               cfg, impl, T, MAX_CTX_BLOCKS, n_splits=s, config=c,
                                                               pv16=pv16)
                            return torch.mm(o.reshape(-1, D), case.H16)
                        try:
                            t = time_graph(shared, a.iters)
                        except Exception as e:
                            print(f"ctx {ctx:>7} T {T}: shared cfg {c} splits {s} pv16 {int(pv16)}: "
                                  f"{type(e).__name__}: {str(e)[:120]}")
                            continue
                        m_pad = tkd.verify_shared_m_pad(T, HQ // HK)
                        cc = c or tkd.verify_shared_config(m_pad)
                        ss = s or tkd.verify_shared_splits(1, HK, MAX_CTX_BLOCKS, dev)
                        rows.append(dict(ctx=ctx, T=T, arm="shared_pv16" if pv16 else "shared", config=str(cc),
                                         splits=ss, us=t, floor_us=floor, dram_frac=floor / t, speedup=t_base / t))
                        print(f"ctx {ctx:>7} T {T}: shared{'+pv16' if pv16 else '     '} {t:9.1f} µs  "
                              f"({floor / t:6.1%} of DRAM)  x{t_base / t:5.2f} vs base  cfg {cc} splits {ss}",
                              flush=True)

    if a.base_sweep:
        print("base stage-1 alone, every autotune config (ctx, T = the last ones above):")
        T = a.qlen[-1]
        ctx = ctxs[-1]
        L = ctx + T
        vq_seqlen = torch.arange(L - T + 1, L + 1, dtype=torch.int32, device=dev)
        vq_req = torch.zeros(T, dtype=torch.int32, device=dev)
        q_rot = torch.randn(T, HQ, D, device=dev).half()
        splits = tkd.adaptive_num_kv_splits(MAX_CTX_BLOCKS)
        mid_o = torch.empty(T * HQ, splits, D, dtype=torch.float32, device=dev)
        mid_lse = torch.empty(T * HQ, splits, dtype=torch.float32, device=dev)
        consts = base_consts(tkd, cfg, impl)
        kern = tkd._kvarn_fused_decode_stage1
        for conf in kern.configs:
            def st1(conf=conf):
                kw = dict(conf.kwargs, num_warps=conf.num_warps, num_stages=conf.num_stages)
                if getattr(conf, "maxnreg", None):
                    kw["maxnreg"] = conf.maxnreg
                kern.fn[(T, HK, splits)](
                    q_rot, vq_req, case.bt, vq_seqlen, impl._block_to_slot_t, case.kv,
                    impl._tail_K_pool, impl._tail_V_pool, mid_o, mid_lse, case.scale,
                    HQ * D, D, case.bt.stride(0), case.kv.stride(0), case.kv.stride(1),
                    impl._tail_K_pool.stride(0), impl._tail_K_pool.stride(1), impl._tail_K_pool.stride(2),
                    mid_o.stride(0), mid_o.stride(1), mid_lse.stride(0),
                    NUM_KV_SPLITS=splits, HQ=HQ, **consts, **kw)
            try:
                t = time_graph(st1, a.iters)
            except Exception as e:
                print(f"  {conf}: {type(e).__name__}: {str(e)[:120]}")
                continue
            rows.append(dict(ctx=ctx, T=T, arm="base_stage1_cfg", config=str(conf), splits=splits, us=t))
            print(f"  {str(conf):70s} {t:9.1f} µs", flush=True)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(device=torch.cuda.get_device_name(), rows=rows,
                           env={k: v for k, v in os.environ.items() if k.startswith("KVARN_")}), f, indent=1)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
