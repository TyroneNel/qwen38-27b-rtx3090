"""attn-3 microbenchmark: one prefill chunk of the pthfa CUSTOM backend, shipped QK vs int8 QK, and the projection
to a cold prefill (docs/research/deepinfer/attention-kv.md, attn-3).

One full-attention layer of the target (Hq/Hkv/D from --model-config, 24/4/256) runs a chunked-prefill launch as
PthFlashAttnImpl._run_slice makes it: Q = --chunk new tokens (2048, the launcher's --max-num-batched-tokens) of one
request whose KV length is L (the chunk included), on the paged int8_per_token_head cache (block 816, blocks
scattered), the plan from planner.plan_launch(is_prefill=True). Arms are pth_flash_attn.ops attn-1/attn-3 mode bits
(ops.ATTN1_ARMS; default base,prefill_s8), set with ops.set_attn1_mode before each arm's launches. Timing: eager
launches (prefill is eager in serving) between CUDA events, --layers distinct caches round-robin so no layer's KV
sits in L2, --samples samples of --inner launches; median ms per layer-chunk.

Projection: attention time of a chunked cold prefill of --project tokens, summed over its chunks (chunk c ends at
depth min((c + 1) * chunk, P)) by linear interpolation of the measured depths (a chunk's attention is linear in
its depth), times --full-layers (16). Printed per arm with the saving against the first arm; that saving over the
measured cold TTFT is the end-to-end gain to expect.

Also checks, per depth, that each non-base arm's output stays close to the base arm's (rel-L2, reported; the
correctness gate is bench/test_pthfa_attn.py with PTHFA_PREFILL_QK=s8).

  python bench/pthfa_prefill_bench.py [--ctx 4k,16k,51k,100k,150k] [--arms base,prefill_s8] [--out PREFIX]
"""
import argparse
import json
import math
import os
import sys

import torch

DEFAULT_CONFIG = "/app/models/Qwen3.8-27B-W4A16-AutoRound-fast/config.json"


def parse_tokens(s):
    s = s.strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ctx", default="4k,16k,51k,100k,150k", help="KV length at the END of the chunk")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--arms", default="base,prefill_s8")
    ap.add_argument("--model-config", default=DEFAULT_CONFIG)
    ap.add_argument("--shape", help="Hq,Hkv,D instead of --model-config")
    ap.add_argument("--block-size", type=int, default=816)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--layers", type=int, default=4, help="distinct layer caches, round-robin (L2 defeat)")
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--project", type=int, default=150000, help="cold prefill length for the projection")
    ap.add_argument("--full-layers", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("pthfa_prefill_bench: NO RUN: no CUDA device"); return 2
    import pth_flash_attn.ops as ops
    from pth_flash_attn import planner
    from pth_flash_attn.vllm_backend import PthFlashAttnImpl
    if not ops.is_available():
        print(f"pthfa_prefill_bench: NO RUN: op library not loaded ({ops.LIBRARY_PATH})"); return 2
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    modes = [ops.attn1_mode_from_arm(a) for a in arms]
    if args.shape:
        hq, hkv, d = (int(x) for x in args.shape.split(","))
    else:
        with open(args.model_config) as f:
            t = json.load(f)
        t = t.get("text_config", t)
        hq, hkv = int(t["num_attention_heads"]), int(t["num_key_value_heads"])
        d = int(t.get("head_dim") or t["hidden_size"] // hq)
    g = hq // hkv
    dev = "cuda"
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    ctxs = sorted(parse_tokens(c) for c in args.ctx.split(","))
    bs = args.block_size
    impl = PthFlashAttnImpl(num_heads=hq, head_size=d, scale=d ** -0.5, num_kv_heads=hkv, alibi_slopes=None,
                            sliding_window=None, kv_cache_dtype="int8_per_token_head")
    w_ts = planner.workspace_token_splits(max_num_seqs=4, q_dec_max=4, Hkv=hkv, D=d, num_sms=nsm)
    plan = planner.plan_launch(num_tokens=args.chunk, num_reqs=1, max_query_len=args.chunk, G=g, Hkv=hkv, D=d,
                               num_sms=nsm, is_prefill=True, ws_token_splits=w_ts, window=0)
    ws_o = torch.empty(plan.ws_o_shape, dtype=torch.float32, device=dev)
    ws_lse = torch.empty(plan.ws_lse_shape, dtype=torch.float32, device=dev)
    print(f"== pthfa prefill chunk, {torch.cuda.get_device_name(0)}, Hq/Hkv/D {hq}/{hkv}/{d}, chunk {args.chunk}, "
          f"wm {plan.wm} splits {plan.num_splits}, arms {arms}, env mode {ops.attn1_mode():#x}", flush=True)
    nb_max = math.ceil(max(ctxs) / bs)
    caches = []
    for _ in range(args.layers):
        # One Impl per layer cache: TritonAttentionImpl binds its scale views to the first cache it sees.
        li = PthFlashAttnImpl(num_heads=hq, head_size=d, scale=d ** -0.5, num_kv_heads=hkv, alibi_slopes=None,
                              sliding_window=None, kv_cache_dtype="int8_per_token_head")
        kv = torch.empty((nb_max + 1, bs, hkv, 2 * (d + 4)), dtype=torch.int8, device=dev).transpose(1, 2)
        k, v, ks, vs = li._caches(kv)
        k.copy_(torch.randint_like(k, -127, 128)); v.copy_(torch.randint_like(v, -127, 128))
        ks.copy_(torch.rand_like(ks) * 0.5 / 127 + 0.5 / 127); vs.copy_(torch.rand_like(vs) * 0.5 / 127 + 0.5 / 127)
        caches.append((k, v, ks, vs, kv))
    q = torch.randn(args.chunk, hq, d, dtype=dt, device=dev)
    qsl = torch.tensor([0, args.chunk], dtype=torch.int32, device=dev)
    rows = []
    prev = ops.attn1_mode()
    for L in ctxs:
        nb = math.ceil(L / bs)
        bt = torch.randperm(nb_max + 1, device=dev, dtype=torch.int32)[:nb].reshape(1, nb).contiguous()
        sl = torch.tensor([L], dtype=torch.int32, device=dev)
        outs = {}
        for arm, mode in zip(arms, modes):
            ops.set_attn1_mode(mode)

            def run(i, o):
                k, v, ks, vs, _ = caches[i % args.layers]
                ops.pth_flash_attn_fwd(o, q, k, v, ks, vs, bt, sl, qsl, args.chunk, plan.wm, plan.num_splits,
                                       impl.scale, ws_o, ws_lse, 0, True)
            o = torch.empty_like(q)
            for i in range(2 * args.layers):
                run(i, o)
            torch.cuda.synchronize()
            ts = []
            for s in range(args.samples):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for i in range(args.inner * args.layers):
                    run(i, o)
                e1.record(); torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) / (args.inner * args.layers))
            ts.sort()
            run(0, o); torch.cuda.synchronize()
            outs[arm] = o.float()
            med = ts[len(ts) // 2]
            flop = 4.0 * args.chunk * (L - args.chunk / 2) * hq * d   # causal chunk at the end of L
            rel = (outs[arm] - outs[arms[0]]).norm().item() / outs[arms[0]].norm().item()
            rows.append({"arm": arm, "mode": mode, "ctx": L, "ms": med, "p10": ts[len(ts) // 10],
                         "p90": ts[(9 * len(ts)) // 10], "tflops": flop / med / 1e9, "rel_l2_vs_first": rel})
            base = next(r["ms"] for r in rows if r["ctx"] == L and r["arm"] == arms[0])
            print(f"ROW ctx={L:>7d} arm={arm:<12s} {med:8.3f} ms/layer-chunk  {flop / med / 1e9:6.1f} TFLOP/s  "
                  f"x{base / med:5.3f} vs {arms[0]}  rel-L2 vs {arms[0]} {rel:.2e}", flush=True)
    ops.set_attn1_mode(prev)

    # Projection to a cold prefill of --project tokens.
    proj = {}
    for arm in arms:
        pts = sorted((r["ctx"], r["ms"]) for r in rows if r["arm"] == arm)

        def at(x):
            if x <= pts[0][0]:
                return pts[0][1] * x / pts[0][0]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if x <= x1:
                    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            return y1 + (y1 - y0) * (x - x1) / (x1 - x0)
        total = 0.0
        c = 0
        while c * args.chunk < args.project:
            total += at(min((c + 1) * args.chunk, args.project))
            c += 1
        proj[arm] = total * args.full_layers / 1000.0
    for arm in arms:
        print(f"PROJECT cold {args.project} tokens: attention {proj[arm]:7.2f} s ({args.full_layers} layers) arm={arm}"
              f"  saving vs {arms[0]} {proj[arms[0]] - proj[arm]:+7.2f} s", flush=True)
    if args.out:
        with open(args.out + ".json", "w") as f:
            json.dump({"meta": {"shape": [hq, hkv, d], "chunk": args.chunk, "plan": [plan.wm, plan.num_splits],
                                "gpu": torch.cuda.get_device_name(0), "arms": arms}, "rows": rows,
                       "projection_s": proj}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
