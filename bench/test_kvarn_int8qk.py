"""attn-3, mode E (docs/research/deepinfer/attention-kv.md attn-3; experiments/patches/attn3-kvarn-int8qk.patch):
KVarN's int8-QK continuation kernel against an fp64 reference and against FA2 on the same tensors, then its time.

The kernel (vllm/v1/attention/backends/kvarn_prefill_int8.py, only in an image built with the patch) runs on what
KVarN's _cached_multiquery_path hands FA2 for a chunked-prefill continuation: q_rot [q_len, 24, 256] fp16 and the
request's whole packed K/V [kv_len, 4, 256] fp16, causal bottom-right. Data: rotated-frame-like K/V (Gaussian,
per-channel spread, a per-channel offset on K, as KVarN's Hadamard frame flattens outliers), q correlated with a
few keys so rows are peaky.

1. Accuracy (q_len 512 / 2048 at kv 3k-16k): rel-L2 and min row cosine of the output against fp64, next to FA2's.
   Gate: rel-L2 <= max(3 x FA2's, 1e-2) and min cosine >= 0.999 (int8 QK is lossy; patches/prefill-attn-int8.patch
   measured cos > 0.99999 on the bf16 arm).
2. Speed (q_len 2048 at kv 16k / 51k / 100k / 150k): ms per layer-chunk, int8 vs FA2 (median of 10), and the
   projection to a 150k cold prefill's attention (16 layers, chunks of 2048).
Exit 0 = gate met, 1 = failed, 0 with SKIP(by design) without CUDA or without the patched module.
"""
import math
import sys

import torch

if not torch.cuda.is_available():
    print("kvarn int8qk: SKIP(by design): no CUDA device"); sys.exit(0)
try:
    from vllm.v1.attention.backends.kvarn_prefill_int8 import prefill_attn_int8_contig
except ImportError as e:
    print(f"kvarn int8qk: SKIP(by design): the image has no attn3-kvarn-int8qk.patch ({e})"); sys.exit(0)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version

DEV = "cuda"
HQ, HKV, D = 24, 4, 256
SCALE = D ** -0.5
FA_VER = get_flash_attn_version(head_size=D)


def data(q_len, kv_len, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ch = torch.exp(torch.randn(D, generator=g) * 0.3)
    mean = torch.randn(HKV, D, generator=g) * 0.5
    k = torch.randn(kv_len, HKV, D, generator=g) * ch + mean
    v = torch.randn(kv_len, HKV, D, generator=g)
    tgt = torch.randint(0, kv_len, (q_len, HQ), generator=g)
    kk = k[tgt.clamp(max=kv_len - 1), torch.arange(HQ)[None, :] // (HQ // HKV)]
    q = 0.5 * kk + torch.randn(q_len, HQ, D, generator=g)
    return q.half().to(DEV), k.half().to(DEV), v.half().to(DEV)


def reference(q, k, v):
    q_len, kv_len = q.shape[0], k.shape[0]
    out = torch.empty(q.shape, dtype=torch.float64, device=DEV)
    pos = kv_len - q_len + torch.arange(q_len, device=DEV)
    vis = torch.arange(kv_len, device=DEV)[None, :] <= pos[:, None]
    g = HQ // HKV
    for h in range(HKV):
        qh = q[:, h * g:(h + 1) * g].double()
        p = (torch.einsum("qgd,kd->gqk", qh, k[:, h].double()) * SCALE).masked_fill(~vis, -math.inf).softmax(-1)
        out[:, h * g:(h + 1) * g] = torch.einsum("gqk,kd->qgd", p, v[:, h].double())
    return out


def fa2(q, k, v):
    cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=DEV)
    kw = {} if FA_VER is None else {"fa_version": FA_VER}
    return flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=q.shape[0],
                                  max_seqlen_k=k.shape[0], softmax_scale=SCALE, causal=True, **kw)


def err(o, ref):
    o = o.double()
    rel = (o - ref).norm().item() / ref.norm().item()
    cos = torch.nn.functional.cosine_similarity(o.reshape(-1, D), ref.reshape(-1, D), dim=-1).min().item()
    return rel, cos


def timed(fn, n=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return sorted(ts)[n // 2]


print(f"== kvarn int8qk, {torch.cuda.get_device_name(0)}, FA version {FA_VER}", flush=True)
ok_all = True
print("== 1. accuracy against fp64 (and FA2's)", flush=True)
for i, (q_len, kv_len) in enumerate(((512, 3000), (2048, 2048), (2048, 9000), (2048, 16384), (700, 16384))):
    q, k, v = data(q_len, kv_len, i)
    ref = reference(q, k, v)
    r8, c8 = err(prefill_attn_int8_contig(q, k, v, SCALE), ref)
    rf, cf = err(fa2(q, k, v), ref)
    ok = r8 <= max(3 * rf, 1e-2) and c8 >= 0.999
    ok_all &= ok
    print(f"  {'OK  ' if ok else 'FAIL'}  q={q_len} kv={kv_len}: int8 rel-L2 {r8:.2e} min cos {c8:.6f} | "
          f"FA2 rel-L2 {rf:.2e} min cos {cf:.6f}", flush=True)
    del ref
    torch.cuda.empty_cache()

print("== 2. speed, q_len 2048 (one chunk), ms per layer-chunk", flush=True)
pts = {"int8": [], "fa2": []}
for kv_len in (16384, 51200, 100000, 150000):
    q, k, v = data(2048, kv_len, 99)
    t8 = timed(lambda: prefill_attn_int8_contig(q, k, v, SCALE))
    tf = timed(lambda: fa2(q, k, v))
    pts["int8"].append((kv_len, t8)); pts["fa2"].append((kv_len, tf))
    print(f"ROW kv={kv_len:>6d}  int8 {t8:8.3f} ms  FA2 {tf:8.3f} ms  x{tf / t8:.3f}", flush=True)
    del q, k, v
    torch.cuda.empty_cache()


def project(p, total=150000, chunk=2048, layers=16):
    def at(x):
        if x <= p[0][0]:
            return p[0][1] * x / p[0][0]
        for (x0, y0), (x1, y1) in zip(p, p[1:]):
            if x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return p[-1][1]
    return sum(at(min((c + 1) * chunk, total)) for c in range(math.ceil(total / chunk))) * layers / 1000


pi, pf = project(pts["int8"]), project(pts["fa2"])
print(f"PROJECT cold 150000 tokens, 16 layers: attention int8 {pi:.2f} s, FA2 {pf:.2f} s, saving {pf - pi:+.2f} s",
      flush=True)
print(f"KVARN INT8QK {'OK' if ok_all else 'FAIL'}")
sys.exit(0 if ok_all else 1)
