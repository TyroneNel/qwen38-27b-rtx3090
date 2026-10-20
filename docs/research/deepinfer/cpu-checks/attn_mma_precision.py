"""CPU check (b): what the cheaper GA102 MMA modes cost in accuracy for pthfa-style decode/verify.

pthfa today (pthfa/csrc/pth_attn/ptx.cuh:334-341): QK and PV both mma.m16n8k16.f32.f16.f16.f32,
K/V int8 per-token-head dequantised to fp16 (scale folded later). On GeForce GA102 the f32-accumulate
fp16 MMA runs at half the f16-accumulate rate and a quarter of the int8 rate (inferred, GA102 whitepaper).

Arms (all vs fp64 reference using the SAME int8 K/V, so only the MMA-path error is measured):
  base      Q fp16, K int8->fp16, fp32 accumulate; P fp16, V int8->fp16, fp32 accumulate (pthfa today)
  q8k8      Q int8 per-row (absmax/127) x K int8, exact int32 accumulate (mma.s32.s8.s8.s32), then *sq*sk
  pv_f16acc PV accumulated in fp16 within each 64-token tile (4 x k16 MMAs), tile sums added in fp32
  both      q8k8 + pv_f16acc
  qk_f16acc QK accumulated in fp16 (Q pre-scaled by 1/sqrt(D)); all_f16acc = qk_f16acc + pv_f16acc
SYNTHETIC tensors (same generator idea as kvq_error.py, regime B, logit std 3). Run: python3 -I mma_precision.py
"""
import numpy as np

rng = np.random.default_rng(1)
D, L, G, TILE, K16 = 256, 8192, 6 * 8, 64, 16   # G rows = 6 heads x 8-token verify block


def q8_rows(x):
    s = np.abs(x).max(-1, keepdims=True) / 127.0
    return np.clip(np.round(x / s), -127, 127), s


def data():
    ch = np.exp(rng.normal(0, 0.6, D))
    ch[rng.choice(D, 6, replace=False)] *= 15.0
    mean = rng.normal(0, 1.0, D) * ch * 0.5
    K = rng.normal(0, 1, (L, D)) * ch + mean
    V = rng.normal(0, 1, (L, D)) * np.exp(rng.normal(0, 0.3, D))
    tgt = rng.choice(L, G * 4, replace=False)
    Q = np.stack([(K[tgt[4 * g:4 * g + 4]] - mean).mean(0) for g in range(G)]) + rng.normal(0, 0.3, (G, D)) * ch
    Q *= 3.0 / (Q @ K.T / np.sqrt(D)).std()
    return Q, K, V


def pv_f16acc(P16, V16):
    """fp16 accumulation inside each 64-token tile, k16 MMA partials rounded into an fp16 accumulator."""
    out = np.zeros((P16.shape[0], V16.shape[1]))
    for t0 in range(0, L, TILE):
        acc = np.zeros((P16.shape[0], V16.shape[1]), np.float16)
        for k0 in range(t0, t0 + TILE, K16):
            part = P16[:, k0:k0 + K16].astype(np.float64) @ V16[k0:k0 + K16].astype(np.float64)
            acc = (acc.astype(np.float64) + part).astype(np.float16)
        out += acc
    return out


def run():
    Q, K, V = data()
    k8, sk = q8_rows(K)          # int8 per-token-head K (HyperQwen int8_per_token_head)
    v8, sv = q8_rows(V)
    Kd, Vd = k8 * sk, v8 * sv    # exact dequantised cache = the reference's K/V
    s_ref = Q @ Kd.T / np.sqrt(D)
    m = s_ref.max(-1, keepdims=True)
    p_ref = np.exp(s_ref - m)
    o_ref = (p_ref @ Vd) / p_ref.sum(-1, keepdims=True)

    Q16 = Q.astype(np.float16).astype(np.float64)
    s_base = (Q16 @ k8.T.astype(np.float16).astype(np.float64)) * sk.T / np.sqrt(D)   # fp32 acc ~ exact here
    qq, sq = q8_rows(Q)
    s_q8 = (qq @ k8.T) * sq * sk.T / np.sqrt(D)                                       # int32 exact

    # QK with fp16 accumulate over D=256 (16 x k16 partials), Q pre-scaled by 1/sqrt(D) and K scale applied after
    Qs = (Q / np.sqrt(D)).astype(np.float16).astype(np.float64)
    k16 = k8.astype(np.float16).astype(np.float64)
    acc = np.zeros((G, L), np.float16)
    for d0 in range(0, D, K16):
        acc = (acc.astype(np.float64) + Qs[:, d0:d0 + K16] @ k16[:, d0:d0 + K16].T).astype(np.float16)
    s_qkf16 = acc.astype(np.float64) * sk.T
    res = {}
    for name, s, f16acc in (("base", s_base, False), ("q8k8", s_q8, False),
                            ("pv_f16acc", s_base, True), ("both", s_q8, True),
                            ("qk_f16acc", s_qkf16, False), ("all_f16acc", s_qkf16, True)):
        mm = s.max(-1, keepdims=True)
        P16 = np.exp(s - mm).astype(np.float16)            # P is fp16 into the PV MMA in pthfa
        V16 = v8.astype(np.float16)                         # int8 -> fp16 exact; sv folded per token into P
        Pw = (P16.astype(np.float64) * sv.T).astype(np.float16)  # fold V scale into P (one possible layout)
        num = pv_f16acc(Pw, V16) if f16acc else Pw.astype(np.float64) @ V16.astype(np.float64)
        o = num / P16.astype(np.float64).sum(-1, keepdims=True)
        res[name] = (np.sqrt(((s - s_ref) ** 2).mean()), np.abs(s - s_ref).max(),
                     np.linalg.norm(o - o_ref) / np.linalg.norm(o_ref))
    print(f"rows={G} L={L} D={D}; reference = fp64 attention on the dequantised int8 cache")
    print(f"{'arm':10s} {'logit rmse':>10s} {'logit maxabs':>12s} {'out relerr':>10s}")
    for k, (a, b, c) in res.items():
        print(f"{k:10s} {a:10.5f} {b:12.5f} {c:10.6f}")
    return res


if __name__ == "__main__":
    r = run()
    assert r["base"][2] < 1e-2, "baseline should be near-exact"
