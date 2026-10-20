"""CPU check (a): KV quantisation error of HyperQwen formats vs DeepInfer's block-scaled formats.

SYNTHETIC data only (no weights). K has per-channel scale spread + a few outlier channels
(the shape that motivates per-channel smoothing / Hadamard); V is milder. Queries are drawn
correlated with a few keys so attention is peaky, like retrieval heads. hd=256, one KV head,
GQA 6 query heads. Metric: relative L2 error of the attention OUTPUT vs fp32 reference,
plus logit RMSE (in logit units after 1/sqrt(d)).

Formats (bits/elt incl. scale overhead):
  int8_pth      int8 per-token-head, fp16 scale            (HyperQwen int8_per_token_head)
  fp8_tensor    e4m3, one per-tensor scale                  (HyperQwen fp8 / fa2, scale=1 here*)
  mxfp8_32      e4m3 per-32 ue8m0 scale                     (DeepInfer V41_FORMAT, cache_common.py)
  int4_pth      int4 per-token-head, fp16 scale             (HyperQwen int4_per_token_head)
  nvfp4_16      e2m1 per-16, e4m3 scale (no global)         (DeepInfer V41_FP4_FORMAT)
  int4_g16      int4 per-16, fp16 scale                     (ablation: int grid, same granularity)
  int4_g32_ue8  int4 per-32, ue8m0 scale                    (ablation)
  +H            same after a 256-pt orthonormal Hadamard on K and V (rotate q too; V output rotated back)
* fp8 with scale 1 is what vLLM's default k_scale/v_scale=1.0 does when no calibrated scale is loaded.
Run: python3 -I kvq_error.py
"""
import numpy as np

rng = np.random.default_rng(0)
D, L, G = 256, 8192, 6


def e4m3(x):
    x = np.clip(x, -448, 448)
    a = np.abs(x)
    e = np.floor(np.log2(np.maximum(a, 1e-30)))
    e = np.maximum(e, -6)  # subnormal step 2^-9
    step = 2.0 ** (e - 3)
    return np.sign(x) * np.round(a / step) * step


E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def e2m1(x):
    a = np.abs(x)
    idx = np.abs(a[..., None] - E2M1).argmin(-1)
    return np.sign(x) * E2M1[idx]


def blocked(x, g, quant, qmax, scale_fmt):
    s = x.reshape(*x.shape[:-1], -1, g)
    amax = np.abs(s).max(-1, keepdims=True)
    sc = np.maximum(amax / qmax, 1e-12)
    if scale_fmt == "ue8m0":
        sc = 2.0 ** np.ceil(np.log2(sc))
    elif scale_fmt == "e4m3":
        sc = np.maximum(e4m3(sc), 2.0 ** -9)
    elif scale_fmt == "fp16":
        sc = sc.astype(np.float16).astype(np.float64)
    return (quant(s / sc) * sc).reshape(x.shape)


def intq(bits):
    q = 2 ** (bits - 1) - 1
    return (lambda v: np.clip(np.round(v), -q, q)), q


i8, q8 = intq(8)
i4, q4 = intq(4)
FORMATS = {
    "int8_pth":     lambda x: blocked(x, D, i8, q8, "fp16"),
    "fp8_tensor":   lambda x: e4m3(x),
    "mxfp8_32":     lambda x: blocked(x, 32, e4m3, 448, "ue8m0"),
    "int4_pth":     lambda x: blocked(x, D, i4, q4, "fp16"),
    "nvfp4_16":     lambda x: blocked(x, 16, e2m1, 6, "e4m3"),
    "int4_g16":     lambda x: blocked(x, 16, i4, q4, "fp16"),
    "int4_g32_ue8": lambda x: blocked(x, 32, i4, q4, "ue8m0"),
}
BITS = {"int8_pth": 8 + 16 / D, "fp8_tensor": 8, "mxfp8_32": 8 + 8 / 32, "int4_pth": 4 + 16 / D,
        "nvfp4_16": 4 + 8 / 16, "int4_g16": 4 + 16 / 16, "int4_g32_ue8": 4 + 8 / 32}


def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


H = hadamard(D)


def make_data(regime="A"):
    ch = np.exp(rng.normal(0, 0.6, D))           # per-channel spread
    ch[rng.choice(D, 6, replace=False)] *= 15.0  # outlier channels (K only)
    mean = rng.normal(0, 1.0, D) * ch * 0.5      # per-channel offset (what SageAttention smooths)
    K = rng.normal(0, 1, (L, D)) * ch + mean
    V = rng.normal(0, 1, (L, D)) * np.exp(rng.normal(0, 0.3, D))
    tgt = rng.choice(L, G * 4, replace=False)
    if regime == "A":  # adversarial: q weights the low-variance channels a per-row scale crushes
        Q = np.stack([K[tgt[4 * g:4 * g + 4]].mean(0) / ch ** 2 * 3.0 + rng.normal(0, 0.05, D) for g in range(G)])
    else:  # B: q aligned with key directions incl. outlier channels (rope-like shared outliers)
        Q = np.stack([(K[tgt[4 * g:4 * g + 4]] - mean).mean(0) for g in range(G)])
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True) * np.sqrt(D) * 1.2 + rng.normal(0, 0.3, (G, D)) * ch
    return Q, K, V


def attn(Q, K, V):
    s = Q @ K.T / np.sqrt(D)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return s, p @ V


def main(regime):
    Q, K, V = make_data(regime)
    Q = Q * (3.0 / (Q @ K.T / np.sqrt(D)).std())  # normalise to logit std 3 (realistic, not one-hot)
    s0, o0 = attn(Q, K, V)
    ent = None
    print(f"regime {regime}: synthetic L={L} D={D} G={G}; ref logit std {s0.std():.2f}, max {s0.max():.1f}")
    print(f"{'format':14s} {'bits':>5s} {'K relerr':>9s} {'logit rmse':>10s} {'out relerr':>10s} | "
          f"{'+H K relerr':>11s} {'+H logit':>9s} {'+H out':>9s}")
    for name, f in FORMATS.items():
        row = []
        for rot in (False, True):
            if rot:
                Kq = f(K @ H) @ H.T
                Vq = f(V @ H) @ H.T
            else:
                Kq, Vq = f(K), f(V)
            s, o = attn(Q, Kq, Vq)
            row += [np.linalg.norm(Kq - K) / np.linalg.norm(K), np.sqrt(((s - s0) ** 2).mean()),
                    np.linalg.norm(o - o0) / np.linalg.norm(o0)]
        print(f"{name:14s} {BITS[name]:5.2f} {row[0]:9.4f} {row[1]:10.4f} {row[2]:10.4f} | "
              f"{row[3]:11.4f} {row[4]:9.4f} {row[5]:9.4f}")


if __name__ == "__main__":
    # self-check: e2m1 grid and e4m3 rounding are exact on representable values
    assert np.allclose(e2m1(np.array([0.74, 2.6, 7.0])), [0.5, 3.0, 6.0])
    assert np.allclose(e4m3(np.array([1.0625, 448.0, 500.0])), [1.0, 448.0, 448.0])
    import sys
    for r in (sys.argv[1:] or ["A", "B"]):
        main(r)
