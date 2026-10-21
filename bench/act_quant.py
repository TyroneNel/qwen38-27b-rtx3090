"""Activation quantizers for act_calib.py --modes and humming_probe.py.

The math is copied from docs/research/deepinfer/cpu-checks/quant_a_act_quant.py (check a);
keep the two in step. Every quantizer takes fp32 activations [T, K] and returns the
dequantized activations in the original (unrotated) domain, so y = W . q(x) is the
quantized layer output. Hadamard-128 quantizes x.H and rotates back with H (block-diagonal,
symmetric and orthonormal, so H.H = I): y = W . (Q(x.H) . H), which is exactly
(W.H) . Q(x.H), the rotated-weight GEMM of quant-4. Weight requantization error after a
real rotation is not part of this number.

Run alone, it reproduces check (a)'s numbers on its synthetic tensors (CPU, ~1 min).
"""
import math
import torch

GROUP = 128


def int8_per_token(x):
    s = x.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
    return (x / s).round().clamp(-127, 127) * s


def int8_group(x, g, pow2=False):
    T, K = x.shape
    xg = x.view(T, K // g, g)
    s = xg.abs().amax(2, keepdim=True).clamp(min=1e-8) / 127
    if pow2:
        s = torch.exp2(torch.ceil(torch.log2(s)))  # ue8m0: round scale up to 2^k
    return ((xg / s).round().clamp(-127, 127) * s).view(T, K)


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def had_block(x, b, H):
    T, K = x.shape
    return (x.view(T, K // b, b) @ H).view(T, K)


def make_acts(T, K, df, n_out, out_mag, seed):
    g = torch.Generator().manual_seed(seed)
    ch = torch.exp(torch.randn(K, generator=g) * 0.5)               # log-normal channel scales
    idx = torch.randperm(K, generator=g)[:n_out]
    ch[idx] *= out_mag                                                  # persistent outlier channels
    t = torch.distributions.StudentT(df).sample((T, K))                # heavy-tailed tokens
    return t * ch


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-8)).item()


_H = {}


def _h128(device):
    key = str(device)
    if key not in _H:
        _H[key] = hadamard(GROUP).to(device)
    return _H[key]


def _had(x, inner):
    H = _h128(x.device)
    return had_block(inner(had_block(x, GROUP, H)), GROUP, H)


# name -> (quantizer, needs K % 128 == 0)
MODES = {
    "per-token": (int8_per_token, False),
    "per-128": (lambda x: int8_group(x, GROUP), True),
    "per-128-pow2": (lambda x: int8_group(x, GROUP, True), True),
    "hadamard-128": (lambda x: _had(x, int8_per_token), True),
    "hadamard-128-g128": (lambda x: _had(x, lambda r: int8_group(r, GROUP)), True),
}
BASELINE = "per-token"
# Not a quantizer: the measurement's own noise. The input is scaled by FLOOR_C (inexact
# in bf16, carried exactly by the hi/lo split) and the output scaled back, so what remains
# is the kernel's bf16 output rounding. A mode's error near this floor can't fall further.
FLOOR = "floor"
FLOOR_C = 1.0 + 2.0 ** -6
ALL_MODES = list(MODES) + [FLOOR]


def parse_modes(spec):
    names = list(ALL_MODES) if spec.strip() == "all" else [m.strip() for m in spec.split(",") if m.strip()]
    bad = [m for m in names if m not in ALL_MODES]
    if bad:
        raise ValueError(f"unknown mode(s) {bad}; choose from {ALL_MODES} or 'all'")
    if BASELINE not in names:
        names.insert(0, BASELINE)       # every ratio is against per-token
    return list(dict.fromkeys(names))


def layer_errors(fwd, xs, modes):
    """Per-mode relative output and activation error of one linear layer on rows xs.

    fwd(x) runs the layer on a tensor of xs's dtype and returns its output tensor. The
    dequantized input is fed as hi + lo (two bf16 tensors) and the outputs summed in fp32,
    with the layer's response to zero (its bias) removed once, so the input reaches the
    layer at fp32 precision instead of being rounded back to bf16. Returns
    {mode: (out_err, act_err)}; modes that need K % 128 are left out when it doesn't hold.
    """
    dt = xs.dtype
    y = fwd(xs).float()
    z = fwd(torch.zeros_like(xs[:1])).float()
    xf = xs.float()
    K = xf.shape[-1]

    def run(q):
        hi = q.to(dt)
        lo = (q - hi.float()).to(dt)
        return fwd(hi).float() + fwd(lo).float() - z

    out = {}
    for mode in modes:
        if mode == FLOOR:
            yq = run(xf * FLOOR_C) / FLOOR_C
            out[mode] = (rel(yq, y), 0.0)
            continue
        fn, needs_group = MODES[mode]
        if needs_group and K % GROUP:
            continue
        q = fn(xf)
        out[mode] = (rel(run(q), y), rel(q, xf))
    return out


if __name__ == "__main__":
    # Reproduce check (a) for the modes act_calib.py uses.
    torch.manual_seed(0)
    T, N = 256, 1024
    cases = [
        ("in_proj-like K=5120, df=5, 6 outlier ch x40", 5120, 5.0, 6, 40.0),
        ("gate_up-like K=5120, df=5, 6 outlier ch x20", 5120, 5.0, 6, 20.0),
        ("down_proj-like K=17408, df=2.5, 16 outlier ch x30", 17408, 2.5, 16, 30.0),
        ("down_proj-like K=17408, df=2.5, no outlier ch", 17408, 2.5, 0, 1.0),
    ]
    for name, K, df, n_out, mag in cases:
        x = make_acts(T, K, df, n_out, mag, 1)
        W = torch.randn(K, N) / math.sqrt(K)
        y = x @ W
        print(f"\n## {name}")
        for mode, (fn, _) in MODES.items():
            print(f"  {mode:20s} act-GEMM rel err {100 * rel(fn(x) @ W, y):6.3f}%")
