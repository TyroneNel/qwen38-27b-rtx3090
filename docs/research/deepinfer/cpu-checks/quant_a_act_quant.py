"""Check (a): activation quantization error, HyperQwen per-token int8 vs DeepSeek-style
fine-grained scales vs Hadamard-128 rotation. Synthetic heavy-tailed activations with
outlier channels (no model weights available on this box). Metric: relative L2 error of
Y = X @ W (W random gaussian; weight quant error excluded on purpose, as in bench/act_calib.py)."""
import torch, math
torch.manual_seed(0)

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

def fp8_group(x, g):
    T, K = x.shape
    xg = x.view(T, K // g, g)
    s = xg.abs().amax(2, keepdim=True).clamp(min=1e-8) / 448
    s = torch.exp2(torch.ceil(torch.log2(s)))
    return ((xg / s).to(torch.float8_e4m3fn).float() * s).view(T, K)

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
    return ((a - b).norm() / b.norm()).item()

T, N = 256, 1024
H128 = hadamard(128)
cases = [
    ("in_proj-like K=5120, df=5, 6 outlier ch x40", 5120, 5.0, 6, 40.0),
    ("gate_up-like K=5120, df=5, 6 outlier ch x20", 5120, 5.0, 6, 20.0),
    ("down_proj-like K=17408, df=2.5, 16 outlier ch x30", 17408, 2.5, 16, 30.0),
    ("down_proj-like K=17408, df=2.5, no outlier ch", 17408, 2.5, 0, 1.0),
]
schemes = {
    "per-token int8 (HyperQwen now)": lambda x: int8_per_token(x),
    "per-token x per-128 int8, fp32 scale": lambda x: int8_group(x, 128),
    "per-token x per-128 int8, pow2 scale": lambda x: int8_group(x, 128, True),
    "per-token x per-32 int8, pow2 scale": lambda x: int8_group(x, 32, True),
    "per-token x per-32 fp8e4m3, pow2 (DeepSeek)": lambda x: fp8_group(x, 32),
    "Hadamard-128 + per-token int8": "had",
    "Hadamard-128 + per-128 int8": "had_g",
}
for name, K, df, n_out, mag in cases:
    x = make_acts(T, K, df, n_out, mag, 1)
    W = torch.randn(K, N) / math.sqrt(K)
    y = x @ W
    print(f"\n## {name}")
    for sn, f in schemes.items():
        if f == "had":
            xr = had_block(x, 128, H128); Wr = (W.view(K // 128, 128, N).transpose(1, 2) @ H128).transpose(1, 2).reshape(K, N)
            yq = int8_per_token(xr) @ Wr
        elif f == "had_g":
            xr = had_block(x, 128, H128); Wr = (W.view(K // 128, 128, N).transpose(1, 2) @ H128).transpose(1, 2).reshape(K, N)
            yq = int8_group(xr, 128) @ Wr
        else:
            yq = f(x) @ W
        print(f"  {sn:45s} act-GEMM rel err {100*rel(yq, y):6.3f}%")
