"""Check (d): GPTQ variants for HyperQwen's int4 g128 requants (lm_head, MTP, DFlash2:
drafter/gptq_utils.py, imported from the repo).
Synthetic layer: W [256, 5120] heavy-tailed, Hessian from heavy-tailed activations with
outlier channels and a low-rank correlated part. Metric: relative output error
tr(E H E^T) / tr(W H W^T)."""
import torch, math
torch.manual_seed(0)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "drafter"))
from gptq_utils import gptq_quantize

def make_layer(N=256, K=5120, n=8192, seed=0):
    g = torch.Generator().manual_seed(seed)
    ch = torch.exp(torch.randn(K, generator=g) * 0.6)
    ch[torch.randperm(K, generator=g)[:12]] *= 25
    X = torch.distributions.StudentT(5.0).sample((n, K)) * ch
    X += (torch.randn(n, 16, generator=g) @ torch.randn(16, K, generator=g)) * 0.5 * ch
    H = 2 * X.t() @ X / n
    W = torch.distributions.StudentT(4.0).sample((N, K)) * torch.exp(torch.randn(N, 1, generator=g) * 0.3) * 0.02
    return W, H

def static_scales(W, group, mode):
    N, K = W.shape
    g = W.view(N, K // group, group)
    amax = g.abs().amax(-1)
    if mode == "amax7":
        return amax / 7
    if mode == "signed8":  # AutoRound-style: extreme element maps to code -8
        sign = torch.where(g.max(-1).values > -g.min(-1).values, -1.0, 1.0)
        return sign * amax / 8
    if mode in ("mse7", "signed8mse"):
        best = None
        for f in torch.linspace(0.75, 1.0, 11):
            s = (amax * f / 7) if mode == "mse7" else torch.where(g.max(-1).values > -g.min(-1).values, -1.0, 1.0) * amax * f / 8
            q = torch.clamp(torch.round(g / s[..., None]), -8, 7) * s[..., None]
            e = ((q - g) ** 2).sum(-1)
            if best is None: best, be = s.clone(), e
            else:
                m = e < be; best[m] = s[m]; be[m] = e[m]
        return best
    raise ValueError(mode)

@torch.no_grad()
def gptq_static(W, H, group, mode, act_order, percdamp=0.01, blocksize=128):
    W = W.float().clone(); N, K = W.shape
    S = static_scales(W, group, mode).repeat_interleave(group, dim=1)  # [N,K], per column scale
    H = H.clone(); H += torch.eye(K) * percdamp * torch.diag(H).mean()
    perm = torch.argsort(torch.diag(H), descending=True) if act_order else torch.arange(K)
    W, S, H = W[:, perm], S[:, perm], H[perm][:, perm]
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = i1 + blocksize
        W1 = W[:, i1:i2].clone(); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            s = S[:, i1 + i]
            q = torch.clamp(torch.round(W1[:, i] / s), -8, 7) * s
            Q[:, i1 + i] = q
            err = (W1[:, i] - q) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    out = torch.empty_like(Q); out[:, perm] = Q
    return out

def rtn(W, group, mode):
    N, K = W.shape
    S = static_scales(W, group, mode).repeat_interleave(group, dim=1)
    return torch.clamp(torch.round(W / S), -8, 7) * S

def relerr(Wq, W, H):
    E = Wq - W
    return (torch.einsum("nk,kl,nl->", E, H, E) / torch.einsum("nk,kl,nl->", W, H, W)).item()

for seed in (0, 1):
    W, H = make_layer(seed=seed)
    res = {}
    res["RTN g128 amax/7"] = relerr(rtn(W, 128, "amax7"), W, H)
    res["RTN g128 signed amax/8 (AutoRound-style)"] = relerr(rtn(W, 128, "signed8"), W, H)
    q, s = gptq_quantize(W, H); res["HQ gptq_quantize (dynamic amax/7) [shipped]"] = relerr((q.float().view(256, -1, 128) * s[..., None]).view(256, -1), W, H)
    q, s = gptq_quantize(W, H, mse_clip=True); res["HQ gptq_quantize --mse-clip"] = relerr((q.float().view(256, -1, 128) * s[..., None]).view(256, -1), W, H)
    res["GPTQ static amax/7"] = relerr(gptq_static(W, H, 128, "amax7", False), W, H)
    res["GPTQ static signed amax/8"] = relerr(gptq_static(W, H, 128, "signed8", False), W, H)
    res["GPTQ static signed amax/8 + mse"] = relerr(gptq_static(W, H, 128, "signed8mse", False), W, H)
    res["GPTQ static amax/7 + act-order"] = relerr(gptq_static(W, H, 128, "amax7", True), W, H)
    res["GPTQ static signed amax/8 + mse + act-order"] = relerr(gptq_static(W, H, 128, "signed8mse", True), W, H)
    res["GPTQ per-channel (group=K) amax/7 + act-order"] = relerr(gptq_static(W, H, 5120, "mse7", True), W, H)
    print(f"\n## seed {seed}")
    base = res["HQ gptq_quantize (dynamic amax/7) [shipped]"]
    for k, v in res.items():
        print(f"  {k:48s} rel out err {v:.4e}  ({100*(v/base-1):+6.1f}% vs shipped)")
