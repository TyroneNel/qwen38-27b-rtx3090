"""CPU test for quant-3's GPTQ options in drafter/gptq_utils.py
(docs/research/deepinfer/quant-heads.md, quant-3; check (d) in cpu-checks/quant_d_gptq.py).

1. Defaults are bit-identical to the committed quantizer (`git show HEAD:drafter/gptq_utils.py`;
   skipped outside a git checkout), with and without --mse-clip, including a dead column.
2. signed + mse + act-order reproduces check (d)'s reference `gptq_static(..., "signed8mse",
   act_order=True)` and lowers the output error tr(E H E^T)/tr(W H W^T) vs the default.
3. The stored layout stays Marlin-compatible: codes in [-8, 7], one scale per contiguous group.

  OMP_NUM_THREADS=4 python bench/test_gptq_q3.py
"""
import importlib.util, os, subprocess, sys, tempfile
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "drafter"))
import gptq_utils  # noqa: E402


def gptq_static(W, H, group, act_order, percdamp=0.01, blocksize=128):
    """Check (d)'s reference (cpu-checks/quant_d_gptq.py, mode "signed8mse"), transcribed:
    that script runs its experiment on import."""
    W = W.float().clone(); N, K = W.shape
    g = W.view(N, K // group, group); amax = g.abs().amax(-1)
    sign = torch.where(g.max(-1).values > -g.min(-1).values, -1.0, 1.0)
    best = None
    for f in torch.linspace(0.75, 1.0, 11):
        s = sign * amax * f / 8
        e = ((torch.clamp(torch.round(g / s[..., None]), -8, 7) * s[..., None] - g) ** 2).sum(-1)
        if best is None: best, be = s.clone(), e
        else:
            m = e < be; best[m] = s[m]; be[m] = e[m]
    S = best.repeat_interleave(group, dim=1)
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

def make_layer(N=256, K=2048, n=4096, seed=0):
    """Same construction as check (d), smaller K to keep the test fast."""
    g = torch.Generator().manual_seed(seed)
    ch = torch.exp(torch.randn(K, generator=g) * 0.6)
    ch[torch.randperm(K, generator=g)[:12]] *= 25
    X = torch.distributions.StudentT(5.0).sample((n, K)) * ch
    X += (torch.randn(n, 16, generator=g) @ torch.randn(16, K, generator=g)) * 0.5 * ch
    H = 2 * X.t() @ X / n
    W = torch.distributions.StudentT(4.0).sample((N, K)) * torch.exp(torch.randn(N, 1, generator=g) * 0.3) * 0.02
    return W, H


def relerr(Wq, W, H):
    E = Wq - W
    return (torch.einsum("nk,kl,nl->", E, H, E) / torch.einsum("nk,kl,nl->", W, H, W)).item()


def head_module():
    try:
        src = subprocess.run(["git", "-C", REPO, "show", "HEAD:drafter/gptq_utils.py"],
                             check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    path = os.path.join(tempfile.mkdtemp(), "gptq_utils_head.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("gptq_utils_head", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    torch.manual_seed(0)
    ok = True
    head = head_module()
    for seed in (0, 1):
        W, H = make_layer(seed=seed)
        if seed == 1:                      # exercise the dead-column branch
            H[7, :] = 0; H[:, 7] = 0
        if head is None:
            print("SKIP default-vs-HEAD (no git checkout)")
        else:
            for kw in ({}, {"mse_clip": True}, {"bits": 8}):
                qa, sa = head.gptq_quantize(W, H, **kw)
                qb, sb = gptq_utils.gptq_quantize(W, H, **kw)
                same = torch.equal(qa, qb) and torch.equal(sa, sb)
                ok &= same
                print(f"seed {seed} defaults {kw or ''}: bit-identical to HEAD: {same}")

    for seed in (0, 1):
        W, H = make_layer(seed=seed)
        dq = lambda q, s: gptq_utils.dequant(q, s)
        base = relerr(dq(*gptq_utils.gptq_quantize(W, H)), W, H)
        rows = {"default (shipped)": base}
        for name, kw in [("mse_clip", dict(mse_clip=True)),
                         ("signed", dict(signed_scale=True)),
                         ("signed+mse", dict(signed_scale=True, mse_clip=True)),
                         ("static", dict(static_groups=True)),
                         ("act_order", dict(act_order=True)),
                         ("signed+mse+act_order (quant-3)", dict(signed_scale=True, mse_clip=True, act_order=True))]:
            q, s = gptq_utils.gptq_quantize(W, H, **kw)
            assert q.min() >= -8 and q.max() <= 7 and s.shape == (W.shape[0], W.shape[1] // 128)
            rows[name] = relerr(dq(q, s), W, H)
            if name.startswith("signed+mse+act"):
                ref = gptq_static(W, H, 128, act_order=True)
                diff = (dq(q, s) - ref).abs().max().item()
                match = diff < 1e-6
                ok &= match
                print(f"seed {seed} quant-3 vs check (d) reference: max |diff| {diff:.2e} -> {match}")
                gain = rows[name] < 0.75 * base
                ok &= gain
        for k, v in rows.items():
            print(f"  seed {seed} {k:34s} rel out err {v:.4e} ({100 * (v / base - 1):+6.1f}%)")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
