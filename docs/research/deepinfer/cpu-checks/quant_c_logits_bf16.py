"""Check (c): what does rounding logits to bf16 (Marlin/vLLM lm_head output dtype = model dtype)
cost vs fp32 logits (DeepInfer sampler_utils.get_logits writes fp32)? Synthetic Qwen-like
logits over V=248,320: bulk ~N(mu, 2.5), a head of 32 plausible tokens, top logit placed at
`top` (absolute level matters: bf16 ulp = 0.0625 in [8,16), 0.125 in [16,32), 0.25 in [32,64))."""
import torch
torch.manual_seed(0)
V, B = 248320, 32
def make(top, gap_sd, mu):
    z = torch.randn(B, V, dtype=torch.float64) * 2.5 + mu
    head = top - torch.rand(B, 32, dtype=torch.float64).mul(gap_sd)  # head tokens within gap_sd of top
    head[:, 0] = top
    z[:, :32] = head
    return z
for temp in (1.0, 0.6):
    print(f"\n## temperature {temp}")
    for top, gap, mu in [(12, 3, -2), (20, 3, 2), (20, 8, 2), (28, 3, 8), (28, 10, 8), (40, 4, 15)]:
        z = make(top, gap, mu)
        zb = z.float().to(torch.bfloat16).double()
        lp = torch.log_softmax(z / temp, -1); lq = torch.log_softmax(zb / temp, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        H = -(lp.exp() * lp).sum(-1)
        flips = (z.argmax(-1) != zb.argmax(-1)).float().mean()
        print(f"  top {top:3d} head-spread {gap:3d} bulk-mean {mu:3d}: entropy {H.mean():5.2f} nats  "
              f"KL(fp32||bf16) mean {kl.mean():.2e} max {kl.max():.2e}  argmax flips {100*flips:.1f}%")
