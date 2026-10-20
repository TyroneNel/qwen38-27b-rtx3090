"""Check (b): cost of HyperQwen's marlin-int8-negative-scales sign fold (code -8 -> clamp +7)
on AutoRound-style symmetric int4 g128, and of the int16 4096-level group-scale requant
(marlin_utils.marlin_act_int8_process_scales). AutoRound's quant_tensor_sym is reproduced
from memory (INFERRED, auto_round not installed here): maxq = 2^(b-1) = 8,
max_v = sign * max(|wmax|*max_scale, |wmin|*min_scale) with sign chosen so the extreme
element maps to code -8, scale = max_v / 8, q = clamp(round(w/scale + v), -8, 7)."""
import torch
torch.manual_seed(0)
G = 128

def autoround_sym(w, min_scale, max_scale, v):
    wmin = w.min(-1).values.clamp(max=0) * min_scale
    wmax = w.max(-1).values.clamp(min=0) * max_scale
    wmin_abs = -wmin
    max_v = (2 * (wmax < wmin_abs).int() - 1) * torch.maximum(wmax, wmin_abs)
    s = (max_v / 8).half().float()
    q = torch.clamp(torch.round(w / s[..., None] + v), -8, 7)
    return q, s

R = 8192  # groups (64 rows x 128 groups worth); weights heavy-tailed like LLM linears
for label, w in [("gaussian", torch.randn(R, G)),
                 ("student-t df=4", torch.distributions.StudentT(4.0).sample((R, G)))]:
    for tune_label, mins, maxs, v in [
        ("no minmax tuning, v=0", torch.ones(R), torch.ones(R), torch.zeros(R, G)),
        ("minmax tuned U(0.85,1), v~U(-.3,.3)", torch.empty(R).uniform_(0.85, 1), torch.empty(R).uniform_(0.85, 1),
         torch.empty(R, G).uniform_(-0.3, 0.3))]:
        q, s = autoround_sym(w, mins, maxs, v)
        neg = s < 0
        deq = q * s[..., None]
        # HyperQwen fold: s -> -s, q -> -q, -q == 8 clamped to 7
        qf = torch.where(neg[..., None], (-q).clamp(max=7), q)
        sf = s.abs()
        deq_fold = qf * sf[..., None]
        base_err = ((deq - w) ** 2).sum()
        fold_err = ((deq_fold - w) ** 2).sum()
        used8 = ((q == -8) & neg[..., None]).any(-1).float()
        n8 = ((q == -8) & neg[..., None]).sum().item()
        # output-error view: y = x.w with x ~ N(0,1) per element -> same as L2 of weight error; also
        # weight by salience (|w| large elements often matter more): report relative change only.
        # exact alternative: uint4 + zero point (code' = 15 - (q+8), zp = 7) -> lossless
        c = q + 8
        c2 = torch.where(neg[..., None], 15 - c, c); zp = torch.where(neg, 7.0, 8.0)
        deq_zp = (c2 - zp[..., None]) * sf[..., None]
        print(f"{label:16s} {tune_label:38s} neg-scale groups {100*neg.float().mean():5.1f}%  "
              f"neg groups using code -8 {100*used8[neg].mean():5.1f}%  clamped elems/neg-group {n8/neg.sum().item():.2f}  "
              f"weight-error energy +{100*(fold_err/base_err-1):5.2f}%  zp-remap max|diff| {(deq_zp-deq).abs().max():.1e}")

print("\n## int16 4096-level group-scale requant (s/s.max()*4096, round): relative scale error")
for ratio in [4, 16, 64, 256, 1024]:
    s = torch.exp(torch.empty(100000).uniform_(0, torch.log(torch.tensor(float(ratio)))))  # log-uniform max/min = ratio
    s = s / s.max()
    s16 = (s * 4096).round() / 4096
    re = ((s16 - s).abs() / s)
    print(f"  max/min scale ratio {ratio:5d}: mean rel err {100*re.mean():.4f}%  max {100*re.max():.3f}%  (int4 g128 RTN rel err ~ 1/(8*sqrt(12)) per elem ~ 3.6% of max)")
