"""Run on the GPU box (CPU only, ~2 GB RAM): measure on the REAL AutoRound checkpoint what
patches/marlin-int8-negative-scales.patch costs and how coarse marlin_act_int8_process_scales'
int16 requant is. Reads compressed-tensors pack-quantized int4 tensors (contiguous packing,
8 values per int32, low nibble first, stored as q+8).
Usage: python3 -E -P real_ckpt_check.py models/Qwen3.8-27B-W4A16-AutoRound [regex]"""
import sys, glob, re, torch
from safetensors import safe_open
d = sys.argv[1]; pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"layers\.\d+\.")
G = 128
tot = dict(groups=0, neg=0, neg_m8=0, clamped=0, fold_e=0.0, w_e=0.0)
rows = []
for f in sorted(glob.glob(f"{d}/*.safetensors")):
    with safe_open(f, "pt") as h:
        keys = set(h.keys())
        for k in sorted(keys):
            if not k.endswith(".weight_packed") or not pat.search(k): continue
            base = k[: -len(".weight_packed")]
            if base + ".weight_scale" not in keys: continue
            p = h.get_tensor(k); s = h.get_tensor(base + ".weight_scale").float()
            if p.dtype != torch.int32 or s.shape[1] * G != p.shape[1] * 8: continue
            u = ((p.unsqueeze(-1) >> torch.arange(0, 32, 4, dtype=torch.int32)) & 0xF).reshape(p.shape[0], -1)
            q = (u - 8).float().view(p.shape[0], -1, G)
            neg = s < 0
            m8 = (q == -8) & neg[..., None]
            n_cl = m8.sum().item()
            fold_e = (s.pow(2)[..., None] * m8).sum().item()          # each clamped elem moves by |s|
            w_e = (q * s[..., None]).pow(2).sum().item()
            a = s.abs(); smax = a.max()
            s16 = torch.round(a / smax * 4096) / 4096 * smax
            req = ((s16 - a).abs() / a.clamp(min=1e-12)).mean().item()
            ratio = (smax / a.clamp(min=1e-12).min()).item()
            rows.append((base, neg.float().mean().item(), (m8.any(-1).sum() / neg.sum().clamp(min=1)).item(),
                         n_cl / max(neg.sum().item(), 1), fold_e / w_e, ratio, req))
            for kk, v in (("groups", s.numel()), ("neg", neg.sum().item()), ("neg_m8", m8.any(-1).sum().item()),
                          ("clamped", n_cl), ("fold_e", fold_e), ("w_e", w_e)):
                tot[kk] += v
print(f"{'tensor':60s} neg%   negW/-8%  clamp/neggrp  foldErr/W%   max/min|s|  int16 req err%")
for r in rows:
    print(f"{r[0][-60:]:60s} {100*r[1]:5.1f}  {100*r[2]:7.1f}  {r[3]:10.2f}  {100*r[4]:9.4f}  {r[5]:10.1f}  {100*r[6]:8.4f}")
print(f"\nTOTAL groups {tot['groups']}, negative {100*tot['neg']/max(tot['groups'],1):.1f}%, negative groups using code -8 "
      f"{100*tot['neg_m8']/max(tot['neg'],1):.1f}%, clamped elems {tot['clamped']}, fold error energy / weight energy "
      f"{100*tot['fold_e']/max(tot['w_e'],1e-30):.4f}%")
print("Compare: int4 RTN error energy / weight energy is typically ~1-2% -> fold adds fold/RTN of that.")
