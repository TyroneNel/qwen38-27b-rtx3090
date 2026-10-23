"""CPU check (attn-5, attn-3): per-token-head KV quantisation with and without a Hadamard rotation, on activations
built from the REAL Qwen3.8-27B attention weights; plus attn-3's int8-Q error on the same data.

attn_kvq_error.py (synthetic regimes A/B) found that a 256-point Hadamard cuts the int8 per-token-head output error
0.069 -> 0.027 and int4 0.84 -> 0.47. This re-runs the comparison on data shaped by the checkpoint, for each of the
16 full-attention layers:
  x       token inputs after input_layernorm: N(0, 1) per channel times (1 + gamma_in) (GemmaRMSNorm weight).
          ASSUMPTION: real residual streams have a few massive channels; isotropic x does not, so the K/V channel
          structure here comes from the weights (W_k, W_v, q/k norm gains), not from activation outliers.
  q, k    q_proj (the q half of each head's [q | gate] rows) and k_proj, dequantised from the pack-quantized
          int4 checkpoint (sym, group 128), per-head RMSNorm times (1 + gamma) (q_norm / k_norm), then neox RoPE
          on the first 64 dims (partial_rotary_factor 0.25, theta 1e7) at positions 0..L-1.
  v       v_proj, no norm.
  queries G = 6 per KV head, the LAST token's position. Regime N: their own x (diffuse attention). Regime R: x of
          4 random earlier tokens averaged plus 0.3 noise (retrieval-like, peaky rows).
Formats (vLLM's writer: scale = fp32 absmax / qmax, round half away from zero, clamp): int8_pth (qmax 127, the D arm),
int4_pth (qmax 7). Frames: none; +H the orthonormal 256-pt Sylvester Hadamard (attn5-kv-hadamard.patch: K H and V H
in the cache, q H, out H); +H/bf16 the same with the patch's bf16 roundings (q H, K H, V H and out H rounded to
bf16, H exact in bf16); +RHT random signs then H (vLLM's int4 mode, single_rht). Metrics against fp64 attention
on unquantised q/k/v: K rel-L2, logit RMSE (logit units after 1/sqrt(D)), output rel-L2. Last column (attn-3):
output rel-L2 of int8_pth K/V with Q quantised per row to int8 by absmax (pthfa stage_q_s8), no rotation.

Run: python3 -I attn5_hadamard_kv.py [checkpoint dir]   (default /mnt/g/models/Qwen3.8-27B-W4A16-AutoRound-fast)
CPU only, torch, 4 threads; ~2-4 min.
"""
import json
import math
import os
import sys

import torch
from safetensors import safe_open

torch.set_num_threads(4)
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/g/models/Qwen3.8-27B-W4A16-AutoRound-fast"
L, G = 4096, 6
GEN = torch.Generator().manual_seed(0)


def load(names):
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for n in names:
        with safe_open(os.path.join(CKPT, idx[n]), "pt") as f:
            out[n] = f.get_tensor(n)
    return out


def dequant(t, pre):
    """compressed-tensors pack-quantized, symmetric int4, group 128: 8 nibbles per int32 along the input dim."""
    p, s = t[pre + ".weight_packed"], t[pre + ".weight_scale"].float()
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    w = ((p.unsqueeze(-1) >> shifts) & 0xF).reshape(p.shape[0], -1).float() - 8.0
    return w * s.repeat_interleave(w.shape[1] // s.shape[1], dim=1)


def rmsnorm_gemma(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())


def rope(x, pos, rot=64, theta=1e7):
    half = rot // 2
    inv = theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    ang = pos[:, None].double() * inv[None, :]
    c, s = ang.cos(), ang.sin()
    while c.dim() < x.dim():
        c, s = c.unsqueeze(1), s.unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:rot]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., rot:]], -1)


def hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def quant_pth(x, qmax):
    """vLLM's per-token-head writer, emulated: fp32 scale, round half away from zero, clamp [-qmax-1, qmax]."""
    x32 = x.float()
    sc = (x32.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-6)
    q = x32 * (1.0 / sc)
    q = torch.where(q >= 0, q + 0.5, q - 0.5).trunc().clamp(-qmax - 1, qmax)
    return (q * sc).double()


def q_int8_rows(q):
    """pthfa stage_q_s8: q8 = rn(q * 127 / absmax), clamp +-127; dequantised."""
    x = q.float()
    amax = x.abs().amax(-1, keepdim=True)
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    return (torch.round(x * inv).clamp(-127, 127) * (amax / 127.0)).double()


def bf16(x):
    return x.to(torch.bfloat16).double()


def attn(q, k, v):
    """q [G, D], k/v [L, D]: the last position sees every key."""
    s = q @ k.T / math.sqrt(k.shape[-1])
    return s, torch.softmax(s, -1) @ v


def main():
    cfg = json.load(open(os.path.join(CKPT, "config.json")))
    t = cfg.get("text_config", cfg)
    D, HQ, HKV = t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"]
    layers = [i for i, lt in enumerate(t["layer_types"]) if lt == "full_attention"]
    H = hadamard(D)
    signs = torch.where(torch.rand(D, generator=GEN) < 0.5, -1.0, 1.0).double()
    RHT = torch.diag(signs) @ H
    pos = torch.arange(L)
    print(f"checkpoint {CKPT}: D={D} Hq={HQ} Hkv={HKV}, full-attention layers {layers}; L={L}, G={G}, "
          f"kv head 0 of each layer")
    print("frames: none | +H | +H with the patch's bf16 roundings | +RHT (int4: vLLM's own); metric = output rel-L2"
          " (K rel-L2, logit RMSE in brackets for int8 none/+H); last column attn-3: int8 KV + int8 Q rows")
    hdr = (f"{'layer':>5s} {'rg':>2s} {'logit sd':>8s} | {'int8':>7s} {'+H':>7s} {'+H bf16':>7s} | "
           f"{'int4':>7s} {'+H':>7s} {'+RHT':>7s} | {'K8 rel':>15s} {'logit8':>15s} | {'+Q8 (a3)':>8s}")
    print(hdr)
    agg = {}
    for li in layers:
        pre = f"model.language_model.layers.{li}."
        names = [pre + "input_layernorm.weight", pre + "self_attn.q_norm.weight", pre + "self_attn.k_norm.weight"]
        for pj in ("q_proj", "k_proj", "v_proj"):
            names += [pre + f"self_attn.{pj}.weight_packed", pre + f"self_attn.{pj}.weight_scale"]
        w = load(names)
        wq = dequant(w, pre + "self_attn.q_proj").view(HQ, 2 * D, -1)[:G, :D].reshape(G * D, -1)  # heads 0..5
        wk = dequant(w, pre + "self_attn.k_proj")[:D]   # kv head 0
        wv = dequant(w, pre + "self_attn.v_proj")[:D]
        gin = w[pre + "input_layernorm.weight"].float()
        x = torch.randn(L, gin.shape[0], generator=GEN) * (1.0 + gin)
        k = rope(rmsnorm_gemma((x @ wk.T).double(), w[pre + "self_attn.k_norm.weight"]), pos)
        v = (x @ wv.T).double()
        for rg in ("N", "R"):
            if rg == "N":
                xq = torch.randn(1, gin.shape[0], generator=GEN) * (1.0 + gin)
            else:
                tgt = torch.randint(0, L - 1, (4,), generator=GEN)
                xq = x[tgt].mean(0, keepdim=True) + 0.3 * torch.randn(1, gin.shape[0], generator=GEN) * (1.0 + gin)
            q = (xq @ wq.T).view(G, D).double()
            q = rope(rmsnorm_gemma(q, w[pre + "self_attn.q_norm.weight"]), pos[-1:].expand(G))
            s0, o0 = attn(q, k, v)
            n0 = o0.norm()

            def run(qm, kq, vq):
                s, o = attn(qm, kq, vq)
                return (o - o0).norm().item() / n0.item(), math.sqrt(((s - s0) ** 2).mean().item())

            row = {}
            for name, qmax in (("int8", 127), ("int4", 7)):
                kq, vq = quant_pth(k, qmax), quant_pth(v, qmax)
                row[name] = run(q, kq, vq)
                row[name + "_krel"] = ((kq - k).norm() / k.norm()).item()
                kh, vh = quant_pth(k @ H, qmax), quant_pth(v @ H, qmax)
                row[name + "+H"] = run(q, kh @ H.T, vh @ H.T)
                row[name + "+H_krel"] = ((kh @ H.T - k).norm() / k.norm()).item()
                # the patch: K H, V H, q H rounded to bf16 before use; out_rot rounded to bf16, then out H to bf16
                kb, vb, qb = quant_pth(bf16(k @ H), qmax), quant_pth(bf16(v @ H), qmax), bf16(q @ H)
                sb = qb @ kb.T / math.sqrt(D)
                ob = bf16(bf16(torch.softmax(sb, -1) @ vb) @ H)
                row[name + "+Hb"] = ((ob - o0).norm().item() / n0.item(), math.sqrt(((sb - s0) ** 2).mean().item()))
                kr, vr = quant_pth(k @ RHT, qmax), quant_pth(v @ RHT, qmax)
                row[name + "+RHT"] = run(q, kr @ RHT.T, vr @ RHT.T)
            row["a3"] = run(q_int8_rows(q), quant_pth(k, 127), quant_pth(v, 127))
            for kk_, vv_ in row.items():
                agg.setdefault((rg, kk_), []).append(vv_)
            print(f"{li:5d} {rg:>2s} {s0.std().item():8.2f} | {row['int8'][0]:7.4f} {row['int8+H'][0]:7.4f} "
                  f"{row['int8+Hb'][0]:7.4f} | {row['int4'][0]:7.4f} {row['int4+H'][0]:7.4f} {row['int4+RHT'][0]:7.4f} | "
                  f"{row['int8_krel']:7.4f}/{row['int8+H_krel']:7.4f} {row['int8'][1]:7.4f}/{row['int8+H'][1]:7.4f} | "
                  f"{row['a3'][0]:8.4f}", flush=True)
        del w, wq, wk, wv, x, k, v

    def med(xs):
        xs = sorted(x[0] if isinstance(x, tuple) else x for x in xs)
        return xs[len(xs) // 2]
    print("median over layers (output rel-L2):")
    for rg in ("N", "R"):
        g = lambda key: med(agg[(rg, key)])
        print(f"  regime {rg}: int8 {g('int8'):.4f} -> +H {g('int8+H'):.4f} (+H with bf16 {g('int8+Hb'):.4f}); "
              f"int4 {g('int4'):.4f} -> +H {g('int4+H'):.4f}, +RHT {g('int4+RHT'):.4f}; "
              f"attn-3 int8 KV + int8 Q {g('a3'):.4f}")
        r8 = sum(1 for a, b in zip(agg[(rg, 'int8')], agg[(rg, 'int8+Hb')]) if b[0] < a[0])
        print(f"            layers where +H (bf16) beats unrotated int8: {r8}/{len(agg[(rg, 'int8')])}")


if __name__ == "__main__":
    # self-checks: H orthonormal and symmetric; the RoPE keeps norms; the writer emulation is exact on the grid
    Hc = hadamard(256)
    assert torch.allclose(Hc @ Hc, torch.eye(256, dtype=torch.float64), atol=1e-12) and torch.equal(Hc, Hc.T)
    xx = torch.randn(3, 2, 256, dtype=torch.float64)
    assert torch.allclose(rope(xx, torch.tensor([0, 5, 99])).norm(dim=-1), xx.norm(dim=-1))
    g = torch.tensor([[127.0, -64.0, 3.0]])
    assert torch.equal(quant_pth(g, 127), g.double())
    main()
