"""quant-4 feasibility (CPU): what does re-quantizing a ROTATED int4 down_proj cost, when the only
source is the int4 checkpoint itself?

quant-4 rotates the K axis of a linear (W' = W.H, H = blockdiag(H128), symmetric orthonormal) so
that per-token int8 activations quantize x' = x.H instead of x. W.H is off the int4 grid, so it
must be re-quantized; with no bf16 source that is a SECOND int4 rounding on top of AutoRound's.
This script measures, on real down_proj weights (dequantized int4, the served -fast checkpoint)
and proxy activations, the relative output error vs today's W4A16 output y = X.Wq^T of:

  a_pt     today's INT8_ACT: per-token int8(X) . Wq
  a_h      Hadamard act only (exact W', no requant): per-token int8(X.H) . (Wq.H)
  w_*      requant only, bf16 acts: (X.H) . Q(Wq.H)   for RTN, shipped GPTQ, quant-3 best GPTQ
  tot_*    the quant-4 arm: per-token int8(X.H) . Q(Wq.H)

quant-4 pays off only if tot_* is well below a_pt. Hessians come from a calibration half of the
tokens, errors from the other half. Activations are proxies (no real captures exist on this box):
  syn   check (a)'s down_proj-like synthetic (t2.5, 16 outlier channels x30)
  mlp   silu(gate).up of the layer's REAL gate/up int4 weights on an RMSNormed synthetic residual
        (t5 tokens, log-normal channel scales, 4 massive channels)
  mlpo  mlp plus persistent hot neurons (--mlpo 'n,mag': n channels of the down_proj input x mag),
        tuned so a_pt and a_h both land on the GPU-measured values
Each proxy's a_pt / a_h is printed next to the GPU-measured act_calib values of the same layer
(quant-1-calib) so its realism can be judged.

Usage (CPU, 4 threads, ~1 min per layer x proxy; quant4_requant.out used --tokens 8192):
  python quant4_requant.py <model dir> [--layers 8,31,55,63] [--rows 1024] [--tokens 8192]
         [--proxies syn,mlp,mlpo] [--mlpo 16,12] [--probe] [--calib act_calib_modes.json]
"""
import argparse, json, math, os, sys, time
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "drafter"))
from gptq_utils import gptq_quantize, rtn_quantize, dequant  # noqa: E402

G = 128


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


H128 = hadamard(G)


def rot(x):                      # x [T, K] -> x . blockdiag(H128)  (also W -> W.H for W [N, K])
    T, K = x.shape
    return (x.view(T, K // G, G) @ H128).view(T, K)


def int8_pt(x):
    s = x.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
    return (x / s).round().clamp(-127, 127) * s


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def get(self, name):
        with safe_open(os.path.join(self.d, self.idx[name]), "pt") as h:
            return h.get_tensor(name)

    def int4(self, base):        # compressed-tensors pack-quantized: 8 nibbles/int32, low first, q+8
        p = self.get(base + ".weight_packed")
        s = self.get(base + ".weight_scale").float()
        u = ((p.unsqueeze(-1) >> torch.arange(0, 32, 4, dtype=torch.int32)) & 0xF).reshape(p.shape[0], -1)
        q = (u - 8).float()
        return (q.view(q.shape[0], -1, G) * s[..., None]).view(q.shape[0], -1)


def acts_syn(T, K, seed):
    g = torch.Generator().manual_seed(seed)
    ch = torch.exp(torch.randn(K, generator=g) * 0.5)
    ch[torch.randperm(K, generator=g)[:16]] *= 30.0
    torch.manual_seed(seed)
    return torch.distributions.StudentT(2.5).sample((T, K)) * ch


def acts_mlp(ck, L, T, seed, df=5.0, sigma=0.5, hot=0, mag=1.0):
    pre = f"model.language_model.layers.{L}."
    g = torch.Generator().manual_seed(seed)
    D = 5120
    ch = torch.exp(torch.randn(D, generator=g) * sigma)
    torch.manual_seed(seed)
    h = torch.distributions.StudentT(df).sample((T, D)) * ch
    mass = torch.randperm(D, generator=g)[:4]
    h[:, mass] += 100.0 * torch.sign(torch.randn(4, generator=g))           # massive residual dims
    w = ck.get(pre + "post_attention_layernorm.weight").float()
    w = 1.0 + w if w.abs().mean() < 0.5 else w                              # zero-centred (1+w) norm
    hn = h * torch.rsqrt(h.pow(2).mean(1, keepdim=True) + 1e-6) * w
    gate = hn @ ck.int4(pre + "mlp.gate_proj").t()
    up = hn @ ck.int4(pre + "mlp.up_proj").t()
    x = torch.nn.functional.silu(gate) * up
    if hot:
        x[:, torch.randperm(x.shape[1], generator=g)[:hot]] *= mag
    return x


def run_layer(ck, L, X, Wq, real, tag, probe=False):
    T = X.shape[0]
    Xc, Xe = X[: T // 2], X[T // 2:]
    Xcr, Xer = rot(Xc), rot(Xe)
    Wr = rot(Wq)                                           # W' = Wq . H
    y = Xe @ Wq.t()
    assert rel(Xer @ Wr.t(), y) < 1e-4, "rotation is not exact"
    qx_pt, qx_h = int8_pt(Xe), int8_pt(Xer)
    r = {"a_pt": rel(qx_pt @ Wq.t(), y), "a_h": rel(qx_h @ Wr.t(), y)}
    if probe:
        print(f"L{L:02d} {tag:4s} a_pt {r['a_pt']:.4f} a_h {r['a_h']:.4f}" +
              ("" if real is None else f" | GPU act_calib per-token {real[0]:.4f} hadamard {real[1]:.4f}"), flush=True)
        return r
    Hs = Xcr.t() @ Xcr * (2.0 / Xcr.shape[0])              # rotated Hessian H^T X^T X H
    t0 = time.time()
    variants = {
        "rtn": lambda: dequant(*rtn_quantize(Wr)),
        "gptq": lambda: dequant(*gptq_quantize(Wr, Hs)),
        "gptq_best": lambda: dequant(*gptq_quantize(Wr, Hs, signed_scale=True, mse_clip=True, act_order=True)),
    }
    for name, fn in variants.items():
        Wn = fn()
        r["w_" + name] = rel(Xer @ Wn.t(), y)
        r["tot_" + name] = rel(qx_h @ Wn.t(), y)
        r["werr_" + name] = rel(Wn, Wr)                    # plain weight-space error of the 2nd rounding
    r["sec"] = round(time.time() - t0, 1)
    rr = "" if real is None else f" | GPU act_calib per-token {real[0]:.4f} hadamard {real[1]:.4f}"
    print(f"L{L:02d} {tag:4s} a_pt {r['a_pt']:.4f} a_h {r['a_h']:.4f}{rr}")
    print("        " + "  ".join(f"{v}: w {r['w_'+v]:.4f} tot {r['tot_'+v]:.4f} (wspace {r['werr_'+v]:.3f})"
                                for v in variants) + f"  [{r['sec']}s]", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", default="8,31,55,63")
    ap.add_argument("--rows", type=int, default=1024, help="random down_proj output rows (GPTQ is row-separable)")
    ap.add_argument("--tokens", type=int, default=8192, help="half build the Hessian, half measure")
    ap.add_argument("--calib", default=None, help="quant-1-calib act_calib_modes.json, for side-by-side real numbers")
    ap.add_argument("--proxies", default="syn,mlp,mlpo", help="syn, mlp, mlpo (mlp + hot neurons, --mlpo)")
    ap.add_argument("--mlpo", default="16,12", help="mlpo hot neurons 'count,magnitude' (16,12 matches L08 on GPU)")
    ap.add_argument("--probe", action="store_true", help="activation errors only (tune proxies), no requant")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    ck = Ckpt(a.model)
    real = {}
    if a.calib:
        for k, v in json.load(open(a.calib))["layers"].items():
            if k.endswith("down_proj"):
                real[int(k.split(".")[-3])] = (v["per-token"]["out_err"], v["hadamard-128"]["out_err"])
    res = {}
    for L in [int(x) for x in a.layers.split(",")]:
        Wq = ck.int4(f"model.language_model.layers.{L}.mlp.down_proj")
        rows = torch.randperm(Wq.shape[0])[: a.rows]
        Wq = Wq[rows].contiguous()
        nh, mg = a.mlpo.split(",")
        make = {"syn": lambda: acts_syn(a.tokens, Wq.shape[1], L),
                "mlp": lambda: acts_mlp(ck, L, a.tokens, L),
                "mlpo": lambda: acts_mlp(ck, L, a.tokens, L, hot=int(nh), mag=float(mg))}
        for tag in a.proxies.split(","):
            res[f"{L}/{tag}"] = run_layer(ck, L, make[tag](), Wq, real.get(L), tag, a.probe)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    keys = [k for k in next(iter(res.values())) if k != "sec"]
    print("\nmedian over layers x proxies:")
    for k in keys:
        vals = sorted(v[k] for v in res.values())
        print(f"  {k:14s} {vals[len(vals) // 2]:.4f}  (min {vals[0]:.4f}, max {vals[-1]:.4f})")


if __name__ == "__main__":
    main()
