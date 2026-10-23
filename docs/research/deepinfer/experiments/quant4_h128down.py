"""quant-4 (docs/research/deepinfer/quant-heads.md): build a block-Hadamard-128 rotated variant of
an int4 model dir, for the online rotation of patches/quant4-act-hadamard.patch.

For every linear matching --layers-re (default: the 64 body mlp.down_proj), with H = the +-1
Hadamard-128 (H.H = 128 I) applied block-diagonally along K:
    stored weight  W' = GPTQ_int4( W . H / 16 )        W = the source's dequantized int4 weight
    online input   x' = x . H / 8                      (the patch, VLLM_ACT_HADAMARD_RE)
so x' . W'^T == x . W^T up to the second int4 rounding. There is no bf16 source on this box, so
W is the int4 checkpoint itself and the requant is a SECOND rounding on top of AutoRound's; the
report measures what that costs. GPTQ is drafter/gptq_utils.py's quant-3 best variant (signed
/8 scales, MSE clip, static-group act-order), on the Hessian of the rotated captured inputs.

Inputs come from bench/act_calib.py --dump-inputs (L<idx>.<suffix>.npy, bf16 bits). Rows are
split 75/25: the Hessian uses the first part, the report the held-out rest. Per layer it
prints, as relative output error against today's W4A16 output y = x . W^T:
    a_pt   today's INT8_ACT         per-token int8(x) . W
    a_h    rotation, exact weights  per-token int8(x') . (W.H/16)
    w      requant only, bf16 x'    x' . W'
    tot    the quant-4 arm          per-token int8(x') . W'
and one QUANT4 line with the medians and gate=yes when median(a_pt / tot) >= --gate (1.5).

Writes a NEW directory: the shards holding target tensors are rewritten (all their other
tensors copied verbatim), every other weight shard is hardlinked, small files are copied, and
quant4_hadamard.json records the knob value and the per-layer report. --report-only writes
nothing but the report JSON (--out), and --rows N quantizes only N random output rows per
layer (GPTQ is row-separable), for a quick CPU gate.

  # GPU (no vLLM in the process), ~10-20 min for 64 layers:
  python quant4_h128down.py /app/models/Qwen3.8-27B-W4A16-AutoRound-fast \\
      /app/models/Qwen3.8-27B-W4A16-AutoRound-fast-h128down --acts /out/q4dump --out /out/q4_report.json
  # CPU gate, 1024 rows x 64 layers, ~30 min at 4 threads:
  python quant4_h128down.py <src> - --acts <dump> --report-only --rows 1024 --device cpu --out report.json
"""
import argparse
import json
import os
import re
import shutil
import statistics
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(REPO, "drafter"))
from gptq_utils import gptq_quantize  # noqa: E402

G = 128
ACT_SCALE, W_SCALE = 1 / 8, 1 / 16          # ACT_SCALE * W_SCALE * 128 == 1; both exact in bf16/fp16
# vLLM prefix form of the default --layers-re. Anchored on language_model. so a W4A16 drafter's own
# model.layers.N.mlp.down_proj (DFlash2, SPEC=dflash2) and the mtp.* layers never match.
KNOB_RE = r"^language_model\.model\.layers\.\d+\.mlp\.down_proj$"
COPY_NOT_LINK = ("model_extra_tensors.safetensors", "mtp_draft_vocab_ids.pt")


def hadamard_pm1(n, device):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device)


def rot(x, H):                               # x [R, K] -> x . blockdiag(H)
    R, K = x.shape
    return (x.view(R, K // G, G) @ H).view(R, K)


def int8_pt(x):
    s = x.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
    return (x / s).round().clamp(-127, 127) * s


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def unpack4(p):                              # [N, K/8] int32 -> [N, K] int codes in [-8, 7]
    u = (p.unsqueeze(-1) >> torch.arange(0, 32, 4, dtype=torch.int32, device=p.device)) & 0xF
    return u.reshape(p.shape[0], -1) - 8


def load_acts(path, rows, device):
    a = np.load(path, mmap_mode="r")[:rows]
    x = torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16)
    return x.to(device).float()


def requant(W, X, H, out_rows, seed):
    """W [N, K] fp32 dequantized source weight, X [T, K] fp32 captured inputs.
    Returns (q int8 [n, K], scale fp16 [n, K/G], report dict)."""
    g = torch.Generator().manual_seed(seed)
    T = X.shape[0]
    perm = torch.randperm(T, generator=g).to(X.device)
    nc = (3 * T) // 4
    Xr = rot(X, H) * ACT_SCALE
    Xc, Xe, Xre = Xr[perm[:nc]], X[perm[nc:]], Xr[perm[nc:]]
    if out_rows:
        W = W[torch.randperm(W.shape[0], generator=g)[:out_rows].to(W.device)]
    Wr = rot(W, H) * W_SCALE
    Hs = Xc.t() @ Xc * (2.0 / nc)
    del Xr, Xc
    q, s = gptq_quantize(Wr, Hs, signed_scale=True, mse_clip=True, act_order=True)
    del Hs
    s16 = s.to(torch.float16)
    Wn = (q.float().view(q.shape[0], -1, G) * s16.float()[..., None]).view_as(Wr)
    y = Xe @ W.t()
    qx, qxr = int8_pt(Xe), int8_pt(Xre)
    r = {"a_pt": rel(qx @ W.t(), y), "a_h": rel(qxr @ Wr.t(), y), "w": rel(Xre @ Wn.t(), y),
         "tot": rel(qxr @ Wn.t(), y), "wspace": rel(Wn, Wr), "calib_rows": nc, "eval_rows": T - nc}
    return q, s16, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", help="new model dir ('-' with --report-only)")
    ap.add_argument("--acts", required=True, help="bench/act_calib.py --dump-inputs directory")
    ap.add_argument("--layers-re", default=r"^model\.language_model\.layers\.\d+\.mlp\.down_proj$",
                    help="checkpoint tensor bases to rotate")
    ap.add_argument("--knob-re", default=KNOB_RE, help="VLLM_ACT_HADAMARD_RE value recorded for these layers")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--rows", type=int, default=0, help="with --report-only: random output rows per layer (0 = all)")
    ap.add_argument("--gate", type=float, default=1.5)
    ap.add_argument("--out", default=None, help="report JSON path")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.rows and not a.report_only:
        ap.error("--rows only makes sense with --report-only")
    src = a.src.rstrip("/") + "/"
    dst = None if a.report_only else a.dst.rstrip("/") + "/"
    if dst and os.path.exists(dst):
        sys.exit(f"{dst} exists; refusing to touch it (remove it yourself to rebuild).")
    meta = json.load(open(os.path.join(a.acts, "meta.json")))
    wm = json.load(open(src + "model.safetensors.index.json"))["weight_map"]
    lre = re.compile(a.layers_re)
    targets = sorted({k[: -len(".weight_packed")] for k in wm if k.endswith(".weight_packed")
                      and lre.search(k[: -len(".weight_packed")])})
    if not targets:
        sys.exit(f"no packed tensor matches {a.layers_re}")
    key = {}
    for b in targets:
        mi = re.search(r"layers\.(\d+)\.(.*)$", b)
        k = f"L{int(mi.group(1)):02d}.{mi.group(2)}"
        if k not in meta["layers"]:
            sys.exit(f"{b}: no capture {k}.npy in {a.acts}")
        key[b] = k
    by_shard = {}
    for b in targets:
        by_shard.setdefault(wm[b + ".weight_packed"], []).append(b)
    print(f"{len(targets)} tensors in {len(by_shard)} shards, device {a.device}", flush=True)
    if not a.report_only:
        from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
    H = hadamard_pm1(G, a.device)
    report = {}
    if dst:
        os.makedirs(dst)
    for shard, bases in sorted(by_shard.items()):
        with safe_open(src + shard, "pt") as f:
            smeta = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()} if dst else None
            for b in sorted(bases, key=lambda s: int(re.search(r"layers\.(\d+)\.", s).group(1))):
                t0 = time.time()
                p = f.get_tensor(b + ".weight_packed").to(a.device)
                sc = f.get_tensor(b + ".weight_scale").to(a.device).float()
                W = (unpack4(p).float().view(p.shape[0], -1, G) * sc[..., None]).view(p.shape[0], -1)
                X = load_acts(os.path.join(a.acts, key[b] + ".npy"), meta["layers"][key[b]]["rows"], a.device)
                assert X.shape[1] == W.shape[1], (b, X.shape, W.shape)
                q, s16, r = requant(W, X, H, a.rows, seed=len(report))
                del W, X
                r["sec"] = round(time.time() - t0, 1)
                report[b] = r
                print(f"{key[b]:18s} a_pt {r['a_pt']:.4f}  a_h {r['a_h']:.4f}  w {r['w']:.4f}  tot {r['tot']:.4f}"
                      f"  (a_pt/tot {r['a_pt'] / r['tot']:.2f}x, weight-space {r['wspace']:.3f})  [{r['sec']}s]",
                      flush=True)
                if dst:
                    pk = pack_to_int32(q.cpu(), 4, packed_dim=1).contiguous()
                    assert torch.equal(unpack4(pk), q.cpu().to(torch.int32)), f"{b}: pack round-trip"
                    assert pk.shape == tensors[b + ".weight_packed"].shape, (b, pk.shape)
                    tensors[b + ".weight_packed"] = pk
                    tensors[b + ".weight_scale"] = s16.cpu().contiguous()
                del q, s16
        if dst:
            save_file(tensors, dst + shard, metadata=smeta or {"format": "pt"})
            del tensors
            print(f"wrote {shard}", flush=True)
    ratios = [r["a_pt"] / r["tot"] for r in report.values()]
    med = {k: statistics.median(r[k] for r in report.values()) for k in ("a_pt", "a_h", "w", "tot", "wspace")}
    gate = statistics.median(ratios) >= a.gate
    summary = {"layers": len(report), "median": med, "median_ratio": statistics.median(ratios),
               "min_ratio": min(ratios), "gate": gate, "gate_threshold": a.gate}
    doc = {"knob": {"VLLM_ACT_HADAMARD_RE": a.knob_re}, "block": G, "act_scale": ACT_SCALE,
           "weight_scale": W_SCALE, "source": a.src, "acts": a.acts, "rows": a.rows or "all",
           "summary": summary, "layers": report}
    if a.out:
        json.dump(doc, open(a.out, "w"), indent=1)
    if dst:
        linked = copied = 0
        for name in sorted(os.listdir(src)):
            p = src + name
            if not os.path.isfile(p) or ".bak" in name or name in by_shard:
                continue
            if name in COPY_NOT_LINK or not name.endswith(".safetensors"):
                shutil.copy2(p, dst + name)
                copied += 1
            else:
                try:
                    os.link(p, dst + name)
                    linked += 1
                except OSError:
                    shutil.copy2(p, dst + name)
                    copied += 1
        json.dump(doc, open(dst + "quant4_hadamard.json", "w"), indent=1)
        print(f"{len(by_shard)} shards rewritten, {linked} hardlinked, {copied} files copied -> {dst}")
        print(f"serve it only with VLLM_ACT_HADAMARD_RE='{a.knob_re}' (patches/quant4-act-hadamard.patch)")
    print(f"QUANT4 layers={len(report)} a_pt={med['a_pt']:.4f} a_h={med['a_h']:.4f} w={med['w']:.4f} "
          f"tot={med['tot']:.4f} ratio_med={summary['median_ratio']:.2f} ratio_min={summary['min_ratio']:.2f} "
          f"gate={'yes' if gate else 'no'}", flush=True)


if __name__ == "__main__":
    main()
