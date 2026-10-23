"""quant-7 (docs/research/deepinfer/quant-heads.md): build an int4-embedding variant of a model dir.

Writes a NEW directory; the source is never written. Every weight shard but the one holding
embed_tokens is hardlinked from the source (files that prepare/ scripts rewrite in place,
model_extra_tensors.safetensors and mtp_draft_vocab_ids.pt, are copied instead, as
drafter/gptq_lm_head.py does). The embed shard is rewritten with embed_tokens packed to int-BITS
(group-128, symmetric, bf16 scales: prepare/quant_schema.pack, the same math prepare/quant_embed.py
uses at BITS=8), and the config group targeting embed_tokens declares BITS. Pre-quant backups
(*.bak*) are not carried over.

The bf16 table comes from --bf16-shard, else <src>/<embed shard>.bak_embed (quant_embed.py's backup).
The fast variant has no backup of its own; the base dir's backup holds the same table, which this
checks: the source's packed int8 table must dequantize to within 1% of the bf16 one.

Run with the image's venv, CPU only (no GPU needed, ~2 min, ~6 GB RAM):
  docker run --rm --entrypoint bash -v /mnt/g/models:/models -v /mnt/g/dev/HyperQwen:/repo:ro \\
    ghcr.io/syv-ai/hyperqwen:latest -c 'cd /repo && /app/venv/bin/python \\
    docs/research/deepinfer/experiments/quant7_emb4.py /models/Qwen3.8-27B-W4A16-AutoRound-fast \\
    /models/Qwen3.8-27B-W4A16-AutoRound-fast-emb4 \\
    --bf16-shard /models/Qwen3.8-27B-W4A16-AutoRound/model-00006-of-00007.safetensors.bak_embed'
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(REPO, "prepare"))
from quant_schema import GROUP, SUFFIXES, pack  # noqa: E402

COPY_NOT_LINK = ("model_extra_tensors.safetensors", "mtp_draft_vocab_ids.pt")


def dequant(packed, scale, bits, shape):
    """compressed-tensors pack-quantized (offset binary in int32 words) -> float32 [out, in]."""
    per = 32 // bits
    mask = (1 << bits) - 1
    out_f, in_f = shape
    rows = []
    for lo in range(0, out_f, 16384):
        p = packed[lo:lo + 16384].to(torch.int64)
        q = torch.stack([((p >> (bits * i)) & mask) - (1 << (bits - 1)) for i in range(per)], -1)
        q = q.reshape(len(p), in_f).to(torch.float32)
        s = scale[lo:lo + 16384].to(torch.float32).repeat_interleave(GROUP, dim=1)
        rows.append(q * s)
    return torch.cat(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bf16-shard", default=None)
    ap.add_argument("--bits", type=int, default=4)
    a = ap.parse_args()
    src, dst = a.src.rstrip("/") + "/", a.dst.rstrip("/") + "/"
    if os.path.exists(dst):
        sys.exit(f"{dst} exists; refusing to touch it (remove it yourself to rebuild).")

    idx = json.load(open(src + "model.safetensors.index.json"))
    wm = idx["weight_map"]
    base = next(k[: -len(".weight_packed")] for k in wm if k.endswith("embed_tokens.weight_packed"))
    shard = wm[base + ".weight_packed"]
    cfg = json.load(open(src + "config.json"))
    qc = cfg["quantization_config"]
    grp = next((g for g in qc["config_groups"].values()
                if any("embed_tokens" in t for t in g["targets"])), None)
    if grp is None:
        sys.exit("no config group targets embed_tokens; run prepare/quant_embed.py on the source first")
    src_bits = grp["weights"]["num_bits"]
    print(f"{base} in {shard}, source int{src_bits} -> int{a.bits}")

    bf16_path = a.bf16_shard or src + shard + ".bak_embed"
    with safe_open(bf16_path, "pt") as f:
        w = f.get_tensor(base + ".weight")
    with safe_open(src + shard, "pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    shape = tuple(tensors[base + ".weight_shape"].tolist())
    assert tuple(w.shape) == shape, (w.shape, shape)
    deq = dequant(tensors[base + ".weight_packed"], tensors[base + ".weight_scale"], src_bits, shape)
    wf = w.to(torch.float32)
    same = ((deq - wf).norm() / wf.norm()).item()
    del deq, wf
    print(f"source int{src_bits} table vs the bf16 one: rel error {same:.4f}")
    assert same < 0.01, f"{bf16_path} does not hold the source's embedding table"

    packed, err = pack(base, w, a.bits, torch.bfloat16)
    print(f"int{a.bits} round-trip relative error: {err:.4f}")
    del w
    for s in SUFFIXES:
        tensors.pop(f"{base}.{s}")
    tensors.update(packed)

    os.makedirs(dst)
    linked = copied = 0
    for name in sorted(os.listdir(src)):
        p = src + name
        if not os.path.isfile(p) or ".bak" in name or name in (shard, "config.json"):
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
    save_file(tensors, dst + shard, metadata=meta or {"format": "pt"})
    grp["weights"]["num_bits"] = a.bits
    json.dump(cfg, open(dst + "config.json", "w"), indent=2)
    old = os.path.getsize(src + shard) / 2**30
    new = os.path.getsize(dst + shard) / 2**30
    print(f"{linked} shards hardlinked, {copied} files copied; {shard}: {old:.3f} -> {new:.3f} GiB "
          f"({old - new:.3f} GiB of weights freed)")
    print("done:", dst)


if __name__ == "__main__":
    main()
