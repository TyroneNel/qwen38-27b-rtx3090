"""Reviewer recomputation for docs/research/02-eagle4/training-recipe.md.

1. Param count from the HF safetensors header (HTTP range request, header only, no weights).
2. Every cost estimate in the doc, at 25 TFLOP/s (doc) and at the throughput implied by our
   trainer's documented 4k tok/s (drafter/README.md:57).
Run: python3 -I phase1-eagle4-training-calc.py > phase1-eagle4-training-calc.out
"""
import json, math, struct, urllib.request

URL = "https://huggingface.co/EAGLE-corp/EAGLE4-Qwen3.8-27B/resolve/main/model.safetensors"


def rng(a, b):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req).read()


n = struct.unpack("<Q", rng(0, 7))[0]
hdr = json.loads(rng(8, 8 + n - 1))
print("metadata", hdr.pop("__metadata__", None), "header bytes", n)
groups, tot, data = {}, 0, 0
for k, v in sorted(hdr.items()):
    p = math.prod(v["shape"]); tot += p; data += v["data_offsets"][1] - v["data_offsets"][0]
    g = "layers.0" if k.startswith("layers.0") else k.split(".")[0].replace("_down_proj", "")
    groups[g] = groups.get(g, 0) + p
    print(f"  {k:45s} {v['dtype']} {v['shape']}")
for g, p in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"{g:22s} {p/1e6:9.2f} M  {p/tot:6.1%}")
print(f"total params {tot:,}  data bytes {data:,}  file bytes {8 + n + data:,}")
assert all(v["dtype"] == "BF16" for v in hdr.values())

# ---- FLOP model (doc's formula) ----------------------------------------------------------
P_layer = groups["layers.0"] + groups["norm"] + groups["fc_norm"]
P_body = P_layer + groups["fc"]
P_lm = groups["lm_head"]
P_fac = groups["efficient_lm_head"]
print(f"\nP_body {P_body/1e9:.4f}B  P_lm {P_lm/1e9:.4f}B  P_factors {P_fac/1e6:.2f}M")
per_depth = (6 * P_body + 4 * P_lm) / 1e9
teacher = 2 * P_lm / 1e9
for K in (1, 3, 7):
    print(f"K={K}: {K*per_depth + teacher:.2f} GFLOP/token (teacher once); "
          f"{K*(per_depth + teacher):.2f} if teacher recomputed per depth (train_mtp.py style)")
# EAGLE-4-faithful variant: draft lm_head trainable (vllm/models.py:146 'The draft trains its own lm_head')
print(f"K=7, lm_head trainable: {7*6*(P_body+P_lm)/1e9 + teacher:.1f} GFLOP/token")

# ---- implied throughput of our trainer ---------------------------------------------------
P_mtp = 0.37e9 + 2 * 5120 * 5120          # Table 1 layer params + fc [H, 2H]
P_slice = 40960 * 5120                      # draft-vocab head, frozen (no --train-head)
resp = 5.4 / 7.2                            # response share of captured rows; head+teacher only on these
for label, r in (("all rows", 1.0), ("response rows only", resp)):
    g = 2 * (6 * P_mtp + r * (4 * P_slice + 2 * P_lm)) / 1e9   # depth 2, teacher per depth (:292, :346)
    print(f"our trainer depth 2 ({label}): {g:.2f} GFLOP/token -> {g*4000/1e3:.1f} TFLOP/s at 4k tok/s")
print(f"4k tok/s check: 7.2M tokens / 4000 = {7.2e6/4000/60:.0f} min (README says ~30 min/epoch)")

# ---- estimates ---------------------------------------------------------------------------
def hours(tokens, gflop, tf):
    return tokens * gflop * 1e9 / (tf * 1e12) / 3600

g7 = 7 * per_depth + teacher
pa = (2 * 0.42e9 + 2 * P_lm + 6 * P_fac) / 1e9
pd = (2 * P_body + 2 * P_lm + 6 * P_fac) / 1e9
gen_rate = 5.4e6 / (2.2 * 3600)
print(f"\ngen rate {gen_rate:.0f} tok/s; cache {4*5120*2} B/token")
for tf in (25, 40, 44):
    print(f"--- {tf} TFLOP/s")
    print(f"P-a {pa:.2f} GFLOP/token -> {hours(7.2e6, pa, tf)*60:.1f} min/epoch")
    print(f"P-c 1M tokens K=7 -> {hours(1e6, g7, tf)*60:.1f} min train + {1.7/7.2*60:.1f} min capture")
    print(f"P-d {pd:.2f} GFLOP/token -> {hours(7.2e6, pd, tf)*60:.1f} min/epoch (7.2M rows need "
          f"{7.2e6*40960/1e9:.0f} GB 4-tap cache)")
    for D in (7.2e6, 50e6, 200e6, 1e9):
        tr = hours(D, g7, tf); gen = D / gen_rate / 3600; cap = 1.7 * D / 7.2e6
        print(f"  D={D/1e6:6.1f}M train {tr:7.1f} h  gen {gen:6.1f} h  capture {cap:6.1f} h  "
              f"total {tr+gen+cap:7.1f} h  cache {D*40960/1e12:5.2f} TB  H100@300 {hours(D, g7, 300):5.1f} h")
    print(f"  candidate stage 50M: {hours(50e6, pd, tf):.1f} h")

# ---- memory -------------------------------------------------------------------------------
print(f"\nbody fp32+Adam {P_body*16/1e9:.1f} GB; body+lm_head {(P_body+P_lm)*16/1e9:.1f} GB")
print(f"frozen bf16 embed+draft lm_head {2*P_lm*2/1e9:.2f} GB; + target lm_head for teacher "
      f"{3*P_lm*2/1e9:.2f} GB")
print(f"T-01 bytes: factors {P_fac*2/1e6:.1f} MB bf16; int4 slice {P_slice/2/1e6:.1f} MB (+scales)")
