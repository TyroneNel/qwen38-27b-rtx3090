"""Phase-3 arithmetic for transferability.md: memory gate, draft byte floors, break-even.

Every output is an estimate (E) built from the cited inputs. Run: python3 transferability-calc.py
"""
GIB = 2**30
BW = 936e9                       # 3090 DRAM bytes/s (dflash-specdec.md §6)
INT4 = 0.5 + 2 / 128             # bytes/param, int4 g128 with fp16 scales (target int4 head: 0.656 GB / 1.271B)

# EAGLE-4 Qwen3.8-27B draft, header counts (architecture.md §1, training-recipe lane)
TOTAL = 3_089_075_200
VOCAB, H = 248_320, 5120
EMB = HEAD = VOCAB * H                        # 1,271,398,400 each
FC = H * 3 * H                                # 5120 x 15360
FACT = VOCAB * 128 + 128 * H                  # rank-128 factors
LAYER = TOTAL - EMB - HEAD - FC - FACT        # attention + MLP + norms
assert EMB == 1_271_398_400 and LAYER > 0

def gb(p, bpp): return p * bpp / 1e9
def gib(p, bpp): return p * bpp / GIB

variants = {
    "V1 bf16 as released (own embed + own head + factors)": TOTAL * 2,
    "V2 bf16, embed dropped (share target embed)":            (TOTAL - EMB) * 2,
    "V3 = V2 minus efficient_lm_head factors (E-20 path)":   (TOTAL - EMB - FACT) * 2,
    "V4 W4A16 layer+fc, int4 own head, no embed, no factors": (LAYER + FC + HEAD) * INT4,
    "V5 = V4 + bf16 factors (efficient head kept)":          (LAYER + FC + HEAD) * INT4 + FACT * 2,
    "V6 bf16 layer+fc, shared target head (retrain/mismatch)": (LAYER + FC) * 2,
    "V7 W4A16 layer+fc, shared target head":                  (LAYER + FC) * INT4,
}
print("== EAGLE-4 draft resident weights ==")
for k, b in variants.items():
    print(f"{k:58s} {b/1e9:6.3f} GB  {b/GIB:6.3f} GiB")

# Modes: (drafter it replaces GiB, usable spare GiB, context tokens per GiB of KV_MEM)
# quantisation.md Headroom + Reviewer additions; E tokens/GiB unverified there.
modes = {
    "A batch fp8 (adds; no drafter)": (0.0,   0.15, 32_768),
    "B dflash2 fast bf16 KV":         (1.193, 0.50, 16_384),
    "C k=15 bf16 KV":                 (1.193, 0.20, 16_384),
    "D mtp long (int8 PTH)":          (0.305, 0.00, 31_800),
    "D' dflash2 long int8 PTH":       (1.193, 0.50, 31_800),
    "E dflash2 huge KVarN":           (1.193, 0.15, 80_000),
}
print("\n== Delta vs the drafter replaced, and context lost (GiB over spare x tokens/GiB) ==")
print(f"{'variant':6s}" + "".join(f"{m.split()[0]:>18s}" for m in modes))
for k, b in variants.items():
    row = []
    for m, (repl, spare, tpg) in modes.items():
        d = b / GIB - repl
        lost = max(0.0, d - spare) * tpg
        row.append(f"{d:+6.2f}/{lost/1000:6.1f}k")
    print(f"{k[:2]:6s}" + "".join(f"{c:>18s}" for c in row))

# Draft KV: MHA 40 heads x 128 x (K,V) x bf16
kv_tok = 40 * 128 * 2 * 2
print(f"\n== Draft KV == {kv_tok} B/token; 4096-window per request {kv_tok*4104/2**20:.0f} MiB;"
      f" if out-of-window blocks are not freed: 64k ctx {kv_tok*65536/GIB:.2f} GiB, 170k {kv_tok*170_000/GIB:.2f} GiB")

# Per draft step bytes and floors
K = 128
steps = {
    "E-20 bf16 layer + bf16 own head":          LAYER * 2 + HEAD * 2,
    "bf16 layer + efficient head (bf16 fact.)": LAYER * 2 + FACT * 2 + K * H * 2,
    "W4A16 layer + int4 own head":              LAYER * INT4 + HEAD * INT4,
    "W4A16 layer + efficient head":             LAYER * INT4 + FACT * 2 + K * H * INT4,
}
print("\n== Draft step byte floor (fc read once per round, not per step) ==")
fc_ms = FC * 2 / BW * 1e3
for k, b in steps.items():
    print(f"{k:44s} {b/1e9:5.3f} GB  {b/BW*1e3:5.2f} ms/step;  k=7 chain {7*b/BW*1e3 + fc_ms:5.2f} ms")

# DFlash2 reference (dflash-specdec.md §6-7): floor 1.80 ms, measured 3.0-3.6 ms
DF_FLOOR, DF_MEAS = 1.80, 3.3
ratio = DF_MEAS / DF_FLOOR
B_STEP = 1e3 / (114.4 / 3.36)      # measured B C1: 114.4 tok/s at 3.36 tok/step (baseline.md)
print(f"\nB C1 step = {B_STEP:.1f} ms (from 114.4 tok/s / 3.36 tok/step); DFlash2 realised/floor = {ratio:.2f}")
print("== Break-even tok/step for an EAGLE-4 k=7 chain replacing DFlash2 at B C1 ==")
for k, b in steps.items():
    for lab, r in (("floor", 1.0), ("x%.2f" % ratio, ratio)):
        t = (7 * b / BW * 1e3 + fc_ms) * r
        need = 3.36 * (B_STEP - DF_MEAS + t) / B_STEP
        print(f"{k:44s} {lab:6s} draft {t:5.2f} ms -> need {need:4.2f} tok/step ({(need/3.36-1)*100:+5.1f}%)")

# Low-rank head inside DFlash2 compute_candidates (7 rows, full 248k int4 head today)
full = HEAD * INT4
lr = FACT * 2 + 7 * K * H * INT4
save_ms = (full - lr) / BW * 1e3
print(f"\n== E-10 on DFlash2 == full head {full/1e9:.3f} GB -> factors+rows {lr/1e9:.3f} GB;"
      f" saves {save_ms:.2f} ms floor = {save_ms/B_STEP*100:.1f}% of B step -> +{(B_STEP/(B_STEP-save_ms)-1)*100:.1f}% tok/s;"
      f" at E 170k: {save_ms/150.8*100:.2f}% of 150.8 ms")

# Low-rank head for MTP (40,960-row int4 slice today, k=3 sequential)
slc = 40_960 * H * INT4
lr1 = FACT * 2 + K * H * INT4
D_STEP = 1e3 / (99.8 / 2.64)
d = 3 * (slc - lr1) / BW * 1e3
print(f"== T-01 on MTP == slice {slc/1e9:.3f} GB vs factors+rows {lr1/1e9:.3f} GB per step;"
      f" 3 steps save {d:.2f} ms of D step {D_STEP:.1f} ms ({d/D_STEP*100:.1f}%)")
