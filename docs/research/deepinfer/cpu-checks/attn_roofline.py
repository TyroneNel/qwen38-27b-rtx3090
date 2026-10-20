"""CPU check (d): roofline of target verify/decode attention per engine step, Qwen3.8-27B on one RTX 3090.

Model geometry (HyperQwen docs): 16 full-attention layers, 4 KV heads, GQA 6 (24 q heads), head_dim 256.
Peaks are GA102 whitepaper numbers for the RTX 3090 at boost clock (inferred; a 250 W cap lowers them):
  HBM 936 GB/s; fp16 MMA fp32-acc 71 TFLOPS; fp16-acc 142; int8 284 TOPS (all dense).
Rows per KV head = 6*T. Padded rows follow pthfa launch.cu:365-385 (16*wm rows per tile, <=10 tokens per tile at G=6).
Times are lower bounds at peak; the measured 3080 point (docs/long-context.md:538) calibrates them.
Run: python3 -I roofline.py
"""
import math

LAYERS, KVH, G, D = 16, 4, 6, 256
BW, F32ACC, F16ACC, I8 = 936e9, 71e12, 142e12, 284e12


def padded_rows(T):
    rows = G * T
    wm = 1
    while wm < 4 and rows > 16 * wm:
        wm *= 2
    cap = 16 * wm // G
    tiles = math.ceil(T / cap)
    return tiles * 16 * wm, tiles


def step(T, L, bytes_per_elt, kvh=KVH, bw=BW, f32=F32ACC):
    byts = LAYERS * kvh * D * 2 * bytes_per_elt * L
    prow, tiles = padded_rows(T)
    fl_real = LAYERS * kvh * 4 * D * G * T * L
    fl_pad = LAYERS * kvh * 4 * D * prow * L
    return dict(mem=byts / bw * 1e3, real=fl_real / f32 * 1e3, pad=fl_pad / f32 * 1e3,
                pv16=fl_pad / 2 / f32 * 1e3 + fl_pad / 2 / (2 * f32) * 1e3,
                q8pv16=fl_pad / 2 / (4 * f32) * 1e3 + fl_pad / 2 / (2 * f32) * 1e3,
                tiles=tiles, ai=fl_real / byts)


MODES = [  # (label, T verify tokens, context, KV bytes per element, note)
    ("B chat dflash2 64k bf16", 8, 65536, 2.0, "CTX=fast bf16 FLASH_ATTN"),
    ("C repro dflash2 T=16 64k bf16", 16, 65536, 2.0, "DFLASH_TOKENS=15"),
    ("D 150k mtp k=3 fp8/int8", 4, 150000, 1.0, "CTX=long DRAFT_TOKENS=3"),
    ("D' 150k dflash2 int8 (pthfa)", 8, 150000, 1.0, "SPEC=dflash2 CTX=long"),
    ("E 240k dflash2 KVarN k4v2", 8, 240000, 0.375, "6 bits/elt, scales ignored"),
    ("E 240k at int8 (if it fit)", 8, 240000, 1.0, "reference"),
]

if __name__ == "__main__":
    assert padded_rows(1) == (16, 1) and padded_rows(8) == (64, 1) and padded_rows(16) == (128, 2)
    # calibration: 3080 TP=2 rank (2 KV heads), 128k, T=8, int8; 3080 = 760 GB/s, ~59.5 TFLOPS fp32-acc (inferred)
    c = step(8, 131072, 1.0, kvh=2, bw=760e9, f32=59.5e12)
    print(f"calib 3080 rank 128k T=8 int8: mem {c['mem']:.2f} ms, MMA real {c['real']:.2f}, padded {c['pad']:.2f}"
          f"  | measured 5.631 ms (docs/long-context.md:538)")
    print(f"\n{'mode':32s} {'T':>3s} {'tiles':>5s} {'AI f/B':>6s} {'mem ms':>7s} {'MMA f32acc':>10s} "
          f"{'padded':>7s} {'PV f16acc':>9s} {'QK i8+PV f16':>12s}")
    for lab, T, L, b, note in MODES:
        r = step(T, L, b)
        print(f"{lab:32s} {T:3d} {r['tiles']:5d} {r['ai']:6.0f} {r['mem']:7.2f} {r['real']:10.2f} "
              f"{r['pad']:7.2f} {r['pv16']:9.2f} {r['q8pv16']:12.2f}")
    print("\nGA102 ridge points (FLOP per byte): f32acc", F32ACC / BW, "f16acc", F16ACC / BW, "int8", I8 / BW)
