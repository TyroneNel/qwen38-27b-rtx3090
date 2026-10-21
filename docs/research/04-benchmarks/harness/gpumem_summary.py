#!/usr/bin/env python3
"""Summarise the WDDM "Non Local Usage" samples (the box's G:\\dev\\hq-bench\\gpumem_sample.ps1 -Match 1144f output,
ENVIRONMENT.md:28) into OUT/spill.json: first / max / last value and growth. This counter is the VRAM-spill source on
the WSL2 box (nvidia-smi cannot see memory demoted to system RAM). Growth over --tol MiB (default 64, the prior
"no growth" bound, baseline.md:58) is SPILL and the arm must be re-run with a smaller GPU_UTIL / KV_MEM.

  gpumem_summary.py SAMPLES.csv OUT.json [--tol 64]

ponytail: the sampler's exact format is not in the repo; each line's LAST numeric field is taken as MiB. Check one
real file against the sampler's header once (README "VRAM and spill") and fix here if the unit differs.
Exit 0 flat, 1 SPILL, 2 no samples.
"""
import json
import re
import sys

tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 64.0
v = []
for line in open(sys.argv[1], errors="replace"):
    m = re.findall(r"-?\d+(?:\.\d+)?", line.split(",")[-1] if "," in line else line.split()[-1] if line.split() else "")
    if m:
        v.append(float(m[-1]))
if not v:
    json.dump({"samples": 0, "verdict": "NO-SAMPLES"}, open(sys.argv[2], "w"))
    print("SUMMARY spill NO-SAMPLES")
    sys.exit(2)
r = {"source": "WDDM Non Local Usage (MiB, assumed)", "samples": len(v), "first": v[0], "max": max(v), "last": v[-1],
     "growth": max(v) - v[0], "tol": tol}
r["verdict"] = "SPILL" if r["growth"] > tol else "flat"
json.dump(r, open(sys.argv[2], "w"))
print(f"SUMMARY spill {r['verdict']} first={v[0]:.0f} max={max(v):.0f} last={v[-1]:.0f} growth={r['growth']:.0f} MiB (n={len(v)})")
sys.exit(1 if r["verdict"] == "SPILL" else 0)
