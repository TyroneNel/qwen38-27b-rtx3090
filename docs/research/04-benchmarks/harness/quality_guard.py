#!/usr/bin/env python3
"""Quality guard: judge a bench/quality_battery.py result against a reference. The battery runs the perplexity
(wikitext-2 en, fineweb-2 da, vLLM source code) and GSM8K n=200 greedy; this adds the pass/fail.

  quality_guard.py RESULT.json [--ref quality_ref_prior.json] [--ppl-tol-pct 1.0] [--gsm-tol-pts 3.0]

RESULT.json is `$QUALITY_DATA/result_<tag>.json` as the battery writes it. Default reference: the prior GPU-box
stock run (DeepInfer evals-3, mode B, `raw/prior-3090/evals-3.md`): ppl all 8.401, GSM8K 0.955.
Gates: perplexity (all and each corpus) may not rise more than --ppl-tol-pct (a lossless change reads 0; evals-3
measured +4.0% for all-linear int8, so 1% separates them); GSM8K may not fall more than --gsm-tol-pts points (n=200
greedy is a garbage detector only, one question = 0.5 pt, the prior pair differed by 3 questions: evals-3).
Exit 0 PASS, 1 FAIL, 2 invalid (a field missing).
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("result")
ap.add_argument("--ref", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_ref_prior.json"))
ap.add_argument("--ppl-tol-pct", type=float, default=1.0)
ap.add_argument("--gsm-tol-pts", type=float, default=3.0)
a = ap.parse_args()
try:
    r, ref = json.load(open(a.result)), json.load(open(a.ref))
    bad = []
    for k, v in ref["ppl"].items():
        d = 100 * (r["ppl"][k][0] / v - 1)
        ok = d <= a.ppl_tol_pct
        print(f"ppl {k:5s} {r['ppl'][k][0]:.4f} vs ref {v:.4f}: {d:+.2f}% {'ok' if ok else 'FAIL'}")
        if not ok: bad.append(f"ppl {k}")
    d = 100 * (r["gsm8k"]["acc"] - ref["gsm8k"]["acc"])
    ok = d >= -a.gsm_tol_pts
    print(f"gsm8k {r['gsm8k']['acc']:.3f} vs ref {ref['gsm8k']['acc']:.3f}: {d:+.1f} pts {'ok' if ok else 'FAIL'}")
    if not ok: bad.append("gsm8k")
except (KeyError, IndexError, TypeError, OSError, ValueError) as e:
    print(f"INVALID: {type(e).__name__}: {e}"); sys.exit(2)
print("SUMMARY quality_guard", "FAIL " + ", ".join(bad) if bad else "PASS")
sys.exit(1 if bad else 0)
