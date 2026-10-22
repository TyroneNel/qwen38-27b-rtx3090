"""Cold time-to-first-token of the running server at given prompt lengths (attn-3, docs/research/deepinfer).

Each request is a fresh random token-id prompt (ids drawn from [1000, 100000) with a per-(length, repeat) seed, the
same in every arm), sent to /v1/completions with max_tokens=1, so nothing is prefix-cached and the wall time is the
cold prefill plus one token. A short warm-up request runs first. One 'TTFT' line per request and a 'TTFT_MEAN' line
per length.

  python bench/ttft_cold.py --lens 51200,112000,148000 --reps 2 [--seed-base 7000] [--out FILE.json]
"""
import argparse
import json
import random
import sys
import time

import harness


def prompt(n, seed):
    r = random.Random(seed)
    return [r.randrange(1000, 100000) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lens", default="51200,112000,148000")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--out")
    a = ap.parse_args()
    harness.post("/v1/completions", {"prompt": prompt(2048, 1), "max_tokens": 1, "temperature": 0})
    rows = []
    for li, n in enumerate(int(x) for x in a.lens.split(",")):
        ts = []
        for rep in range(a.reps):
            ids = prompt(n, a.seed_base + 100 * li + rep)
            t0 = time.perf_counter()
            r = harness.post("/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 0}, timeout=3600)
            dt = time.perf_counter() - t0
            used = r.get("usage", {}).get("prompt_tokens")
            ts.append(dt)
            rows.append({"len": n, "rep": rep, "s": dt, "prompt_tokens": used})
            print(f"TTFT len={n} rep={rep} {dt:.2f} s (prompt_tokens={used})", flush=True)
        print(f"TTFT_MEAN len={n} {sum(ts) / len(ts):.2f} s over {len(ts)} (min {min(ts):.2f})", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
