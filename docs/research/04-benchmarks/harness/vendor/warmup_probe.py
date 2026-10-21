#!/usr/bin/env python3
"""Warmup detection (evals-11): repeat a fixed short bench until two consecutive runs agree, instead of
"run it twice after a restart and keep the second" (run_benchmarks.sh).

  python3 bench/warmup_probe.py [--prompts 8] [--tokens 256] [--conc 1,4] [--tol 0.03] [--max-runs 6] [--out f.json]

One run is the same fixed bench every time: --prompts distinct short prompts, each decoding exactly --tokens
tokens (greedy, ignore_eos, thinking off), once at each concurrency in --conc. Per concurrency it records the
mean per-request decode rate (completion tokens after the first, over first-to-last token time; TTFT is
excluded, so prefix-cache hits on the repeated prompts do not move it) and the aggregate output rate over the
wall clock. The server is WARM after run k when every concurrency's decode rate is within --tol of run k-1's.
The prompts and greedy decoding are identical every run, so speculative acceptance is too, and only the
server's own state can move the numbers.

Prints one line per run and a summary: `SUMMARY warmup_probe WARM runs=k ...` (exit 0) or
`SUMMARY warmup_probe NOT-WARM runs=N ...` (exit 2: no verdict on the server's speed, it never settled). Exit 2
as well when a request fails or no server answers.
"""
import argparse
import json
import sys
import threading
import time

import harness

TOPICS = ["how a GPU schedules warps", "the history of the Øresund bridge", "why decode is memory-bound",
          "how a B-tree splits a node", "the rules of Danish handball", "what a page table walk costs",
          "how sourdough fermentation works", "the design of the TCP congestion window",
          "how Copenhagen's district heating works", "why hash tables resize", "how tides form",
          "what speculative decoding verifies"]


def prompt(i):
    return f"Write a long, detailed essay about {TOPICS[i % len(TOPICS)]}. (prompt {i})"


def one(i, tokens, out):
    payload = {"messages": [{"role": "user", "content": prompt(i)}], "max_tokens": tokens, "min_tokens": tokens,
               "ignore_eos": True, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    try:
        t0, first, last, ntok = time.perf_counter(), None, None, 0
        for d in harness.stream("/v1/chat/completions", payload, timeout=900):
            if d.get("usage"):
                ntok = d["usage"]["completion_tokens"]
            ch = d.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                last = time.perf_counter()
                first = first or last
        out[i] = {"ttft": first - t0 if first else None, "ntok": ntok,
                  "rate": (ntok - 1) / (last - first) if first and last and last > first else None}
    except Exception as e:  # noqa: BLE001 - reported as an invalid run
        out[i] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def phase(n_prompts, tokens, conc):
    """All prompts, at most `conc` in flight. Returns the per-request results and the wall time."""
    out, sem = {}, threading.Semaphore(conc)

    def worker(i):
        with sem:
            one(i, tokens, out)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_prompts)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [out[i] for i in range(n_prompts)], time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--conc", default="1,4")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--max-runs", type=int, default=6)
    ap.add_argument("--out", help="write every run as JSON")
    a = ap.parse_args()
    concs = [int(x) for x in a.conc.split(",")]
    runs, prev, warm_at = [], None, None
    for k in range(1, a.max_runs + 1):
        run = {"run": k, "phases": {}}
        for c in concs:
            res, wall = phase(a.prompts, a.tokens, c)
            errors = [r["error"] for r in res if "error" in r]
            rates = [r["rate"] for r in res if r.get("rate")]
            if errors or len(rates) != len(res):
                print(f"run {k} C{c}: INVALID, {len(errors)} request(s) failed, {len(res) - len(rates)} without a rate"
                      + (f": {errors[0]}" if errors else ""))
                print(f"SUMMARY warmup_probe INVALID runs={k} server={harness.base_url()}")
                sys.exit(2)
            run["phases"][c] = {"decode_tps": sum(rates) / len(rates), "agg_tps": sum(r["ntok"] for r in res) / wall,
                                "ttft_med": sorted(r["ttft"] for r in res)[len(res) // 2], "wall_s": wall}
        if prev:
            run["delta"] = {c: run["phases"][c]["decode_tps"] / prev["phases"][c]["decode_tps"] - 1 for c in concs}
        runs.append(run)
        print(f"run {k}: " + "  ".join(
            f"C{c} decode {p['decode_tps']:.1f} tok/s agg {p['agg_tps']:.1f} tok/s ttft_med {p['ttft_med'] * 1000:.0f}ms"
            + (f" ({run['delta'][c]:+.1%})" if prev else "") for c, p in run["phases"].items()), flush=True)
        if prev and all(abs(d) <= a.tol for d in run["delta"].values()):
            warm_at = k
            break
        prev = run
    last = runs[-1]["phases"]
    rates = " ".join(f"c{c}={last[c]['decode_tps']:.1f}" for c in concs)
    verdict = "WARM" if warm_at else "NOT-WARM"
    print(f"SUMMARY warmup_probe {verdict} runs={len(runs)} tol={a.tol:.0%} {rates}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"verdict": verdict, "runs": runs, "args": vars(a)}, f, indent=1)
    sys.exit(0 if warm_at else 2)


if __name__ == "__main__":
    main()
