#!/usr/bin/env python3
"""Fixed-prompt-set sweep: TTFT, decode tok/s, throughput, spec-decode acceptance and peak VRAM, each as
mean and spread over repeated rounds, per (context, concurrency) cell. Stdlib only; reaches the server
through bench/harness.py (run with PYTHONPATH=/app/bench, as run_suite.sh does).

  sweep.py LABEL --set short_chat|code|long_context [--ctx 8000,32000] [--conc 1,2,4,8] [--reps 3]
           [--warmup 1] [--max-tokens 256] [--exact] [--think] [--salt S] [--corpus PATH] [--out DIR]

One round = max(conc, len(set)) requests (the set, cycled) with at most `conc` in flight. Every request
starts with a distinct salt line, so none is served from the prefix cache (TTFT is cold) and the same
--salt gives the same prompts on every boot, which an A/B needs (acceptance depends on the prompt). A run
that reuses a salt on a server that has seen it prints `!` and records cached tokens.
Per round: mean TTFT, mean per-stream decode rate ((tokens-1)/(last-first token time)), aggregate tok/s
(all output tokens / round wall clock, TTFT included), and from /metrics deltas: tokens/step (1 + accepted/
drafts), acceptance rate (accepted/drafted tokens) and accepted-per-position rates. Across the --reps
rounds (after --warmup discarded ones): mean, stdev, min, max, CV%. Peak VRAM is nvidia-smi memory.used (secondary; the WDDM spill reading is gpumem_summary.py)
over the cell (vram.py). --exact sets min_tokens = max_tokens and ignore_eos so every request decodes the
same length. long_context builds the document from the labd corpus (3.6 chars per token, as labd_bench.py).
Writes OUT/sweep_<LABEL>_<set>.json and prints a markdown table. Exit 0 measured, 2 no valid round.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import harness
from vram import Peak

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = ["/cache/bench/labd_corpus_long.txt", "~/bench/labd_corpus_long.txt",
           "/cache/bench/labd_corpus.txt", "~/bench/labd_corpus.txt"]
POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
SC = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
      "vllm:spec_decode_num_accepted_tokens_total")


def spec_counters():
    """{drafts, draft_tokens, accepted, pos: {i: accepted}} summed over engines; zeros if /metrics lacks them."""
    out = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "pos": {}}
    for name, labels, v in harness.samples():
        if name == SC[0]: out["drafts"] += v
        elif name == SC[1]: out["draft_tokens"] += v
        elif name == SC[2]: out["accepted"] += v
        elif name == POS:
            p = int(labels.split('position="')[1].split('"')[0])
            out["pos"][p] = out["pos"].get(p, 0.0) + v
    return out


def corpus_text(path):
    for p in ([path] if path else []) + [os.environ.get("HARNESS_CORPUS", "")] + CORPORA:
        p = os.path.expanduser(p) if p else p
        if p and os.path.isfile(p):
            return open(p).read(), p
    sys.exit("no corpus: pass --corpus (the frozen labd corpus, /cache/bench/labd_corpus_long.txt on the GPU box)")


def build(args):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "prompts", args.set + ".jsonl"))]
    if args.set != "long_context":
        return [(0, [(r["id"], r["prompt"]) for r in rows])], None
    text, path = corpus_text(args.corpus)
    cells = []
    for ctx in args.ctx:
        doc = text[: int(ctx * 3.6)]
        if len(doc) < int(ctx * 3.6):
            print(f"WARNING: corpus {path} has {len(text)} chars, ctx {ctx} wants {int(ctx * 3.6)}", file=sys.stderr)
        cells.append((ctx, [(r["id"], "Dokument:\n\n" + doc + "\n\n" + r["question"]) for r in rows]))
    return cells, path


def one(prompt, args, res, i):
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": args.think}}
    if args.exact:
        payload.update(min_tokens=args.max_tokens, ignore_eos=True)
    t0, first, last, ntok, ptok, cached = time.perf_counter(), None, None, 0, 0, 0
    try:
        for d in harness.stream("/v1/chat/completions", payload, timeout=3600):
            u = d.get("usage")
            if u:
                ntok, ptok = u["completion_tokens"], u["prompt_tokens"]
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            ch = d.get("choices") or []
            delta = ch[0].get("delta", {}) if ch else {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                last = time.perf_counter()
                first = first or last
            elif first and ch and ch[0].get("finish_reason"):  # review: an EOS-only last step streams an empty finish chunk
                last = time.perf_counter()  # (vLLM serving.py:745-755); ntok counts that EOS, so its step belongs in the window
        if first is None:
            raise RuntimeError("no content")
        res[i] = dict(ttft=first - t0, ntok=ntok, ptok=ptok, cached=cached,
                      rate=(ntok - 1) / (last - first) if last > first and ntok > 1 else None)
    except Exception as e:  # noqa: BLE001 - an invalid round, reported
        res[i] = dict(error=f"{type(e).__name__}: {str(e)[:160]}")


def rnd(prompts, conc, tag, args):
    n = max(conc, len(prompts))
    res, sem, ths = {}, threading.Semaphore(conc), []

    def run(i):
        with sem:
            one(f"[run {tag}-{i}]\n" + prompts[i % len(prompts)][1], args, res, i)
    s0, t0 = spec_counters(), time.perf_counter()
    for i in range(n):
        ths.append(threading.Thread(target=run, args=(i,)))
        ths[-1].start()
    for t in ths:
        t.join()
    wall, s1 = time.perf_counter() - t0, spec_counters()
    bad = [r["error"] for r in res.values() if "error" in r]
    if bad:
        return {"error": bad[0], "n_failed": len(bad)}
    good = list(res.values())
    d = {k: s1[k] - s0[k] for k in ("drafts", "draft_tokens", "accepted")}
    out = dict(n=n, wall=wall, ttft=statistics.mean(r["ttft"] for r in good) * 1000,
               rate=statistics.mean(r["rate"] for r in good if r["rate"]) if any(r["rate"] for r in good) else None,
               agg=sum(r["ntok"] for r in good) / wall, ntok=sum(r["ntok"] for r in good),
               ptok=statistics.mean(r["ptok"] for r in good), cached=sum(r["cached"] for r in good))
    if d["drafts"] > 0:
        out["tok_step"] = 1 + d["accepted"] / d["drafts"]
        out["accept_rate"] = d["accepted"] / d["draft_tokens"] if d["draft_tokens"] else None
        out["pos_rate"] = [round((s1["pos"].get(p, 0) - s0["pos"].get(p, 0)) / d["drafts"], 4) for p in sorted(s1["pos"])]
    return out


def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return dict(mean=m, sd=sd, min=min(vals), max=max(vals), cv_pct=100 * sd / m if m else 0.0, n=len(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("label")
    ap.add_argument("--set", required=True, choices=["short_chat", "code", "long_context"])
    ap.add_argument("--ctx", type=lambda s: [int(x) for x in s.split(",")], default=[8000, 32000])
    ap.add_argument("--conc", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--salt", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    if args.reps < 3:
        sys.exit("--reps must be >= 3 (mean and spread need it)")
    salt = args.salt or args.label
    cells, corpus = build(args)
    os.makedirs(args.out, exist_ok=True)
    print(f"| set | ctx | conc | prompt tok | TTFT ms | decode tok/s/stream | agg tok/s | tok/step | accept rate | peak VRAM MiB | rounds |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    result, valid_any = [], False
    for ctx, prompts in cells:
        for conc in args.conc:
            rounds = []
            with Peak() as pk:
                for r in range(args.warmup + args.reps):
                    kind = "warmup" if r < args.warmup else "rep"
                    x = rnd(prompts, conc, f"{salt}-{args.set}-{ctx}-c{conc}-{kind}{r}", args)
                    x["kind"] = kind
                    rounds.append(x)
            reps = [x for x in rounds if x["kind"] == "rep" and "error" not in x]
            cell = dict(set=args.set, ctx=ctx, conc=conc, rounds=rounds, vram=pk.as_dict(), valid_reps=len(reps))
            if len(reps) == args.reps:
                valid_any = True
                cell["summary"] = {k: stat([x.get(k) for x in reps])
                                   for k in ("ttft", "rate", "agg", "tok_step", "accept_rate", "ptok")}
                if any(x["cached"] for x in reps):
                    print(f"! ctx {ctx} conc {conc}: prompt tokens served from prefix cache; change --salt", file=sys.stderr)
            result.append(cell)
            s = cell.get("summary")
            f = lambda k, p=1: "n/a" if not s or not s[k] else f"{s[k]['mean']:.{p}f} ± {s[k]['sd']:.{p}f} (cv {s[k]['cv_pct']:.1f}%)"
            vr = "n/a" if pk.peak_mib is None else f"{pk.peak_mib:.0f}"
            print(f"| {args.set} | {ctx or '-'} | {conc} | {s['ptok']['mean']:.0f} | {f('ttft', 0)} | {f('rate')} | {f('agg')} | "
                  f"{f('tok_step', 2)} | {f('accept_rate', 3)} | {vr} | {len(reps)}/{args.reps} ok |"
                  if s else f"| {args.set} | {ctx or '-'} | {conc} | INVALID: {[x.get('error') for x in rounds if 'error' in x][:1]} |", flush=True)
    path = os.path.join(args.out, f"sweep_{args.label}_{args.set}.json")
    json.dump(dict(label=args.label, set=args.set, salt=salt, corpus=corpus, reps=args.reps, warmup=args.warmup,
                   max_tokens=args.max_tokens, exact=args.exact, think=args.think, cells=result,
                   env={k: os.environ[k] for k in ("SPEC", "CTX", "DFLASH_TOKENS", "PREFIX_CACHE", "PTH_FA") if k in os.environ}),
              open(path, "w"), indent=1)
    print(f"\nwrote {path}")
    sys.exit(0 if valid_any else 2)


main()
