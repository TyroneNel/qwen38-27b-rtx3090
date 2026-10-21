#!/usr/bin/env python3
"""Paired comparison of two quality_battery.py --jsonl runs: A is the reference, B the candidate (evals-3).

  python3 bench/compare_quality.py A.jsonl B.jsonl [--alpha 0.05] [--out f.json]

Per GSM8K arm present in both files (greedy, seed1..seedN), paired on the item index: accuracy with a Wilson
95% interval for each side, the McNemar discordant counts b (A right, B wrong) and c (A wrong, B right), and
the exact two-sided binomial p on them. The sampled arms are also pooled per item: an item counts for A when A
got it right in more of the shared seed arms than B, and the same sign test runs over those items, which stay
independent where a pool of seed-items would not. Perplexity rows give B's change against A per corpus.

Warns on every row that hit finish_reason=length (truncated at max_tokens) or whose answer came from the
last-number fallback rather than "Final answer:", since either can move accuracy without the model changing.

The verdict reads two tests: the greedy arm, and the pooled sampled arm if there is one. Exit 1 (FAIL) only
when one of them is a significant paired loss of B against A (b > c and p < alpha); exit 0 otherwise, including
a significant gain. Exit 2 when the files share no GSM8K item or perplexity corpus. Stdlib only.
"""
import argparse
import json
import math
import sys

Z = 1.959963984540054


def load(path):
    gsm, ppl = {}, {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "gsm8k":
                gsm.setdefault(r["arm"], {})[r["idx"]] = r
            elif r.get("kind") == "ppl":
                ppl[r["corpus"]] = r
    return gsm, ppl


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def binom_p(b, c):
    """Exact two-sided binomial test of b against c at p=0.5 (McNemar's exact test)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(name, a_ok, b_ok, alpha):
    """a_ok, b_ok: {idx: correct} on the same items."""
    n = len(a_ok)
    ka, kb = sum(a_ok.values()), sum(b_ok.values())
    b = sum(1 for i in a_ok if a_ok[i] and not b_ok[i])
    c = sum(1 for i in a_ok if b_ok[i] and not a_ok[i])
    p = binom_p(b, c)
    return {"arm": name, "n": n, "acc_a": ka / n, "acc_b": kb / n, "ci_a": wilson(ka, n), "ci_b": wilson(kb, n),
            "b": b, "c": c, "p": p, "loss": b > c and p < alpha, "gain": c > b and p < alpha}


def warnings(tag, gsm, items):
    out = []
    for arm in sorted(gsm):
        rows = [gsm[arm][i] for i in items.get(arm, ())]
        trunc = [r["idx"] for r in rows if r.get("finish_reason") == "length"]
        fall = [r["idx"] for r in rows if r.get("extracted_via") != "final_answer"]
        if trunc:
            out.append(f"WARN {tag} {arm}: {len(trunc)} item(s) finish_reason=length {trunc[:12]}")
        if fall:
            out.append(f"WARN {tag} {arm}: {len(fall)} item(s) without 'Final answer:' (fallback extraction) {fall[:12]}")
    return out


def fmt(r):
    return (f"{r['arm']:<16} n={r['n']:<4} A={r['acc_a']:.3f} [{r['ci_a'][0]:.3f},{r['ci_a'][1]:.3f}]  "
            f"B={r['acc_b']:.3f} [{r['ci_b'][0]:.3f},{r['ci_b'][1]:.3f}]  b={r['b']} c={r['c']} p={r['p']:.4g}"
            + ("  LOSS" if r["loss"] else "  gain" if r["gain"] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", help="write the comparison as JSON")
    args = ap.parse_args()
    (ga, pa), (gb, pb) = load(args.a), load(args.b)

    arms = sorted(set(ga) & set(gb), key=lambda s: (s != "greedy", len(s), s))
    items = {arm: sorted(set(ga[arm]) & set(gb[arm])) for arm in arms}
    results = []
    for arm in arms:
        if items[arm]:
            results.append(paired(arm, {i: bool(ga[arm][i]["correct"]) for i in items[arm]},
                                  {i: bool(gb[arm][i]["correct"]) for i in items[arm]}, args.alpha))
    seeds = [a for a in arms if a != "greedy" and items[a]]
    pooled = None
    if seeds:
        common = sorted(set.intersection(*(set(items[a]) for a in seeds)))
        if common:
            sa = {i: sum(bool(ga[a][i]["correct"]) for a in seeds) for i in common}
            sb = {i: sum(bool(gb[a][i]["correct"]) for a in seeds) for i in common}
            b = sum(1 for i in common if sa[i] > sb[i])
            c = sum(1 for i in common if sb[i] > sa[i])
            p = binom_p(b, c)
            ka, kb, m = sum(sa.values()), sum(sb.values()), len(common) * len(seeds)
            pooled = {"arm": f"sampled({len(seeds)})", "n": len(common), "acc_a": ka / m, "acc_b": kb / m,
                      "ci_a": wilson(ka, m), "ci_b": wilson(kb, m), "b": b, "c": c, "p": p,
                      "loss": b > c and p < args.alpha, "gain": c > b and p < args.alpha}

    ppl = []
    for corpus in sorted(set(pa) & set(pb), key=lambda s: (s == "all", s)):
        x, y = pa[corpus]["ppl"], pb[corpus]["ppl"]
        ppl.append({"corpus": corpus, "ppl_a": x, "ppl_b": y, "delta": y - x, "delta_pct": 100 * (y - x) / x,
                    "tokens_a": pa[corpus].get("tokens"), "tokens_b": pb[corpus].get("tokens")})

    print(f"A={args.a}\nB={args.b}")
    for r in results:
        print("GSM8K " + fmt(r))
    if pooled:
        print("GSM8K " + fmt(pooled) + "   (per item: seeds A got right vs B; sign test)")
    for r in ppl:
        flag = "" if r["tokens_a"] == r["tokens_b"] else f"  (token counts differ: {r['tokens_a']} vs {r['tokens_b']})"
        print(f"PPL   {r['corpus']:<5} A={r['ppl_a']:.4f} B={r['ppl_b']:.4f} delta={r['delta']:+.4f} ({r['delta_pct']:+.2f}%){flag}")
    warns = warnings("A", ga, items) + warnings("B", gb, items)
    for w in warns:
        print(w)

    if not results and not ppl:
        print("SUMMARY compare_quality INVALID: no shared GSM8K item or perplexity corpus")
        sys.exit(2)
    greedy = next((r for r in results if r["arm"] == "greedy"), None)
    gated = [r for r in (greedy, pooled) if r]
    loss = [r["arm"] for r in gated if r["loss"]]
    verdict = "FAIL" if loss else "PASS"
    parts = [f"{r['arm']} b={r['b']} c={r['c']} p={r['p']:.4g}" for r in gated]
    allp = next((r for r in ppl if r["corpus"] == "all"), None)
    if allp:
        parts.append(f"ppl_all {allp['delta_pct']:+.2f}%")
    parts.append(f"warnings={len(warns)}")
    print(f"SUMMARY compare_quality {verdict}" + (f" loss={','.join(loss)}" if loss else "") + " " + " ".join(parts))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"a": args.a, "b": args.b, "alpha": args.alpha, "verdict": verdict, "arms": results,
                       "sampled_pooled": pooled, "ppl": ppl, "warnings": warns}, f, indent=1)
    sys.exit(1 if loss else 0)


if __name__ == "__main__":
    main()
