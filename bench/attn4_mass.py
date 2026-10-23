#!/usr/bin/env python3
"""attn-4 offline study, CPU step (docs/research/deepinfer/attention-kv.md "attn-4"): how much attention mass do
top-k KV pages capture on real Q/K, per full-attention layer? Gate: >= 0.99.

Input: a bench/attn4_dump.py directory (L<NN>.k.npy, L<NN>.q.npy, L<NN>.qpos.npy, meta.json). Per layer and KV head,
fp32 on the CPU: scores s = q . k * scale over every key at or before the query's position (causal), p = softmax(s).
Keys are cut into pages of P tokens (--pages 16,64); a query at position pos sees v = pos // P + 1 pages. A
selection keeps k = max(2, ceil(pct * v)) pages (--pcts 2,5,10), always including page 0 (the sink) and the
query's own page (the local window), both counted in k. Captured mass = the sum of p over the kept pages.
Selectors:
  oracle    the k pages with the largest exact mass (upper bound of any page selector at this budget)
  bounds    per page and KV head, min and max of K per channel (16 B/token/head at P = 64); the page's score
            upper bound is sum_d max(q_d min_d, q_d max_d) (Quest-style); top-k by bound
Granularity:
  qh        every query head chooses its own pages
  kvh       the G = Hq / Hkv query heads of a KV head share one page set (what a GQA kernel that loads each KV
            tile once, like pthfa, would run): oracle ranks pages by the mass summed over the G heads; bounds by
            the max over the G heads of the bound (raw logit units: the peaky heads, whose mass a page choice
            decides, have the large bounds)
Query sets: 'tail' = the last --q-tail prompt positions plus every decode position (the attn-4 use case: tokens
generated at full depth); 'spread' = the depth-curve positions (reported, not gated).

Per (layer, P, pct, selector, granularity) over the tail's (query, head) rows: mean, p10 and min captured mass and
the share of rows >= 0.99. Verdict, one ATTN4 line: GO at the smallest pct (and its P) where bounds/kvh reaches a
mean >= --gate (0.99) in EVERY layer; else NO-GO, with whether the oracle would have met it (if not, no page
selector can at that budget). JSON with every cell via --out.

  python bench/attn4_mass.py <dump dir> [--pages 16,64] [--pcts 2,5,10] [--threads 4] [--out mass.json]
  python bench/attn4_mass.py --self-test     synthetic dump with planted structure; checks the arithmetic
"""
import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch


def bf16_load(path, mmap=True):
    a = np.load(path, mmap_mode="r" if mmap else None)
    t = torch.from_numpy(np.array(a, copy=True).view(np.int16)).view(torch.bfloat16)
    return t.float()


def select_mass(score, pm, vis, k_row):
    """score/pm [R, NP]; vis [R] visible pages; k_row [R] budget. Forced: page 0 and page vis-1."""
    R, NP = score.shape
    idx = torch.arange(NP)
    sc = score.clone()
    sc[idx[None, :] >= vis[:, None]] = -math.inf
    sc[:, 0] = math.inf
    sc[torch.arange(R), vis - 1] = math.inf
    kmax = int(k_row.max())
    top = sc.topk(min(kmax, NP), dim=1).indices
    keep = torch.arange(top.shape[1])[None, :] < k_row[:, None]
    return (pm.gather(1, top) * keep).sum(1)


def page_reduce(x, P, NP, fill, op):
    """x [N, ...] -> [NP, ...] by op over pages of P rows; rows past N filled with `fill`."""
    N = x.shape[0]
    pad = NP * P - N
    if pad:
        x = torch.cat([x, torch.full((pad,) + tuple(x.shape[1:]), fill, dtype=x.dtype)], 0)
    x = x.view(NP, P, *x.shape[1:])
    return op(x)


def analyse_layer(K, Q, qpos, scale, G, pages, pcts, mask_sets):
    """K [N, Hkv, D]; Q [nq, Hq, D]; qpos [nq]. Returns {(P, pct, sel, gran, set): tensor of row masses}."""
    N, Hkv, D = K.shape
    nq = Q.shape[0]
    out = {}
    for h in range(Hkv):
        Kh = K[:, h]                                       # [N, D]
        Qh = Q[:, h * G:(h + 1) * G].reshape(nq * G, D)    # rows: query-major, head-minor
        pos = qpos.repeat_interleave(G)
        s = (Qh @ Kh.T) * scale
        s[torch.arange(N)[None, :] > pos[:, None]] = -math.inf
        p = torch.softmax(s, dim=1)
        del s
        for P in pages:
            NP = (N + P - 1) // P
            pm = page_reduce(p.T, P, NP, 0.0, lambda x: x.sum(1)).T          # [R, NP]
            kmin = page_reduce(Kh, P, NP, math.inf, lambda x: x.amin(1))     # [NP, D]
            kmax = page_reduce(Kh, P, NP, -math.inf, lambda x: x.amax(1))
            kmin = torch.where(torch.isfinite(kmin), kmin, torch.zeros_like(kmin))
            kmax = torch.where(torch.isfinite(kmax), kmax, torch.zeros_like(kmax))
            bound = Qh.clamp(min=0) @ kmax.T + Qh.clamp(max=0) @ kmin.T      # [R, NP]
            vis = pos // P + 1
            pm_g = pm.view(nq, G, NP).sum(1).repeat_interleave(G, 0)
            b_g = bound.view(nq, G, NP).amax(1).repeat_interleave(G, 0)
            for pct in pcts:
                k_row = torch.clamp(torch.ceil(vis.double() * pct / 100.0).long(), min=2)
                k_row = torch.minimum(k_row, vis)
                for sel, gran, score in (("oracle", "qh", pm), ("bounds", "qh", bound),
                                         ("oracle", "kvh", pm_g), ("bounds", "kvh", b_g)):
                    m = select_mass(score, pm, vis, k_row)
                    for name, qmask in mask_sets.items():
                        key = (P, pct, sel, gran, name)
                        out.setdefault(key, []).append(m.view(nq, G)[qmask].reshape(-1))
        del p
    return {k: torch.cat(v) for k, v in out.items()}


def stats(m):
    m = m.double()
    return {"n": int(m.numel()), "mean": float(m.mean()), "p10": float(torch.quantile(m, 0.10)),
            "min": float(m.min()), "ge99": float((m >= 0.99).double().mean())}


def run(dump, pages, pcts, gate, out_path):
    meta = json.load(open(os.path.join(dump, "meta.json")))
    n_prompt, q_tail = meta["prompt_tokens"], meta["q_tail"]
    layers = sorted(meta["layers"])
    print(f"attn4_mass: {dump}: {len(layers)} layers, prompt {n_prompt}, gen {meta['gen_tokens']}, "
          f"pages {pages}, pcts {pcts}", flush=True)
    cells = {}
    for ln in layers:
        lm = meta["layers"][ln]
        npos = meta["expected_positions"]
        K = bf16_load(os.path.join(dump, f"L{ln}.k.npy"))[:npos]
        Q = bf16_load(os.path.join(dump, f"L{ln}.q.npy"), mmap=False)
        qpos = torch.from_numpy(np.load(os.path.join(dump, f"L{ln}.qpos.npy")))
        G = lm["hq"] // lm["hkv"]
        sets = {"tail": qpos >= n_prompt - q_tail, "spread": qpos < n_prompt - q_tail}
        sets = {k: v for k, v in sets.items() if bool(v.any())}
        res = analyse_layer(K, Q, qpos, lm["scale"], G, pages, pcts, sets)
        for key, m in res.items():
            cells[(ln,) + key] = stats(m)
        line = " ".join(f"P{P}/{pct}%:{cells[(ln, P, pct, 'bounds', 'kvh', 'tail')]['mean']:.4f}"
                        f"({cells[(ln, P, pct, 'oracle', 'kvh', 'tail')]['mean']:.4f})"
                        for P in pages for pct in pcts if (ln, P, pct, 'bounds', 'kvh', 'tail') in cells)
        print(f"L{ln} tail bounds/kvh mean (oracle/kvh): {line}", flush=True)
        del K, Q

    print("\nper layer, tail queries: mean / p10 / share>=0.99 of captured mass")
    for sel, gran in (("bounds", "kvh"), ("bounds", "qh"), ("oracle", "kvh"), ("oracle", "qh")):
        print(f"-- {sel}/{gran}")
        print("layer " + " ".join(f"{f'P{P} {pct}%':>21s}" for P in pages for pct in pcts))
        for ln in layers:
            row = []
            for P in pages:
                for pct in pcts:
                    c = cells.get((ln, P, pct, sel, gran, "tail"))
                    row.append(f"{c['mean']:.4f}/{c['p10']:.3f}/{c['ge99']:.2f}" if c else f"{'-':>21s}")
            print(f"L{ln:>4s} " + " ".join(f"{r:>21s}" for r in row))
    if any(k[-1] == "spread" for k in cells):
        print("-- depth curve (spread queries), bounds/kvh mean, min over layers:")
        for P in pages:
            print("   " + " ".join(f"P{P} {pct}%: {min(cells[(ln, P, pct, 'bounds', 'kvh', 'spread')]['mean'] for ln in layers):.4f}"
                                  for pct in pcts))

    def all_layers(sel, gran, P, pct):
        return all(cells[(ln, P, pct, sel, gran, "tail")]["mean"] >= gate for ln in layers)
    verdict, cfg = "NO-GO", None
    for pct in sorted(pcts):
        ok = [P for P in pages if all_layers("bounds", "kvh", P, pct)]
        if ok:
            verdict, cfg = "GO", (max(ok), pct)
            break
    oracle_ok = [(P, pct) for pct in sorted(pcts) for P in pages if all_layers("oracle", "kvh", P, pct)]
    worst = {(P, pct): min((cells[(ln, P, pct, "bounds", "kvh", "tail")]["mean"], ln) for ln in layers)
             for P in pages for pct in pcts}
    w_str = " ".join(f"P{P}/{pct}%:{v:.4f}@L{ln}" for (P, pct), (v, ln) in sorted(worst.items()))
    line = (f"ATTN4 verdict={verdict}" + (f" at P={cfg[0]} k={cfg[1]}% of pages" if cfg else "")
            + f" gate=mean>={gate} in all {len(layers)} layers (bounds/kvh, tail queries); worst layer per cell: {w_str}; "
            f"oracle/kvh meets the gate at {oracle_ok[0] if oracle_ok else 'no tested budget'}")
    print(line, flush=True)
    if out_path:
        with open(out_path, "w") as f:
            json.dump({"meta": {k: meta[k] for k in ("prompt_tokens", "gen_tokens", "q_tail")}, "gate": gate,
                       "verdict": verdict, "config": cfg, "oracle_ok": oracle_ok,
                       "cells": [dict(zip(("layer", "P", "pct", "sel", "gran", "set"), k), **v)
                                 for k, v in sorted(cells.items())]}, f, indent=1)
    return 0


def self_test():
    """Planted structure: one query head per KV head attends almost only to one far page (a retrieval head); the
    others to the sink and local pages. Oracle at 2% must capture ~1; bounds must find the planted page."""
    g = torch.Generator().manual_seed(0)
    N, Hkv, G, D, P = 4096, 2, 3, 64, 64
    K = torch.randn(N, Hkv, D, generator=g) * 0.3
    nq = 4
    qpos = torch.arange(N - nq, N)
    Q = torch.randn(nq, Hkv * G, D, generator=g) * 0.1
    far = 1000
    K[far:far + P, :, :] += 0.0
    K[far:far + P, :, 0] = 4.0                      # a page with a large channel-0 key
    Q[:, 0::G, 0] = 40.0                            # head 0 of each KV head looks for it (logit 20)
    d = tempfile.mkdtemp()
    to16 = lambda t: t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    meta = {"prompt_tokens": N - nq + 1, "gen_tokens": nq, "expected_positions": N, "q_tail": 8, "layers": {}}
    for ln in ("03", "07"):
        np.save(os.path.join(d, f"L{ln}.k.npy"), to16(K))
        np.save(os.path.join(d, f"L{ln}.q.npy"), to16(Q))
        np.save(os.path.join(d, f"L{ln}.qpos.npy"), qpos.numpy())
        meta["layers"][ln] = {"hq": Hkv * G, "hkv": Hkv, "d": D, "scale": D ** -0.5, "nq": nq, "seen": N}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    res = analyse_layer(bf16_load(os.path.join(d, "L03.k.npy")), bf16_load(os.path.join(d, "L03.q.npy")), qpos,
                        D ** -0.5, G, [P], [5, 50], {"tail": torch.ones(nq, dtype=torch.bool)})
    o = res[(P, 5, "oracle", "qh", "tail")].view(-1, G)
    b = res[(P, 5, "bounds", "qh", "tail")].view(-1, G)
    full = res[(P, 50, "oracle", "qh", "tail")]
    print(f"self-test: retrieval head oracle {o[:, 0].min():.4f} bounds {b[:, 0].min():.4f}; diffuse heads oracle "
          f"{o[:, 1:].mean():.4f}; 50% oracle min {full.min():.4f}")
    assert o[:, 0].min() > 0.95 and b[:, 0].min() > 0.95, "the planted page must be found"
    assert (o >= b - 1e-6).all(), "the oracle is an upper bound"
    assert o[:, 1:].mean() < 0.5, "diffuse heads cannot be captured by 5% of pages"
    assert full.min() > 0.45
    rc = run(d, [P], [5, 50], 0.99, None)
    print("ATTN4 SELF-TEST OK")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--pages", default="16,64")
    ap.add_argument("--pcts", default="2,5,10")
    ap.add_argument("--gate", type=float, default=0.99)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.self_test:
        return self_test()
    if not a.dump:
        ap.error("dump dir required")
    return run(a.dump, [int(x) for x in a.pages.split(",")], [int(x) if float(x).is_integer() else float(x) for x in a.pcts.split(",")], a.gate, a.out)


if __name__ == "__main__":
    sys.exit(main())
