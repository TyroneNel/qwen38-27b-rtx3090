"""Teacher-forced logprobs of one server against a reference server's: how much a KV-cache format moves the model.

decode_kl_probe.py compares a server's decode path with its own prefill path; both read the same quantised cache,
so it cannot see the cache format's error. This tool can (attn-5, docs/research/deepinfer/attention-kv.md): run the
SAME prompt token ids through two boots (a reference with an unquantised bf16 KV cache, and the arm under test) with
prompt_logprobs = top-k, and compare the two distributions at every prompt position. A single fresh prefill reads
every earlier position from the cache the chunk just wrote (vLLM's prompt_logprobs request skips prefix-cache reads),
so the arm's numbers carry its cache format at every depth.

  python bench/kv_ref_kl.py dump --ctx 4000 --prompts 8 [--ctx2 16000 --prompts2 2] [--topk 20]
         [--ids-from REF.json] --label ARM --out ARM.json
      Builds the prompts as decode_kl_probe.py does (docs corpus slices + a question, chat template, thinking off),
      or replays the prompt ids stored in REF.json (use this for every arm, so all arms score identical tokens).
  python bench/kv_ref_kl.py compare REF.json ARM.json [ARM2.json ...]
      Per arm against REF, over every scored position (1 .. len-1 of each prompt):
        kl_topk   sum over REF's top-k tokens of p_ref (lp_ref - lp_arm), lp_arm missing from the arm's top-k taken as
                  the arm's k-th logprob (a lower bound on its loss there): KL(ref || arm) restricted to ref's top-k;
        k3        mean(exp(d) - 1 - d), d = lp_arm - lp_ref of the actual next token (decode_kl_probe's estimator);
        dnll      mean(lp_ref - lp_arm) of the actual token (perplexity ratio = exp(dnll));
        top1      fraction of positions where the two top-1 tokens agree;
      overall and by depth quarter of the prompt. One 'KVREF' line per arm.
Exit: dump 0 = written, 2 = invalid; compare 0.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import urllib.error

import harness
from decode_kl_probe import build_prompt, load_corpus

D_CLIP = 50.0


def dump_one(ids, topk):
    r = harness.post("/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": topk},
                     timeout=3600)
    pl = r["choices"][0].get("prompt_logprobs")
    if not pl or len(pl) != len(ids):
        raise ValueError(f"prompt_logprobs: got {len(pl) if pl else 0} entries for {len(ids)} prompt tokens")
    act, tops = [], []
    for j in range(1, len(ids)):
        e = pl[j] or {}
        v = e.get(str(ids[j]))
        act.append(None if v is None else (v["logprob"] if isinstance(v, dict) else v))
        items = []
        for tid, val in e.items():
            lp = val["logprob"] if isinstance(val, dict) else val
            rank = val.get("rank") if isinstance(val, dict) else None
            items.append((rank if rank is not None else 10 ** 9, int(tid), lp))
        items.sort()
        tops.append([[t, round(lp, 6)] for _, t, lp in items[:topk]])
    return act, tops


def run_dump(a):
    if a.ids_from:
        with open(a.ids_from) as f:
            prompts = [p["ids"] for p in json.load(f)["prompts"]]
    else:
        corpus = load_corpus(a.corpus)
        prompts = [build_prompt(corpus, i, a.prompts, a.ctx)[0] for i in range(a.prompts)]
        if a.prompts2:
            prompts += [build_prompt(corpus, i, a.prompts2, a.ctx2)[0] for i in range(a.prompts2)]
    out = {"label": a.label, "model": harness.model(), "topk": a.topk, "prompts": []}
    for i, ids in enumerate(prompts):
        act, tops = dump_one(ids, a.topk)
        out["prompts"].append({"i": i, "sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16], "ids": ids,
                               "actual": act, "top": tops})
        miss = sum(1 for x in act if x is None)
        print(f"kv_ref_kl dump: prompt {i} tokens={len(ids)} actual-missing={miss}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"kv_ref_kl dump: {len(prompts)} prompts -> {a.out}")
    return 0


def acc_new():
    return {"n": 0, "kl": 0.0, "k3": 0.0, "dnll": 0.0, "top1": 0, "nk3": 0, "clipped": 0}


def compare_pair(ref, arm):
    if [p["sha"] for p in ref["prompts"]] != [p["sha"] for p in arm["prompts"]]:
        raise ValueError(f"{arm.get('label')}: prompt ids differ from the reference (dump with --ids-from REF.json)")
    total = acc_new()
    quarters = [acc_new() for _ in range(4)]
    for pr, pa in zip(ref["prompts"], arm["prompts"]):
        n = len(pr["top"])
        for j in range(n):
            tr, ta = pr["top"][j], pa["top"][j]
            if not tr or not ta:
                continue
            amap = {t: lp for t, lp in ta}
            floor = min(lp for _, lp in ta)
            kl = sum(math.exp(lr) * (lr - amap.get(t, floor)) for t, lr in tr)
            buckets = (total, quarters[min(3, 4 * j // n)])
            xr, xa = pr["actual"][j], pa["actual"][j]
            for b in buckets:
                b["n"] += 1
                b["kl"] += kl
                b["top1"] += int(tr[0][0] == ta[0][0])
                if xr is not None and xa is not None:
                    d = xa - xr
                    b["clipped"] += int(abs(d) > D_CLIP)
                    dc = max(-D_CLIP, min(D_CLIP, d))
                    b["k3"] += math.exp(dc) - 1.0 - dc
                    b["dnll"] += -d
                    b["nk3"] += 1
    fin = lambda b: {"n": b["n"], "kl_topk": b["kl"] / max(b["n"], 1), "k3": b["k3"] / max(b["nk3"], 1),
                     "dnll": b["dnll"] / max(b["nk3"], 1), "top1": b["top1"] / max(b["n"], 1),
                     "clipped": b["clipped"]}
    return fin(total), [fin(q) for q in quarters]


def run_compare(a):
    with open(a.ref) as f:
        ref = json.load(f)
    for path in a.arms:
        with open(path) as f:
            arm = json.load(f)
        tot, qs = compare_pair(ref, arm)
        print(f"KVREF {arm.get('label') or path} vs {ref.get('label') or a.ref}: n={tot['n']} "
              f"kl_topk={tot['kl_topk']:.3e} k3={tot['k3']:.3e} dnll={tot['dnll']:+.3e} "
              f"(ppl x{math.exp(tot['dnll']):.4f}) top1={tot['top1']:.4f} clipped={tot['clipped']}")
        print("      by depth quarter kl_topk: " + " ".join(f"{q['kl_topk']:.3e}" for q in qs)
              + " | top1: " + " ".join(f"{q['top1']:.4f}" for q in qs))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--ctx", type=int, default=4000)
    d.add_argument("--prompts", type=int, default=8)
    d.add_argument("--ctx2", type=int, default=16000)
    d.add_argument("--prompts2", type=int, default=0)
    d.add_argument("--topk", type=int, default=20)
    d.add_argument("--corpus")
    d.add_argument("--ids-from")
    d.add_argument("--label", default="")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("arms", nargs="+")
    a = ap.parse_args()
    try:
        return run_dump(a) if a.cmd == "dump" else run_compare(a)
    except urllib.error.HTTPError as e:
        print(f"kv_ref_kl: INVALID HTTP {e.code}: {e.read()[:500]!r}")
        return 2
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        print(f"kv_ref_kl: INVALID {e.__class__.__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
