"""Acceptance by domain (evals-9): tokens per step over DeepSpec's nine eval sets.

The panel is `bench/accept_panel_data.py`'s JSONL: 50 first-turn prompts from each of gsm8k,
math500, aime25 (30), humaneval, mbpp, livecodebench, mt-bench, alpaca and arena-hard-v2,
the same prompts DeepSpec's eval.py reads. Each prompt is sent through the chat template
(thinking off, as DeepSpec's evaluator does; `--think` turns it on) and measured two ways.

FORCED ARM (default). labd_accept.py's prefix-resync teacher forcing, one item at a time:
a greedy continuation of up to --max-tokens is captured once and frozen under --targets,
then replayed in --chunk-token chunks, each chunk a raw-id /v1/completions request on
`prompt + target[:off]`, with the vllm:spec_decode_* counters diffed around it. A chunk
whose output differs from the target is dirty and left out of the numbers (--keep-dirty
keeps it). Read labd_accept.py's docstring before trusting a comparison: freeze the
targets once and reuse them for every configuration, the clean-only number is biased
slightly high, each chunk resets the adaptive drafting state, and /metrics is
process-global, so run this alone against the server.

SAMPLED ARM (--sampled). One free-running request per item at --temperature (default 1.0
with top_p 1 and top_k off: DeepSpec eval.py's plain temperature sampling, not the
server's generation_config defaults), seeded per item, with the same counters diffed
around it. Each configuration then follows its own trajectory, so this arm is the
realistic number and the forced arm the controlled one.

UNITS. HyperQwen's tokens/step is 1 + accepted / drafts: each verify step emits its
accepted drafts plus one target token. DeepSpec's "acceptance length" is the mean over
proposals of accepted + 1 (base_evaluator.generate_decoding_sample), the same quantity,
except that a proposal ending on a stop token counts accepted without the +1 (one
proposal per finished item at most). So deepspec.accept_len = tok_per_step. DeepSpec's
"#propose" k+1 is slots_per_step+1 here, its verify_rate is
tok_per_step / (slots_per_step + 1), and its accept_rate@i (the share of proposals whose
first i+1 drafts were all accepted) is accepted_per_pos[i] / drafts here, exact when every
step drafts all slots; with a dynamic block the denominator here also counts steps that
did not draft position i, so the rate reads low. Decode tok/s counts tokens after the first
SSE frame over first frame to last.

STATISTICS. Every number is a ratio of sums over the items of a domain. 95% CIs are
percentile bootstraps over items (--boot resamples, fixed --seed); the pooled row
resamples within each domain, and the macro row is the unweighted mean of the domains.

Usage (inside the serving container, against a server with speculative decoding on):

  venv/bin/python bench/accept_panel.py <tag> --data /shared/panel.jsonl --out /out/panel.json
      [--max-tokens 256] [--chunk 128] [--think] [--sampled] [--temperature 1.0]
      [--domains gsm8k,mbpp] [--per-domain N] [--targets DIR] [--recapture] [--keep-dirty]
      [--boot 2000] [--seed 980406] [--base URL] [--model NAME]

--targets defaults to targets/ beside --out. Exits 0 with numbers, 1 if the server ignores
return_tokens_as_token_ids, 2 if no item produced a counted speculative step (no server,
speculation off, every request failed, or every chunk dirty).
"""
import argparse
import hashlib
import http.client
import json
import os
import random
import re
import sys
import time

import harness

SCALARS = ("vllm:spec_decode_num_drafts_total",
           "vllm:spec_decode_num_draft_tokens_total",
           "vllm:spec_decode_num_accepted_tokens_total")
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
POS_RE = re.compile(r'position="(\d+)"')


class NoTokenIds(Exception):
    pass


def metrics():
    """(drafts, draft_slots, accepted, {position: accepted}) summed over engines."""
    scal = {k: 0.0 for k in SCALARS}
    pos = {}
    for name, labels, val in harness.samples():
        if name in scal:
            scal[name] += val
        elif name == PER_POS:
            m = POS_RE.search(labels)
            if m:
                pos[int(m.group(1))] = pos.get(int(m.group(1)), 0.0) + val
    return scal[SCALARS[0]], scal[SCALARS[1]], scal[SCALARS[2]], pos


def tokenize_chat(content, think):
    """The prompt token ids /v1/chat/completions would build for this one user turn."""
    r = harness.post("/tokenize", {"messages": [{"role": "user", "content": content}],
                                   "add_generation_prompt": True,
                                   "chat_template_kwargs": {"enable_thinking": think}}, timeout=600)
    return r["tokens"], r["max_model_len"]


def generate(prompt_ids, max_tokens, sampling=None):
    """Stream a raw-id completion. -> (ids, decoded_after_first_frame, decode_seconds).
    Greedy unless sampling (temperature, top_p, top_k, seed) is given."""
    payload = {"prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0,
               "logprobs": 0, "return_tokens_as_token_ids": True, "stream": True}
    if sampling:
        payload.update(sampling)
    toks, first_n, t_first = [], 0, None
    for ev in harness.stream("/v1/completions", payload):
        for ch in ev.get("choices", []):
            got = (ch.get("logprobs") or {}).get("tokens") or []
            if got and t_first is None:
                t_first, first_n = time.perf_counter(), len(got)
            toks.extend(got)
    t_end = time.perf_counter()
    if any(not t.startswith("token_id:") for t in toks):
        raise NoTokenIds()
    ids = [int(t[9:]) for t in toks]
    return ids, len(ids) - first_n, (t_end - t_first) if t_first is not None else 0.0


def new_rec(item, nspec):
    return {"domain": item["domain"], "rank": item["rank"], "steps": 0.0, "slots": 0.0,
            "acc": 0.0, "pos": [0.0] * nspec, "tokens": 0, "dec_tokens": 0, "dec": 0.0,
            "chunks": 0, "clean": 0, "first_bad": None, "error": None}


def add_window(rec, before, after):
    d0, s0, a0, p0 = before
    d1, s1, a1, p1 = after
    rec["steps"] += d1 - d0
    rec["slots"] += s1 - s0
    rec["acc"] += a1 - a0
    for p in set(p0) | set(p1):
        if p >= len(rec["pos"]):
            rec["pos"].extend([0.0] * (p + 1 - len(rec["pos"])))
        rec["pos"][p] += p1.get(p, 0.0) - p0.get(p, 0.0)


def forced(a, item, prompt_ids, max_model_len, nspec):
    sig = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()[:16]
    path = os.path.join(a.targets, f"{item['domain']}_{item['rank']:03d}_{sig}_{a.max_tokens}.json")
    rec = new_rec(item, nspec)
    if a.recapture or not os.path.exists(path):
        ids, _, _ = generate(prompt_ids, a.max_tokens)
        with open(path, "w") as f:
            json.dump({"domain": item["domain"], "rank": item["rank"], "prompt_sig": sig,
                       "captured_by": a.tag, "think": a.think, "max_tokens": a.max_tokens,
                       "ids": ids}, f)
        rec["captured"] = True
    with open(path) as f:
        target = json.load(f)["ids"][: max(0, max_model_len - len(prompt_ids) - 1)]
    rec["prompt_tokens"], rec["target_tokens"] = len(prompt_ids), len(target)
    for off in range(0, len(target), a.chunk):
        want = target[off: off + a.chunk]
        before = metrics()
        got, dec_n, dec = generate(prompt_ids + target[:off], len(want))
        after = metrics()
        clean = got == want
        rec["chunks"] += 1
        rec["clean"] += clean
        if not clean and rec["first_bad"] is None:
            rec["first_bad"] = off + next((i for i, (g, w) in enumerate(zip(got, want)) if g != w),
                                          min(len(got), len(want)))
        if clean or a.keep_dirty:
            add_window(rec, before, after)
            rec["tokens"] += len(got)
            rec["dec_tokens"] += dec_n
            rec["dec"] += dec
    return rec


def sampled(a, item, idx, prompt_ids, max_model_len, nspec):
    rec = new_rec(item, nspec)
    rec["prompt_tokens"] = len(prompt_ids)
    sampling = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
                "seed": a.seed + idx}
    before = metrics()
    got, dec_n, dec = generate(prompt_ids, min(a.max_tokens, max_model_len - len(prompt_ids) - 1),
                               sampling)
    add_window(rec, before, metrics())
    rec.update(tokens=len(got), dec_tokens=dec_n, dec=dec, chunks=1, clean=1)
    return rec


def stats(recs, nspec):
    steps = sum(r["steps"] for r in recs)
    if steps <= 0:
        return None
    slots = sum(r["slots"] for r in recs) / steps
    tps = 1 + sum(r["acc"] for r in recs) / steps
    pos = [sum(r["pos"][i] for r in recs if i < len(r["pos"])) / steps for i in range(nspec)]
    dec = sum(r["dec"] for r in recs)
    return {"tok_per_step": tps, "slots_per_step": slots, "accept_at_pos": pos,
            "decode_tok_s": sum(r["dec_tokens"] for r in recs) / dec if dec > 0 else None}


def flat(s):
    """Scalar views of a stats dict that get a CI."""
    out = {"tok_per_step": s["tok_per_step"], "decode_tok_s": s["decode_tok_s"]}
    out.update({f"accept_at_pos[{i}]": v for i, v in enumerate(s["accept_at_pos"])})
    return out


def ci(groups, nspec, boot, rng, combine):
    """95% percentile CIs. Each bootstrap draw resamples every group (a domain's records)
    independently, with replacement; combine maps the resampled groups to a stats dict."""
    draws = {}
    for _ in range(boot):
        s = combine([[g[rng.randrange(len(g))] for _ in g] for g in groups])
        if s is None:
            continue
        for k, v in flat(s).items():
            if v is not None:
                draws.setdefault(k, []).append(v)
    out = {}
    for k, vs in draws.items():
        vs.sort()
        out[k] = [vs[int(0.025 * (len(vs) - 1))], vs[int(0.975 * (len(vs) - 1))]]
    return out


def macro(per):
    """Unweighted mean of per-domain stats."""
    if not per or any(p is None for p in per):
        return None
    dec = [p["decode_tok_s"] for p in per if p["decode_tok_s"] is not None]
    return {"tok_per_step": sum(p["tok_per_step"] for p in per) / len(per),
            "slots_per_step": sum(p["slots_per_step"] for p in per) / len(per),
            "accept_at_pos": [sum(p["accept_at_pos"][i] for p in per) / len(per)
                              for i in range(len(per[0]["accept_at_pos"]))],
            "decode_tok_s": sum(dec) / len(dec) if dec else None}


def deepspec_view(s):
    return {"accept_len": s["tok_per_step"], "propose": f"{s['slots_per_step']:.2f}+1",
            "verify_rate": s["tok_per_step"] / (s["slots_per_step"] + 1),
            "accept_rate_at": s["accept_at_pos"]}


def summarize(recs, domains, nspec, a):
    rng = random.Random(a.seed)
    ok = [r for r in recs if r["error"] is None and r["steps"] > 0]
    by = {d: [r for r in ok if r["domain"] == d] for d in domains}
    out = {"domains": {}}
    for d in domains:
        mine = [r for r in recs if r["domain"] == d]
        row = {"items": len(by[d]), "items_total": len(mine),
               "errors": sum(r["error"] is not None for r in mine),
               "chunks": sum(r["chunks"] for r in mine),
               "clean_chunks": sum(r["clean"] for r in mine)}
        s = stats(by[d], nspec)
        if s:
            row.update(s, steps=sum(r["steps"] for r in by[d]), deepspec=deepspec_view(s),
                       ci=ci([by[d]], nspec, a.boot, rng, lambda gs: stats(gs[0], nspec)))
        out["domains"][d] = row
    live = [by[d] for d in domains if by[d]]
    if live:
        s = stats(ok, nspec)
        out["pooled"] = dict(s, items=len(ok), deepspec=deepspec_view(s), ci=ci(
            live, nspec, a.boot, rng, lambda gs: stats([r for g in gs for r in g], nspec)))
        s = macro([stats(g, nspec) for g in live])
        out["macro"] = dict(s, items=len(ok), deepspec=deepspec_view(s), ci=ci(
            live, nspec, a.boot, rng, lambda gs: macro([stats(g, nspec) for g in gs])))
    return out


def fmt_ci(row, key):
    lo_hi = row.get("ci", {}).get(key)
    return f"[{lo_hi[0]:.2f},{lo_hi[1]:.2f}]" if lo_hi else "[-]"


def print_table(tag, arm, summ):
    print(f"PANEL {tag} {arm:7s} {'domain':14s} {'n':>3s} {'tok/step':>8s} {'95% CI':13s} "
          f"{'slots':>5s} {'acc@0':>5s} {'acc@1':>5s} {'acc@2':>5s} {'acc@3':>5s} "
          f"{'dec tok/s':>9s} {'clean':>9s}")
    rows = list(summ["domains"].items()) + [(k, summ[k]) for k in ("pooled", "macro") if k in summ]
    for name, r in rows:
        if "tok_per_step" not in r:
            print(f"PANEL {tag} {arm:7s} {name:14s} {r.get('items', 0):3d} no counted steps "
                  f"(errors={r.get('errors')}, clean chunks={r.get('clean_chunks')}/{r.get('chunks')})")
            continue
        acc = (r["accept_at_pos"] + [float("nan")] * 4)[:4]
        dec = r["decode_tok_s"]
        clean = f"{r['clean_chunks']}/{r['chunks']}" if "chunks" in r else ""
        print(f"PANEL {tag} {arm:7s} {name:14s} {r.get('items', 0):3d} {r['tok_per_step']:8.3f} "
              f"{fmt_ci(r, 'tok_per_step'):13s} {r['slots_per_step']:5.2f} "
              + " ".join(f"{v:5.3f}" for v in acc)
              + f" {dec if dec is not None else float('nan'):9.1f} {clean:>9s}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag")
    ap.add_argument("--data", required=True, help="accept_panel_data.py's JSONL")
    ap.add_argument("--out", required=True, help="JSON results path")
    ap.add_argument("--targets", help="frozen greedy targets (default: targets/ beside --out)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--think", action="store_true", help="enable_thinking in the chat template")
    ap.add_argument("--sampled", action="store_true", help="also run the free-running sampled arm")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 disables top-k in vLLM")
    ap.add_argument("--domains", default="", help="comma list; default all in --data")
    ap.add_argument("--per-domain", type=int, default=0, help="first N items of each domain")
    ap.add_argument("--recapture", action="store_true")
    ap.add_argument("--keep-dirty", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=980406)
    ap.add_argument("--base", help="sets VLLM_API")
    ap.add_argument("--model", help="sets VLLM_MODEL")
    a = ap.parse_args()
    if a.base:
        os.environ["VLLM_API"] = a.base
    if a.model:
        os.environ["VLLM_MODEL"] = a.model
    a.targets = a.targets or os.path.join(os.path.dirname(os.path.abspath(a.out)), "targets")
    os.makedirs(a.targets, exist_ok=True)

    with open(a.data, "rb") as f:
        raw = f.read()
    items = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]
    want = [d for d in a.domains.split(",") if d]
    domains = [d for d in dict.fromkeys(i["domain"] for i in items) if not want or d in want]
    items = [i for i in items if i["domain"] in domains and (a.per_domain <= 0 or i["rank"] < a.per_domain)]
    if not items:
        print(f"PANEL {a.tag} INVALID: no items in {a.data} for domains {want or 'all'}")
        return 2
    arms = ["forced"] + (["sampled"] if a.sampled else [])
    errs = (OSError, ValueError, KeyError, http.client.HTTPException)

    # Warm-up before any counter is read, as labd_accept.py does: first steps JIT-compile
    # kernels. Then the per-position counter's cardinality is num_speculative_tokens.
    try:
        warm, _ = tokenize_chat(items[0]["prompt"], a.think)
        for _ in range(2):
            generate(warm, 32)
        nspec = len(metrics()[3])
        model = harness.model()
    except NoTokenIds:
        print(f"PANEL {a.tag} FAIL: server ignores return_tokens_as_token_ids; cannot pin ids")
        return 1
    except errs as e:
        print(f"PANEL {a.tag} INVALID: no server at {harness.base_url()}: {e}")
        return 2
    if nspec == 0:
        print(f"PANEL {a.tag} INVALID: no {PER_POS} on /metrics after warm-up; speculation off?")
        return 2
    print(f"PANEL {a.tag} model={model} nspec={nspec} items={len(items)} domains={','.join(domains)} "
          f"arms={','.join(arms)} max_tokens={a.max_tokens} chunk={a.chunk} think={a.think} "
          f"targets={a.targets}", flush=True)

    prompts = {}
    result = {"tag": a.tag, "model": model, "nspec": nspec, "args": vars(a),
              "data_sha256": hashlib.sha256(raw).hexdigest(), "arms": {}}
    counted = 0
    for arm in arms:
        recs = []
        for idx, item in enumerate(items):
            key = (item["domain"], item["rank"])
            try:
                if key not in prompts:
                    prompts[key] = tokenize_chat(item["prompt"], a.think)
                ids, mml = prompts[key]
                rec = (forced(a, item, ids, mml, nspec) if arm == "forced"
                       else sampled(a, item, idx, ids, mml, nspec))
            except NoTokenIds:
                print(f"PANEL {a.tag} FAIL: server ignores return_tokens_as_token_ids; cannot pin ids")
                return 1
            except errs as e:
                rec = new_rec(item, nspec)
                rec["error"] = f"{type(e).__name__}: {e}"
            recs.append(rec)
            tps = 1 + rec["acc"] / rec["steps"] if rec["steps"] > 0 else float("nan")
            print(f"PANEL {a.tag} {arm:7s} {item['domain']:14s} #{item['rank']:02d} "
                  f"tokens={rec['tokens']:4d} steps={rec['steps']:4.0f} tok/step={tps:5.2f} "
                  f"clean={rec['clean']}/{rec['chunks']}"
                  + (" captured" if rec.get("captured") else "")
                  + (f" ERROR {rec['error']}" if rec["error"] else ""), flush=True)
        summ = summarize(recs, domains, nspec, a)
        counted += sum(r["error"] is None and r["steps"] > 0 for r in recs)
        result["arms"][arm] = {"summary": summ, "items": recs}
        print_table(a.tag, arm, summ)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
    print(f"PANEL {a.tag} wrote {a.out}")
    if counted == 0:
        print(f"PANEL {a.tag} INVALID: no item produced a counted speculative step")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
