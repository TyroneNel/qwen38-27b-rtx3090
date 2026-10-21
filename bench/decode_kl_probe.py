"""Decode path against prefill path: how far apart the server's two logprobs for the same token are.

evals-1 in docs/research/deepinfer/evals.md. Decode-only bugs ("benchmarks great, outputs garbage")
show up here as a KL rise long before an accuracy battery moves.

  1. For each of N fixed prompts, a greedy /v1/completions from the prompt's token ids with
     `logprobs: 0` and `return_tokens_as_token_ids: true` for --gen tokens (ignore_eos, so every
     prompt decodes exactly --gen). That gives each generated token's logprob as the DECODE path
     (decode steps, drafts and verify blocks, the decode attention kernels) computed it.
  2. The prompt ids plus the generated ids go back with `max_tokens: 1, prompt_logprobs: 0`. vLLM
     skips reading the prefix cache for any request with prompt_logprobs (SamplingParams sets
     skip_reading_prefix_cache), so this is a fresh (chunked) PREFILL of the same sequence. Entry
     P+j of `prompt_logprobs` is keyed by token id as a string and holds the logprob prefill gives
     generated token j. -inf comes back clamped to -9999.
  3. With d = logp_decode - logp_prefill over the generated positions: k3 KL mean(exp(d) - 1 - d)
     (DeepInfer/deepinfer/testing.py), mean and max |d|, RMSE, the cosine of the two logprob
     vectors, and the fraction of positions where prefill would not have picked the decoded token
     (its rank there is not 1).

The baseline is not zero: decode and chunked prefill legitimately differ. Calibrate per mode (two
runs give the repeat spread) and gate at about 2x that with --max-k3.

Prompts. Thinking is off, through the chat template: /tokenize builds the ids from a user message
with enable_thinking false, exactly what /v1/chat/completions would send, and step 1 replays those
ids. Each prompt is a slice of the repo's docs (docs/**/*.md, sorted, under the repo this file sits
in, so /app/docs in the image) starting at an evenly spaced offset, plus one of eight fixed
questions. The slice is sized against the server's own tokenizer until the whole prompt is within
1% of --ctx tokens. Each prompt's sha256 over its ids goes in the JSON so two runs can be checked
for the same prompts; a different docs tree gives different prompts, so --corpus pins a file.

--negative-control also pairs decode position j with prefill position j+1 (the tokens shifted by
one) over the same data. That metric must come out above 2x the aligned one, or the probe cannot
tell a broken path from a good one and exits 3.

  docker exec <c> /app/venv/bin/python /app/bench/decode_kl_probe.py --ctx 4000 --negative-control \\
      --label B --out /app/bench/results/decode_kl_B.json

Prints one line per prompt, then
  decode_kl_probe: prompts=8 ctx=4003 gen=512 k3=... mean_abs=... max_abs=... rmse=... cos=... rank_ne1=...
Exit 0 (a measurement), 1 if --max-k3 is given and the pooled k3 is above it, 2 if the run is
invalid (no server, token ids or prompt_logprobs not returned, ctx + gen over max_model_len), 3 if
the negative control did not rise.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import urllib.error

import harness

QUESTIONS = [
    "Summarise the document above in detail, section by section.",
    "List every command, flag and environment variable the document mentions, and explain each.",
    "What are the main measurements in the document? Quote the numbers and explain them.",
    "Rewrite the first part of the document as a tutorial for a new contributor.",
    "Which claims in the document are inferred rather than measured? Explain why for each.",
    "Explain the design decisions in the document and the alternatives it rejects.",
    "Write a FAQ of ten questions a reader would ask about this document, with answers.",
    "Continue the document with the next section it would plausibly contain.",
]
D_CLIP = 50.0  # exp(d) overflows long before a -9999 (clamped -inf) difference; clip and count


def tokenize_chat(content):
    """The prompt ids /v1/chat/completions would build for one user message, thinking off."""
    r = harness.post("/tokenize", {"messages": [{"role": "user", "content": content}],
                                   "add_generation_prompt": True,
                                   "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
    return r["tokens"], r["max_model_len"]


def ids_from_tokens(tokens):
    """`return_tokens_as_token_ids` renders each token as "token_id:1234"; None if it did not."""
    out = []
    for t in tokens:
        if not isinstance(t, str) or not t.startswith("token_id:"):
            return None
        out.append(int(t[9:]))
    return out


def load_corpus(path):
    if path:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    files = sorted(glob.glob(os.path.join(harness.REPO, "docs", "**", "*.md"), recursive=True))
    text = []
    for p in files:
        with open(p, encoding="utf-8", errors="replace") as f:
            text.append(f.read())
    return "\n\n".join(text)


def build_prompt(corpus, i, n, ctx):
    """Prompt i: the corpus from offset i*len/n (next line start), sized to ctx tokens in total."""
    start = (i * len(corpus)) // n
    nl = corpus.find("\n", start)
    start = nl + 1 if 0 <= nl < start + 2000 else start
    ring = corpus[start:] + "\n\n" + corpus
    while len(ring) < 8 * ctx:
        ring += "\n\n" + corpus
    q = QUESTIONS[i % len(QUESTIONS)]
    overhead = None
    nch = int(ctx * 3.5)
    for _ in range(6):
        ids, max_len = tokenize_chat("Document:\n\n" + ring[:nch] + "\n\n" + q)
        if abs(len(ids) - ctx) <= max(8, ctx // 100):
            break
        if overhead is None:
            overhead = len(tokenize_chat("Document:\n\n\n\n" + q)[0])
        got = max(1, len(ids) - overhead)
        nch = max(1, int(nch * (ctx - overhead) / got))
    return ids, max_len


def decode(prompt_ids, gen):
    r = harness.post("/v1/completions", {"prompt": prompt_ids, "max_tokens": gen, "temperature": 0,
                                         "logprobs": 0, "return_tokens_as_token_ids": True,
                                         "ignore_eos": True})
    ch = r["choices"][0]
    lp = ch.get("logprobs") or {}
    ids = ids_from_tokens(lp.get("tokens") or [])
    return ids, lp.get("token_logprobs") or [], ch.get("finish_reason")


def prefill(seq_ids, p_len, gen_ids):
    """The prefill logprob of each generated token, or None where the server returned none."""
    r = harness.post("/v1/completions", {"prompt": seq_ids, "max_tokens": 1, "temperature": 0,
                                         "prompt_logprobs": 0})
    pl = r["choices"][0].get("prompt_logprobs")
    if not pl or len(pl) < p_len + len(gen_ids):
        return None, None
    lps, ranks = [], []
    for j, tok in enumerate(gen_ids):
        e = pl[p_len + j] or {}
        v = e.get(str(tok))
        if v is None:
            return None, None
        lps.append(v["logprob"] if isinstance(v, dict) else v)
        ranks.append(v.get("rank") if isinstance(v, dict) else None)
    return lps, ranks


def stats(x, y):
    """k3 KL, mean/max |d|, RMSE and cosine over paired logprobs x (decode) and y (prefill)."""
    n = len(x)
    if n == 0:
        return None
    k3 = s_abs = s_sq = mx = dot = nx = ny = 0.0
    clipped = 0
    for a, b in zip(x, y):
        d = a - b
        if abs(d) > D_CLIP:
            clipped += 1
        dc = max(-D_CLIP, min(D_CLIP, d))
        k3 += math.exp(dc) - 1.0 - dc
        s_abs += abs(d)
        s_sq += d * d
        mx = max(mx, abs(d))
        dot += a * b
        nx += a * a
        ny += b * b
    cos = dot / math.sqrt(nx * ny) if nx > 0 and ny > 0 else (1.0 if nx == ny else 0.0)
    return {"n": n, "k3": k3 / n, "mean_abs": s_abs / n, "max_abs": mx, "rmse": math.sqrt(s_sq / n),
            "cos": cos, "clipped": clipped}


def fmt(s):
    return (f"k3={s['k3']:.3e} mean_abs={s['mean_abs']:.3e} max_abs={s['max_abs']:.3e} "
            f"rmse={s['rmse']:.3e} cos={s['cos']:.6f}")


def main():
    try:
        return run()
    except urllib.error.HTTPError as e:
        print(f"decode_kl_probe: INVALID HTTP {e.code} from {e.url}: {e.read()[:500]!r}")
        return 2


def run():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4000, help="prompt length in tokens (4000-16000 in the fast tiers)")
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--corpus", help="text file to slice prompts from (default: the repo's docs/**/*.md)")
    ap.add_argument("--negative-control", action="store_true")
    ap.add_argument("--max-k3", type=float, help="FAIL (exit 1) if the pooled k3 is above this")
    ap.add_argument("--label", default="", help="mode label stored in the JSON")
    ap.add_argument("--out", help="write the JSON here")
    a = ap.parse_args()

    try:
        corpus = load_corpus(a.corpus)
    except OSError as e:
        print(f"decode_kl_probe: INVALID corpus: {e}")
        return 2
    if len(corpus) < 1000:
        print(f"decode_kl_probe: INVALID corpus has {len(corpus)} chars")
        return 2
    try:
        model = harness.model()
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
        print(f"decode_kl_probe: INVALID no server at {harness.base_url()}: {e}")
        return 2

    rows, X, Y, R, NX, NY = [], [], [], [], [], []
    for i in range(a.prompts):
        p_ids, max_len = build_prompt(corpus, i, a.prompts, a.ctx)
        if len(p_ids) + a.gen > max_len:
            print(f"decode_kl_probe: INVALID prompt {i}: {len(p_ids)} + {a.gen} tokens > max_model_len {max_len}")
            return 2
        g_ids, g_lp, finish = decode(p_ids, a.gen)
        if g_ids is None:
            print("decode_kl_probe: INVALID server did not honour return_tokens_as_token_ids")
            return 2
        if not g_ids or len(g_lp) != len(g_ids) or any(v is None for v in g_lp):
            print(f"decode_kl_probe: INVALID prompt {i}: {len(g_ids)} ids, {len(g_lp)} logprobs")
            return 2
        p_lp, ranks = prefill(p_ids + g_ids, len(p_ids), g_ids)
        if p_lp is None:
            print(f"decode_kl_probe: INVALID prompt {i}: prompt_logprobs missing for the generated positions")
            return 2
        s = stats(g_lp, p_lp)
        ne1 = sum(1 for r in ranks if r is not None and r != 1)
        row = {"i": i, "prompt_tokens": len(p_ids),
               "prompt_sha": hashlib.sha256(json.dumps(p_ids).encode()).hexdigest()[:16],
               "gen_tokens": len(g_ids), "finish_reason": finish, **s, "rank_ne1": ne1 / len(g_ids),
               "decode_logprobs": g_lp, "prefill_logprobs": p_lp, "prefill_ranks": ranks}
        if a.negative_control:
            neg = stats(g_lp[:-1], p_lp[1:])
            if neg:
                row["negative_control"] = neg
                NX += g_lp[:-1]
                NY += p_lp[1:]
        rows.append(row)
        X += g_lp
        Y += p_lp
        R += ranks
        print(f"decode_kl_probe: prompt {i} ctx={len(p_ids)} gen={len(g_ids)} {fmt(s)} rank_ne1={ne1}")

    pooled = stats(X, Y)
    pooled["rank_ne1"] = sum(1 for r in R if r is not None and r != 1) / len(R)
    ctx_mean = round(sum(r["prompt_tokens"] for r in rows) / len(rows))
    line = (f"decode_kl_probe: prompts={len(rows)} ctx={ctx_mean} gen={a.gen} {fmt(pooled)} "
            f"rank_ne1={pooled['rank_ne1']:.4f}")
    rc = 0
    result = {"label": a.label, "model": model, "ctx": a.ctx, "gen": a.gen, "pooled": pooled, "prompts": rows}
    if a.negative_control:
        neg = stats(NX, NY)
        ok = neg is not None and neg["k3"] > 2 * pooled["k3"] and neg["k3"] > pooled["k3"] + 1e-9
        result["negative_control"] = {**(neg or {}), "rose": ok}
        line += f" neg_k3={neg['k3'] if neg else float('nan'):.3e} neg={'PASS' if ok else 'FAILED'}"
        if not ok:
            rc = 3
    if rc == 0 and a.max_k3 is not None:
        verdict = pooled["k3"] <= a.max_k3
        line += f" gate={a.max_k3:.3e} {'PASS' if verdict else 'FAIL'}"
        rc = 0 if verdict else 1
    if pooled["clipped"]:
        line += f" clipped={pooled['clipped']}"
    print(line)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
