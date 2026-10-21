"""Teacher-forced acceptance on THINKING-ON generations, split at `</think>` (evals-4).

labd_accept.py's method, unchanged, on a target captured with `enable_thinking: True`: the target
continuation is frozen once as exact token ids, then replayed in chunks of C tokens as
`prompt = P + T[:off]` to /v1/completions with max_tokens = the chunk, diffing the
vllm:spec_decode_* counters around each chunk and dropping chunks that went off-trajectory. Read
labd_accept.py's docstring for why that is valid and what it cannot control for; all six caveats hold
here.

What this copy adds: the target is cut at its first `</think>` token into a reasoning phase
(T[:b+1], the `</think>` itself included) and an answer phase (T[b+1:]), the chunks never straddle the
cut, and every counter -- steps, draft slots, accepted tokens and the per-position accepted counter
vllm:spec_decode_num_accepted_tokens_per_pos_total -- is kept per phase. So each phase gets its own
tokens/step, its acceptance at each draft position (accepted at position i / steps), its decode rate,
and the task gets the fraction of its tokens spent reasoning.

  * The cut costs one extra resync per task (the reasoning phase's last chunk is short), so steps per
    chunk is slightly lower than labd_accept's at the same --chunk; labd_accept's caveat 1 applies.
  * `</think>` is looked up on the server (/tokenize of the literal text, no special tokens) and must be
    a single token; --think-end-id N overrides that. A target with no `</think>` inside --max-tokens is
    all reasoning, and its task line says so: raise --max-tokens if the answer phase is the question.
  * Targets are cached apart from labd_accept's (labd_think_target_*), because the prompt differs.
  * Besides labd_accept's six document tasks, two document-free tasks (math, plan) are thinking-shaped
    prompts with short prompts; --tasks picks any subset.

Usage (against a server started with SPEC=dflash2 or mtp, inside the container or with the venv):

  venv/bin/python bench/labd_accept_think.py <tag> [--ctx 20000] [--max-tokens 4096] [--chunk 128]
                                             [--block 7] [--tasks copy,code,edit,quote,summary,qa,math,plan]
                                             [--base http://127.0.0.1:18020] [--recapture] [--keep-dirty]
                                             [--model NAME] [--think-end-id N] [--corpus PATH] [--out f.json]

Prints one `LABDTHINK <tag> <task> <phase>` line per task and phase, a `<task> split` line with the
reasoning fraction, and `LABDTHINK <tag> TOTAL` lines per phase. Writes ~/bench/results/labdthink_<tag>.json
unless --out names another file.

Exit: 0 measurement; 1 the server ignores return_tokens_as_token_ids; 2 `</think>` is not one token on this
server, or no speculative step was counted in either phase (no speculator, or every chunk dirty).
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

import harness

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def flag(name):
    return name in sys.argv


if flag("--base"):
    os.environ["VLLM_API"] = arg("--base", "")
if flag("--model"):
    os.environ["VLLM_MODEL"] = arg("--model", "")
CORPUS = os.path.expanduser(arg("--corpus", "~/bench/labd_corpus.txt"))
TARGETS = os.path.expanduser("~/bench/targets")
CTX = int(arg("--ctx", 20000))
MAXTOK = int(arg("--max-tokens", 4096))
CHUNK = int(arg("--chunk", 128))
BLOCK = int(arg("--block", 7))
OUT = arg("--out", os.path.expanduser(f"~/bench/results/labdthink_{TAG}.json"))
PHASES = ("reasoning", "answer")

SCALARS = ("vllm:spec_decode_num_drafts_total",
           "vllm:spec_decode_num_draft_tokens_total",
           "vllm:spec_decode_num_accepted_tokens_total")
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
POS_RE = re.compile(r'position="(\d+)"')


def invalid(msg):
    print(f"LABDTHINK {TAG} INVALID {msg}", flush=True)
    sys.exit(2)


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
    return (scal[SCALARS[0]], scal[SCALARS[1]], scal[SCALARS[2]], pos)


def tokenize_chat(content):
    """The prompt token ids /v1/chat/completions would build for this message, thinking on."""
    r = harness.post("/tokenize", {"messages": [{"role": "user", "content": content}],
                                   "add_generation_prompt": True,
                                   "chat_template_kwargs": {"enable_thinking": True}}, timeout=600)
    return r["tokens"], r["max_model_len"]


def think_end_id():
    if flag("--think-end-id"):
        return int(arg("--think-end-id", ""))
    toks = harness.post("/tokenize", {"prompt": "</think>", "add_special_tokens": False}, timeout=60)["tokens"]
    if len(toks) != 1:
        invalid(f"`</think>` tokenizes to {toks} on this server, not one token; pass --think-end-id")
    return toks[0]


def ids_from_logprobs(tokens):
    """`return_tokens_as_token_ids` renders each token as "token_id:1234"."""
    out = []
    for t in tokens:
        if not t.startswith("token_id:"):
            return None
        out.append(int(t[9:]))
    return out


def generate(prompt_ids, max_tokens):
    """Stream a completion from raw prompt ids. -> (ids, ttft, decode_seconds, usage)."""
    payload = {"prompt": prompt_ids, "max_tokens": max_tokens,
               "temperature": 0, "logprobs": 0, "return_tokens_as_token_ids": True,
               "stream": True, "stream_options": {"include_usage": True}}
    toks, usage, t_first = [], {}, None
    t0 = time.perf_counter()
    for ev in harness.stream("/v1/completions", payload):
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            lp = ch.get("logprobs") or {}
            if lp.get("tokens"):
                if t_first is None:
                    t_first = time.perf_counter()
                toks.extend(lp["tokens"])
    t_end = time.perf_counter()
    ids = ids_from_logprobs(toks)
    if ids is None:
        raise SystemExit("server did not honour return_tokens_as_token_ids -- "
                         "cannot pin an exact token sequence, refusing to guess")
    return ids, (t_first or t_end) - t0, t_end - (t_first or t_end), usage


if not os.path.exists(CORPUS):
    docs = [open(f).read() for f in sorted(glob.glob(os.path.expanduser("~/qwen-serving/*.md")) +
                                           glob.glob(os.path.expanduser("~/qwen-serving/*/README.md")))]
    text = "\n\n".join(docs)
    if not text:
        invalid(f"no corpus at {CORPUS} and no ~/qwen-serving/*.md to build one from; pass --corpus")
    while len(text) < 200000:
        text += "\n\n" + "\n\n".join(docs)
    os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
    open(CORPUS, "w").write(text)
doc = open(CORPUS).read()[: int(CTX * 3.6)]

DOC_TASKS = [
    ("copy", "Gengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten."),
    ("code", "Gengiv alle bash-kommandoer fra teksten i én samlet liste, ordret."),
    ("edit", "Omskriv afsnittet om Docker så det er kortere, men behold alle kommandoer ordret."),
    ("quote", "Citér ordret de afsnit der handler om hukommelse og KV-cache, og forklar dem."),
    ("summary", "Skriv en grundig, struktureret opsummering på dansk."),
    ("qa", "Svar udførligt: hvilke optimeringer gav mest, og hvorfor?"),
]
FREE_TASKS = [
    ("math", "A train leaves at 09:40 and averages 84 km/h for 2 h 25 min, then 61 km/h for the remaining "
             "137 km. A second train leaves the same station at 10:15 on the same route at a constant speed "
             "and arrives at the same time. What is its speed? Show your reasoning, then give the answer."),
    ("plan", "Plan a migration of a 2 TB PostgreSQL 13 database to PostgreSQL 16 with under five minutes of "
             "downtime. Weigh logical replication against pg_upgrade --link, list the risks, and end with a "
             "numbered runbook."),
]
want = arg("--tasks", "")
TASKS = [(n, q, True) for n, q in DOC_TASKS] + [(n, q, False) for n, q in FREE_TASKS]
TASKS = [t for t in TASKS if not want or t[0] in want.split(",")]
os.makedirs(TARGETS, exist_ok=True)
END = think_end_id()

# Warm-up, as in labd_accept: the first long-block step JIT-compiles Triton kernels. Done before any
# counter is read, so it never enters a measurement.
warm_ids, _ = tokenize_chat("Dokument:\n\n" + doc[:4000] +
                            "\n\nGengiv ordret de første 10 linjer af dokumentet.")
for _ in range(2):
    generate(warm_ids, 64)
# num_speculative_tokens, from the per-position counter's cardinality.
NSPEC = max(len(metrics()[3]), BLOCK + 1)


def counters():
    return {"steps": 0.0, "slots": 0.0, "acc": 0.0, "pos": [0.0] * NSPEC, "tokens": 0, "forced": 0.0,
            "dec": 0.0, "chunks": 0, "clean": 0, "first_bad": None}


def add(into, c):
    for k in ("steps", "slots", "acc", "tokens", "forced", "dec", "chunks", "clean"):
        into[k] += c[k]
    into["pos"] = [a + b for a, b in zip(into["pos"], c["pos"])]


def report(c):
    steps = max(c["steps"], 1.0)
    slots = c["slots"] / steps
    head = sum(c["pos"][:BLOCK]) / steps
    return {"tokens": c["tokens"], "forced": int(c["forced"]), "chunks": c["chunks"], "clean": c["clean"],
            "steps": c["steps"], "tok_per_step": 1 + c["acc"] / steps, "slots_per_step": slots,
            "long_frac": max(0.0, (slots - BLOCK) / max(NSPEC - BLOCK, 1)),
            "head": head, "tail": sum(c["pos"][BLOCK:]) / steps,
            "accept_at_pos": [p / steps for p in c["pos"]],
            "decode_tok_s": c["forced"] / max(c["dec"], 1e-3), "first_bad": c["first_bad"]}


def line(label, phase, r):
    bad = "-" if r["first_bad"] is None else str(r["first_bad"])
    acc = ",".join(f"{a:.2f}" for a in r["accept_at_pos"])
    return (f"LABDTHINK {TAG} {label:8s} {phase:9s} tokens={r['tokens']:5d} forced={r['forced']:5d} "
            f"chunks={r['clean']:3d}/{r['chunks']:<3d} steps={r['steps']:5.0f} tok/step={r['tok_per_step']:5.2f} "
            f"slots/step={r['slots_per_step']:5.2f} long={r['long_frac'] * 100:5.1f}% "
            f"head={r['head']:5.2f} tail={r['tail']:5.2f} decode={r['decode_tok_s']:6.1f} "
            f"acc@pos=[{acc}] firstbad={bad}")


tot = {p: counters() for p in PHASES}
rows = []
for name, q, with_doc in TASKS:
    prompt_ids, max_model_len = tokenize_chat(("Dokument:\n\n" + doc + "\n\n" + q) if with_doc else q)
    sig = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()[:16]
    path = os.path.join(TARGETS, f"labd_think_target_{name}_{sig}_{MAXTOK}.json")

    if flag("--recapture") or not os.path.exists(path):
        ids, _, _, _ = generate(prompt_ids, MAXTOK)
        json.dump({"task": name, "prompt_sig": sig, "captured_by": TAG, "thinking": True,
                   "max_tokens": MAXTOK, "ctx": CTX, "ids": ids}, open(path, "w"))
        print(f"LABDTHINK {TAG} {name:8s} CAPTURED target n={len(ids)} -> {path} "
              f"(this run is scoring itself; freeze and reuse for the comparison)", flush=True)
    target = json.load(open(path))["ids"]
    if len(prompt_ids) + len(target) > max_model_len:
        target = target[: max_model_len - len(prompt_ids) - 1]
    cut = target.index(END) + 1 if END in target else len(target)
    spans = {"reasoning": (0, cut), "answer": (cut, len(target))}

    per = {}
    for phase in PHASES:
        lo, hi = spans[phase]
        c = counters()
        c["tokens"] = hi - lo
        for off in range(lo, hi, CHUNK):
            want_ids = target[off: min(off + CHUNK, hi)]
            d0, s0, a0, p0 = metrics()
            got, _, dec, _ = generate(prompt_ids + target[:off], len(want_ids))
            d1, s1, a1, p1 = metrics()
            clean = got == want_ids
            c["chunks"] += 1
            c["clean"] += 1 if clean else 0
            if not clean and c["first_bad"] is None:
                c["first_bad"] = off + next((i for i, (g, w) in enumerate(zip(got, want_ids)) if g != w),
                                            len(got))
            if clean or flag("--keep-dirty"):
                c["steps"] += d1 - d0
                c["slots"] += s1 - s0
                c["acc"] += a1 - a0
                for pos in set(p0) | set(p1):
                    if pos < NSPEC:
                        c["pos"][pos] += p1.get(pos, 0.0) - p0.get(pos, 0.0)
                c["forced"] += len(got)
                c["dec"] += dec
        add(tot[phase], c)
        per[phase] = report(c)
        if c["tokens"]:
            print(line(name, phase, per[phase]), flush=True)
    frac = cut / max(len(target), 1)
    note = "" if END in target else f" (no </think> within {MAXTOK} tokens: all reasoning)"
    print(f"LABDTHINK {TAG} {name:8s} split     reasoning={cut} answer={len(target) - cut} "
          f"reasoning_frac={frac:.3f}{note}", flush=True)
    rows.append({"task": name, "prompt": len(prompt_ids), "target": len(target), "reasoning_tokens": cut,
                 "answer_tokens": len(target) - cut, "reasoning_frac": frac, "think_end_found": END in target,
                 **per})

totals = {p: report(tot[p]) for p in PHASES}
for p in PHASES:
    if tot[p]["tokens"]:
        print(line("TOTAL", p, totals[p]), flush=True)
all_tokens = sum(tot[p]["tokens"] for p in PHASES)
frac = tot["reasoning"]["tokens"] / max(all_tokens, 1)


def tps(p):
    return f"{totals[p]['tok_per_step']:.3f}" if tot[p]["steps"] else "n/a"


print(f"LABDTHINK {TAG} TOTAL    split     reasoning_frac={frac:.3f} "
      f"tok/step reasoning={tps('reasoning')} answer={tps('answer')}", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
json.dump({"tag": TAG, "ctx": CTX, "chunk": CHUNK, "block": BLOCK, "max_tokens": MAXTOK, "think_end_id": END,
           "nspec": NSPEC, "rows": rows, "totals": totals, "reasoning_frac": frac}, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
if not any(tot[p]["steps"] for p in PHASES):
    invalid("no speculative step counted in either phase: the server runs no speculator, or every chunk was dirty")
