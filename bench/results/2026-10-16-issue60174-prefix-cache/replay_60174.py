"""vllm#60174 replay: does DFlash2 + prefix caching collapse after a cache hit on the default config?

Runs inside the container (`cd /app && venv/bin/python /tmp/replay_60174.py <out.json>`), stdlib only, through
bench/harness.py for the key, URL and /metrics, as the other bench tools do. `--cold-rerun <prev.json> <out.json>`
re-sends only the cold-T0 pass from a finished run (see cold_rerun).

Four passes, one request at a time, each conversation with its own document header so no two passes share a prefix:
  warm-T0         5-turn chat, temperature 0; turns 2-5 resend turn 1's prefix and should hit the prefix cache
  cold-T0         warm-T0's exact 5 request bodies, each with a unique cache_salt: no hits (the control)
  warm-T1         5-turn chat with the issue's sampling (temperature 1.0, top_p 0.95, top_k 20, seed 17)
  warm-T0-repeat  warm-T0 again with a fresh header
Per request: prompt/cached tokens, finish reason, reply, latency, the spec-decode counter deltas (acceptance length =
accepted/drafts + 1), the longest run of one repeated token id and of one word, and the distinct-word ratio.
"""
import json
import random
import re
import sys
import time
import urllib.error
import uuid

sys.path.insert(0, "/app/bench")
import harness  # noqa: E402

DOC_TOKENS = 20000
MAX_TOKENS = 256
T1 = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 17}
T0 = {"temperature": 0.0}

# ---------------------------------------------------------------- the document: a synthetic storage-system spec
COMPONENTS = ["ingest gateway", "write-ahead log", "compaction scheduler", "replica manager", "placement service",
              "metadata journal", "lease coordinator", "block allocator", "checksum auditor", "snapshot exporter",
              "tiering daemon", "quota enforcer", "read-repair worker", "membership gossip layer", "index builder",
              "tombstone collector", "rebalance planner", "erasure-coding encoder", "admission controller",
              "telemetry relay", "backup verifier", "hot-key detector", "stripe reconstructor", "lock table"]
VERBS = ["flushes", "rotates", "validates", "throttles", "re-shards", "persists", "evicts", "batches", "reconciles",
         "retries", "fences", "prefetches", "compresses", "replays", "rebuilds", "samples", "pins", "drains",
         "publishes", "acknowledges"]
OBJECTS = ["dirty segments", "pending leases", "stale replicas", "partial stripes", "journal entries",
           "cold extents", "hint records", "range locks", "manifest files", "bloom filters", "repair tickets",
           "heartbeat digests", "quota ledgers", "checkpoint markers", "tenant budgets", "shard maps",
           "commit vectors", "fragment headers", "retention tags", "delta indexes"]
CONDS = ["the backlog exceeds its high-water mark", "a node misses three consecutive heartbeats",
         "the disk queue depth stays above the soft limit", "a tenant crosses its burst allowance",
         "the leader epoch advances", "a checksum mismatch is reported", "free capacity drops below the reserve",
         "the rack-awareness policy changes", "a rolling upgrade begins", "clock skew exceeds the tolerance",
         "the replication lag grows past the alert threshold", "an operator issues a drain request",
         "the compaction debt doubles within one window", "a snapshot export is requested",
         "read amplification rises above the configured ratio", "the encoder reports a degraded stripe"]
REASONS = ["to keep tail latency bounded", "so that recovery never reads a torn page",
           "because the journal must stay strictly ordered", "to avoid a thundering herd on restart",
           "so that quota accounting remains monotonic", "to protect the foreground write path",
           "because replicas may otherwise diverge silently", "to cap the memory held by in-flight buffers",
           "so that a partitioned minority cannot accept writes", "to make the rebuild cost proportional to damage",
           "because operators audit this value during incidents", "to keep cross-zone traffic predictable"]
UNITS = ["ms", "seconds", "MiB", "requests per second", "percent", "entries", "minutes", "GiB", "extents", "epochs"]
SECTION_TOPICS = ["Durability guarantees", "Write path", "Read path", "Failure detection", "Recovery procedure",
                  "Capacity planning", "Multi-tenant isolation", "Upgrade protocol", "Observability",
                  "Background maintenance", "Consistency model", "Security boundaries", "Data placement",
                  "Operational runbook", "Performance tuning", "Disaster recovery"]


def make_document(target_tokens):
    """Deterministic, never repeats a sentence. Returns (text, facts) where facts are (section, component) pairs the
    questions ask about."""
    rng = random.Random(60174)
    seen, sections, facts = set(), [], []

    def sentence():
        while True:
            kind = rng.randrange(4)
            c, v, o = rng.choice(COMPONENTS), rng.choice(VERBS), rng.choice(OBJECTS)
            n, u = rng.choice([2, 3, 4, 5, 8, 10, 12, 16, 20, 24, 30, 32, 45, 50, 64, 90, 120, 128, 250, 256, 500,
                               512, 750, 1000, 1500, 2048, 4096]), rng.choice(UNITS)
            if kind == 0:
                s = f"The {c} {v} {o} every {n} {u} while {rng.choice(CONDS)}."
            elif kind == 1:
                s = f"When {rng.choice(CONDS)}, the {c} {v} {o} {rng.choice(REASONS)}."
            elif kind == 2:
                s = (f"Operators can raise the {c.replace(' ', '_')}.{o.split()[-1]}_limit setting to {n} {u}, "
                     f"but the {rng.choice(COMPONENTS)} then {rng.choice(VERBS)} {rng.choice(OBJECTS)} less often.")
            else:
                s = (f"If the {c} cannot reach the {rng.choice(COMPONENTS)} within {n} {u}, it {v} {o} locally "
                     f"and reports the condition {rng.choice(REASONS)}.")
            if s not in seen:
                seen.add(s)
                return s

    total_words, i = 0, 0
    while total_words * 1.3 < target_tokens:   # ~1.3 tokens per word for this English; the caller measures exactly
        i += 1
        topic = SECTION_TOPICS[(i - 1) % len(SECTION_TOPICS)]
        comp = rng.choice(COMPONENTS)
        paras = []
        for _ in range(rng.randrange(2, 4)):
            paras.append(" ".join(sentence() for _ in range(rng.randrange(4, 8))))
        table = "\n".join(f"| {comp.replace(' ', '_')}.{k} | {rng.randrange(1, 4097)} | {rng.choice(UNITS)} |"
                          for k in rng.sample(["timeout", "batch", "window", "budget", "depth", "fanout", "ttl",
                                               "interval"], 3))
        body = (f"## Section {i}: {topic} ({comp})\n\n" + "\n\n".join(paras)
                + f"\n\n| parameter | default | unit |\n|---|---|---|\n{table}\n")
        sections.append(body)
        facts.append((i, topic, comp))
        total_words += len(body.split())
    return "\n".join(sections), facts


QUESTIONS = [
    "Question: According to Section {a}, what does the {ca} do, and under which conditions? Answer in one paragraph.",
    "Question: Compare what Section {a} and Section {b} say about the {cb}. Cite the parameter table defaults.",
    "Question: List the configuration parameters and defaults given in Section {c}, and explain one trade-off the "
    "document mentions there.",
    "Question: Based on Section {d} ({td}), describe the failure handling the document specifies, step by step.",
    "Question: Summarise the document's overall design philosophy in a few sentences, referring to Sections {a} "
    "and {d}.",
]


def questions(facts):
    rng = random.Random(17)
    a, b, c, d = (facts[k] for k in sorted(rng.sample(range(len(facts)), 4)))
    kw = {"a": a[0], "ca": a[2], "b": b[0], "cb": b[2], "c": c[0], "d": d[0], "td": d[1]}
    return [q.format(**kw) for q in QUESTIONS]


def system_prompt(header, doc):
    return (f"{header}\n\nYou are a careful technical assistant. Answer questions about the storage-system "
            f"specification below. Quote section numbers when you refer to it.\n\n# Specification\n\n{doc}")


# ---------------------------------------------------------------- metrics
def tokenize(text):
    r = harness.post("/tokenize", {"prompt": text, "add_special_tokens": False}, timeout=60)
    return r["tokens"]


def longest_run(seq):
    best, cur, prev = 0, 0, object()
    item = None
    for x in seq:
        cur = cur + 1 if x == prev else 1
        prev = x
        if cur > best:
            best, item = cur, x
    return best, item


def collapse(text):
    words = re.findall(r"[A-Za-z0-9_.'-]+", text.lower())
    toks = tokenize(text) if text else []
    wrun, wword = longest_run(words)
    trun, tid = longest_run(toks)
    return {
        "n_words": len(words),
        "distinct_word_ratio": round(len(set(words)) / len(words), 3) if words else 0.0,
        "longest_word_run": wrun, "longest_word_run_word": wword,
        "n_reply_tokens": len(toks),
        "longest_token_run": trun, "longest_token_run_id": tid,
    }


def send(messages, sampling, salt=None):
    body = {"messages": list(messages), "max_tokens": MAX_TOKENS, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}, **sampling}
    if salt:
        body["cache_salt"] = salt
    d0, a0 = harness.spec()
    t0 = time.time()
    try:
        r = harness.post("/v1/chat/completions", body, timeout=1200)
    except urllib.error.HTTPError as e:
        sys.exit(f"REQUEST REFUSED: HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}")
    lat = time.time() - t0
    d1, a1 = harness.spec()
    dd, da = d1 - d0, a1 - a0
    u = r.get("usage") or {}
    ch = r["choices"][0]
    reply = ch["message"].get("content") or ""
    rec = {
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "finish_reason": ch.get("finish_reason"),
        "latency_s": round(lat, 3),
        "drafts": dd, "accepted": da,
        "acceptance_length": round(da / dd + 1, 3) if dd else None,
        "reply": reply,
    }
    rec.update(collapse(reply))
    return rec, body


def run_chat(name, header, doc, qs, sampling, log):
    msgs = [{"role": "system", "content": system_prompt(header, doc)}]
    out, bodies = [], []
    for i, q in enumerate(qs, 1):
        msgs.append({"role": "user", "content": q})
        rec, body = send(msgs, sampling)
        rec.update(pass_=name, turn=i, header=header)
        out.append(rec)
        bodies.append(body)
        log(rec)
        msgs.append({"role": "assistant", "content": rec["reply"]})
    return out, bodies


def log(rec):
    print(f"{rec['pass_']:15s} turn {rec['turn']}  prompt={rec['prompt_tokens']} cached={rec['cached_tokens']} "
          f"acc_len={rec['acceptance_length']} (acc {rec['accepted']:.0f}/drafts {rec['drafts']:.0f}) "
          f"tokrun={rec['longest_token_run']} wordrun={rec['longest_word_run']} "
          f"distinct={rec['distinct_word_ratio']} finish={rec['finish_reason']} lat={rec['latency_s']}s",
          flush=True)
    print("    reply: " + rec["reply"].replace("\n", " \\n ")[:600], flush=True)


def build_doc():
    """Size the document to ~DOC_TOKENS by measurement; deterministic for a given tokenizer."""
    target = DOC_TOKENS
    for _ in range(4):
        doc, facts = make_document(target)
        n = len(tokenize(doc))
        if abs(n - DOC_TOKENS) < 600:
            break
        target = int(target * DOC_TOKENS / n)
    print(f"document: {len(facts)} sections, {n} tokens (generator target {target})", flush=True)
    return doc, facts, n


def cold_pass(bodies, header, tag):
    cold = []
    for i, body in enumerate(bodies, 1):
        salt = f"cold-{tag}-{i}-{uuid.uuid4().hex}"
        rec, _ = send(body["messages"], T0, salt=salt)
        rec.update(pass_="cold-T0", turn=i, header=header, cache_salt=salt)
        cold.append(rec)
        log(rec)
    return cold


def finish(out_path, started, passes, extra):
    # warm-T0 vs its cold-T0 twin (identical request bodies): first differing character
    div = []
    for w, c in zip(passes["warm-T0"], passes["cold-T0"]):
        a, b = w["reply"], c["reply"]
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
        if k is None and len(a) != len(b):
            k = min(len(a), len(b))
        w["first_diff_vs_cold"] = k   # None = identical
        div.append(k)
    print("warm-T0 vs cold-T0 first differing char per turn (None = identical):", div, flush=True)

    def agg(recs):
        d = sum(r["drafts"] for r in recs)
        a = sum(r["accepted"] for r in recs)
        return round(a / d + 1, 3) if d else None
    summary = {p: {"acceptance_length": agg(r), "acceptance_length_turns2_5": agg(r[1:])} for p, r in passes.items()
               if p in ("warm-T0", "cold-T0", "warm-T1", "warm-T0-repeat")}
    print("pass acceptance length (all turns / turns 2-5):", json.dumps(summary), flush=True)
    with open(out_path, "w") as f:
        json.dump({"started": started, "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra,
                   "max_tokens": MAX_TOKENS, "sampling": {"T0": T0, "T1": T1}, "passes": passes,
                   "first_diff_warmT0_vs_coldT0": div, "summary": summary}, f, indent=1)
    print(f"wrote {out_path}", flush=True)


def cold_rerun(prev_path, out_path):
    """Rebuild warm-T0's exact request bodies from a finished run's JSON (the deterministic document, the header,
    the questions and warm-T0's replies) and send them again with unique cache_salts. The first run's cold pass
    is kept under "cold-T0-invalid-run1"."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev = json.load(open(prev_path))
    doc, facts, n = build_doc()
    if n != prev["doc_tokens"]:
        sys.exit(f"document differs from the first run: {n} vs {prev['doc_tokens']} tokens")
    warm = prev["passes"]["warm-T0"]
    header = warm[0]["header"]
    msgs, bodies = [{"role": "system", "content": system_prompt(header, doc)}], []
    for q, w in zip(prev["questions"], warm):
        msgs.append({"role": "user", "content": q})
        bodies.append({"messages": list(msgs)})
        msgs.append({"role": "assistant", "content": w["reply"]})
    passes = prev["passes"]
    passes["cold-T0-invalid-run1"] = passes.pop("cold-T0")
    passes["cold-T0"] = cold_pass(bodies, header, "rerun-" + uuid.uuid4().hex[:8])
    for w, c in zip(warm, passes["cold-T0"]):
        if w["prompt_tokens"] != c["prompt_tokens"]:
            sys.exit(f"cold turn {c['turn']} prompt {c['prompt_tokens']} != warm {w['prompt_tokens']}")
    print("cold-T0 prompt_tokens equal warm-T0 prompt_tokens on every turn", flush=True)
    finish(out_path, prev["started"], passes,
           {"doc_tokens": n, "doc_sections": len(facts), "questions": prev["questions"],
            "cold_rerun_started": started})


def main():
    if sys.argv[1] == "--cold-rerun":
        return cold_rerun(sys.argv[2], sys.argv[3])
    out_path = sys.argv[1]
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc, facts, n = build_doc()
    qs = questions(facts)

    tag = uuid.uuid4().hex[:8]
    hdr = lambda p: f"Document ID: DOC-60174-{p}-{tag}-{uuid.uuid4().hex}"
    passes = {}

    passes["warm-T0"], bodies = run_chat("warm-T0", hdr("warmT0"), doc, qs, T0, log)

    passes["cold-T0"] = cold_pass(bodies, passes["warm-T0"][0]["header"], tag)

    passes["warm-T1"], _ = run_chat("warm-T1", hdr("warmT1"), doc, qs, T1, log)
    passes["warm-T0-repeat"], _ = run_chat("warm-T0-repeat", hdr("warmT0r"), doc, qs, T0, log)

    finish(out_path, started, passes, {"doc_tokens": n, "doc_sections": len(facts), "questions": qs})


if __name__ == "__main__":
    main()
