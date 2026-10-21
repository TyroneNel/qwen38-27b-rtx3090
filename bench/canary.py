"""Golden greedy canary: 8 fixed prompts x 256 greedy tokens at C1, compared with a stored golden.

evals-7 in docs/research/deepinfer/evals.md. Each prompt goes through /tokenize as one user
message with thinking off (the ids /v1/chat/completions would send), then a greedy /v1/completions
from those ids with `logprobs: 0`, `return_tokens_as_token_ids: true` and ignore_eos, one request
at a time. The token ids and each token's logprob (the top-1, since decoding is greedy) are kept.

The golden file (--golden) holds one entry per key "<GPU name>|<vLLM version>|<mode>": the GPU
from `nvidia-smi` (or --gpu), the version from the server's /version (or --vllm), the mode from
--mode. Only --refresh writes it, and only the entry for this key; it never compares.

Without --refresh each prompt is compared with the golden entry: MATCH (ids and logprobs equal,
logprobs within --lp-tol, default exact), LOGPROB-DRIFT (ids equal, a logprob moved more than
--lp-tol), MISMATCH at the first position whose id differs, or PROMPT-CHANGED (the chat template
or tokenizer now builds different prompt ids). Anything but MATCH means "inspect", not "broken":
Triton, torch and driver versions can move bits.

  docker exec <c> /app/venv/bin/python /app/bench/canary.py --mode B --golden /app/bench/canary_golden.json --refresh
  docker exec <c> /app/venv/bin/python /app/bench/canary.py --mode B --golden /app/bench/canary_golden.json

Prints one line per prompt, then
  canary: MATCH 8/8 key=NVIDIA GeForce RTX 3090|0.31.0|B
  canary: INSPECT 6/8 key=... first_mismatch=[-, 117, -, ...]
Exit 0 on a match or after --refresh, 1 on anything to inspect, 2 if the run is invalid (no
server, no golden entry for the key, token ids not returned, GPU name unknown).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import harness

PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and tests.",
    "Explain how a hash table handles collisions. Cover chaining and open addressing.",
    "Solve step by step: a train leaves at 09:40 and travels 312 km at 78 km/h. When does it arrive?",
    "Skriv en kort historie på dansk om en fyrpasser, der finder en flaske med et brev.",
    "Translate into German: The meeting was moved to Thursday because the projector was broken.",
    "List the planets of the solar system in order and give one fact about each.",
    "Write a bash script that finds the ten largest files under a directory and prints their sizes.",
    "Describe what happens, step by step, when you type a URL into a browser and press Enter.",
]


def tokenize_chat(content):
    """The prompt ids /v1/chat/completions would build for one user message, thinking off."""
    r = harness.post("/tokenize", {"messages": [{"role": "user", "content": content}],
                                   "add_generation_prompt": True,
                                   "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
    return r["tokens"]


def greedy(prompt_ids, n):
    """(ids, logprobs) of n greedy tokens; ids is None if the server ignores return_tokens_as_token_ids."""
    r = harness.post("/v1/completions", {"prompt": prompt_ids, "max_tokens": n, "temperature": 0,
                                         "logprobs": 0, "return_tokens_as_token_ids": True,
                                         "ignore_eos": True})
    lp = r["choices"][0].get("logprobs") or {}
    toks = lp.get("tokens") or []
    if not toks or not all(isinstance(t, str) and t.startswith("token_id:") for t in toks):
        return None, None
    return [int(t[9:]) for t in toks], lp.get("token_logprobs") or []


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    names = [l.strip() for l in out.splitlines() if l.strip()]
    return names[0] if names else None


def server_version():
    with urllib.request.urlopen(harness.request("/version"), timeout=30) as r:
        return json.load(r)["version"]


def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def compare(gold, ids, lps, prompt_sha, tol):
    """(status, first mismatch position or None, max |dlogp| over the common id prefix)."""
    if gold["prompt_sha"] != prompt_sha:
        return "PROMPT-CHANGED", -1, None
    g_ids, g_lp = gold["ids"], gold["logprobs"]
    first = next((j for j, (a, b) in enumerate(zip(g_ids, ids)) if a != b), None)
    if first is None and len(g_ids) != len(ids):
        first = min(len(g_ids), len(ids))
    upto = len(ids) if first is None else first
    drift = max((abs(a - b) for a, b in zip(g_lp[:upto], lps[:upto])), default=0.0)
    if first is not None:
        return "MISMATCH", first, drift
    return ("MATCH" if drift <= tol else "LOGPROB-DRIFT"), None, drift


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, help="mode label, part of the golden key")
    ap.add_argument("--golden", required=True, help="golden JSON file")
    ap.add_argument("--refresh", action="store_true", help="write this key's golden entry instead of comparing")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--lp-tol", type=float, default=0.0, help="max |dlogp| still a MATCH (default exact)")
    ap.add_argument("--gpu", help="GPU name for the key (default: nvidia-smi)")
    ap.add_argument("--vllm", help="vLLM version for the key (default: the server's /version)")
    ap.add_argument("--out", help="write this run's per-prompt results as JSON here")
    a = ap.parse_args()

    try:
        version = a.vllm or server_version()
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        print(f"canary: INVALID no server at {harness.base_url()}: {e}")
        return 2
    gpu = a.gpu or gpu_name()
    if not gpu:
        print("canary: INVALID GPU name unknown (no nvidia-smi); pass --gpu")
        return 2
    key = f"{gpu}|{version}|{a.mode}"

    golden = {"entries": {}}
    if os.path.exists(a.golden):
        with open(a.golden) as f:
            golden = json.load(f)
    entry = golden["entries"].get(key)
    if not a.refresh and entry is None:
        print(f"canary: INVALID no golden entry for key={key} in {a.golden}; run --refresh first")
        return 2
    if entry is not None and not a.refresh and entry["tokens"] != a.tokens:
        print(f"canary: INVALID golden has {entry['tokens']} tokens per prompt, run asked {a.tokens}")
        return 2

    runs = []
    try:
        for i, text in enumerate(PROMPTS):
            p_ids = tokenize_chat(text)
            ids, lps = greedy(p_ids, a.tokens)
            if ids is None or len(lps) != len(ids):
                print("canary: INVALID server did not return token ids and logprobs")
                return 2
            runs.append({"prompt_sha": sha(p_ids), "ids": ids, "logprobs": lps})
    except urllib.error.HTTPError as e:
        print(f"canary: INVALID HTTP {e.code} from {e.url}: {e.read()[:500]!r}")
        return 2

    if a.refresh:
        golden["entries"][key] = {"tokens": a.tokens, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                  "prompts": runs}
        os.makedirs(os.path.dirname(os.path.abspath(a.golden)), exist_ok=True)
        tmp = a.golden + ".tmp"
        with open(tmp, "w") as f:
            json.dump(golden, f, indent=1, sort_keys=True)
        os.replace(tmp, a.golden)
        print(f"canary: REFRESHED {len(runs)} prompts x {a.tokens} tokens key={key} -> {a.golden}")
        return 0

    results, firsts = [], []
    for i, (gold, run) in enumerate(zip(entry["prompts"], runs)):
        status, first, drift = compare(gold, run["ids"], run["logprobs"], run["prompt_sha"], a.lp_tol)
        results.append({"i": i, "status": status, "first_mismatch": first, "max_dlogp": drift, **run})
        firsts.append("-" if first is None else str(first))
        where = ""
        if status == "MISMATCH":
            g = gold["ids"][first] if first < len(gold["ids"]) else None
            r = run["ids"][first] if first < len(run["ids"]) else None
            where = f" at {first}/{a.tokens} (golden {g} got {r})"
        d = "" if drift is None else f" max_dlogp={drift:.3e}"
        print(f"canary: prompt {i} {status}{where}{d}")
    n_ok = sum(r["status"] == "MATCH" for r in results)
    ok = n_ok == len(results) == len(entry["prompts"])
    print(f"canary: {'MATCH' if ok else 'INSPECT'} {n_ok}/{len(entry['prompts'])} key={key}"
          + ("" if ok else f" first_mismatch=[{', '.join(firsts)}]"))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"key": key, "match": ok, "prompts": results}, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
