"""Builds the evals-9 acceptance panel: 50 prompts from each of DeepSpec's nine eval sets.

DeepSpec's eval.py (deepseek-ai/DeepSpec, eval.py:11-27 at the pinned commit) runs nine
sets, each with a sample cap: gsm8k 500, math500 500, aime25 30, humaneval 164, mbpp 256,
livecodebench 500, mt-bench 80, alpaca 500, arena-hard-v2 500. It reads them from the
`turns` JSONL files checked in under `eval_datasets/`, which
`eval_datasets/convert_eval_datasets_to_jsonl.py` and `download_arena_hard_dataset.py`
derived from the Hugging Face sources in SOURCES below (with their prompt formats: the
"\\boxed{}" suffix on gsm8k/math500/aime25, the HumanEval and LiveCodeBench wrappers).
This script downloads those files at the pinned commit, so the prompts are byte for byte
the ones DeepSpec's evaluator reads, and repeats its selection
(`base_evaluator.run_dataset`): keep `turns[:1]`, and when a set is longer than its cap,
`random.Random(980406).shuffle` it and keep the first `cap` rows. The panel is the first 50
of that selection, so every panel item is also in DeepSpec's own sample. aime25 has 30 rows,
so its domain has 30 items.

The licences of the nine sets are unchecked: keep the output outside the repo.

Usage (stdlib only, any Python 3.8+):

  python bench/accept_panel_data.py [--out PATH] [--per-domain 50] [--seed 980406]
                                    [--commit SHA]

Writes PATH (JSONL: domain, rank, row, source, prompt) and PATH.meta.json (commit, caps,
seed, per-domain counts and sha256 of each downloaded file). Exits 0, or 2 if a download
fails or a set comes back empty.
"""
import argparse
import io
import hashlib
import json
import os
import random
import sys
import urllib.request

COMMIT = "005e03b81cec38b7da6399833d609ee89a2587f2"  # DeepSpec main, 2026-07-09
RAW = "https://raw.githubusercontent.com/deepseek-ai/DeepSpec/{commit}/eval_datasets/{name}.jsonl"
# (domain, DeepSpec's sample cap from eval.py TASKS, HF source the checked-in file came from)
SOURCES = [
    ("gsm8k", 500, "openai/gsm8k main/test"),
    ("math500", 500, "HuggingFaceH4/MATH-500 test"),
    ("aime25", 30, "MathArena/aime_2025 train"),
    ("humaneval", 164, "openai/openai_humaneval test"),
    ("mbpp", 256, "google-research-datasets/mbpp sanitized/test"),
    ("livecodebench", 500, "livecodebench/code_generation_lite test.jsonl..test6.jsonl"),
    ("mt-bench", 80, "HuggingFaceH4/mt_bench_prompts train (first turn only)"),
    ("alpaca", 500, "tatsu-lab/alpaca train"),
    ("arena-hard-v2", 500, "lmarena-ai/arena-hard-auto data/arena-hard-v2.0/question.jsonl"),
]
DEFAULT_OUT = ("G:/dev/hq-results/deepinfer-20261020/evals-9/data/panel.jsonl" if os.name == "nt"
               else "/mnt/g/dev/hq-results/deepinfer-20261020/evals-9/data/panel.jsonl")


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--per-domain", type=int, default=50)
    ap.add_argument("--seed", type=int, default=980406, help="DeepSpec eval.py's --seed default")
    ap.add_argument("--commit", default=COMMIT)
    a = ap.parse_args()

    meta = {"deepspec_commit": a.commit, "seed": a.seed, "per_domain": a.per_domain,
            "selection": "turns[:1]; if len > cap: random.Random(seed).shuffle, keep [:cap]; "
                         "then keep [:per_domain]", "domains": {}}
    out_rows = []
    for name, cap, source in SOURCES:
        url = RAW.format(commit=a.commit, name=name)
        try:
            body = fetch(url)
        except OSError as e:
            sys.stderr.write(f"accept_panel_data: {name}: {url}: {e}\n")
            return 2
        rows = []
        # io.StringIO(newline=None) iterates like DeepSpec's open(): universal newlines, but
        # no split on U+2028 and friends, which str.splitlines would do inside a prompt.
        for i, line in enumerate(io.StringIO(body.decode("utf-8"), newline=None)):
            if line.strip():
                turns = json.loads(line)["turns"]
                assert isinstance(turns, list) and turns and isinstance(turns[0], str), (name, i)
                rows.append((i, turns[0]))
        if not rows:
            sys.stderr.write(f"accept_panel_data: {name}: empty file\n")
            return 2
        n_file = len(rows)
        if n_file > cap:
            random.Random(a.seed).shuffle(rows)
            rows = rows[:cap]
        picked = rows[: a.per_domain]
        for rank, (row, prompt) in enumerate(picked):
            out_rows.append({"domain": name, "rank": rank, "row": row, "source": source,
                             "prompt": prompt})
        meta["domains"][name] = {"source": source, "deepspec_cap": cap, "file_rows": n_file,
                                 "items": len(picked), "url": url,
                                 "sha256": hashlib.sha256(body).hexdigest()}
        print(f"{name:14s} items={len(picked):3d} file_rows={n_file:5d} cap={cap:3d} source={source}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.out + ".meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {len(out_rows)} items -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
