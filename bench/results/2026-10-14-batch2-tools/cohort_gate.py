"""S2c: is the cohort row a measurement? Repeat it inside one boot and report the spread.

The cohort row is bench/run_benchmarks.sh's: `vllm bench serve --dataset-name custom --dataset-path
bench/prompts_real.jsonl --custom-output-len 1024 --num-prompts P --max-concurrency C --temperature 0`, read as
decode = C*1000/meanTPOT and tok/step from the spec-decode counters. Around every repetition this also diffs
vllm:num_preemptions_total and samples vllm:num_requests_running every 250 ms for the peak resident count.
prompts_real.jsonl holds 8 prompts, so C16 needs --num-prompts 16 and vllm bench serve repeats the set
(it says so in its log); that row therefore carries 8 prefix-cache-shared pairs, which the C8 row does not.

  venv/bin/python cohort_gate.py <tag> [--conc 8,16] [--reps 3] [--out-dir /tmp/cohort]
"""
import json, os, re, subprocess, sys, threading, time

sys.path.insert(0, "/app/bench")
import harness  # noqa: E402


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


TAG = sys.argv[1]
CONCS = [int(x) for x in arg("--conc", "8,16").split(",")]
REPS = int(arg("--reps", 3))
OUT = arg("--out-dir", "/tmp/cohort")
os.makedirs(OUT, exist_ok=True)
name = harness.model()
root = json.load(__import__("urllib.request").request.urlopen(harness.request("/v1/models")))["data"][0].get("root") or name
B = ["venv/bin/vllm", "bench", "serve", "--host", "127.0.0.1", "--port", os.environ.get("PORT", "18020"),
     "--model", name, "--tokenizer", root, "--served-model-name", name]
SC = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total", "vllm:num_preemptions_total")


def counters():
    m = harness.metrics(*SC)
    return [m.get(k, 0.0) for k in SC]


def num(pat, text):
    m = re.search(pat + r"\s*([\d.]+)", text)
    return float(m.group(1)) if m else float("nan")


print(f"# {TAG} model={name} tokenizer={root} conc={CONCS} reps={REPS}", flush=True)
subprocess.run(B + ["--dataset-name", "random", "--seed", "1001", "--random-input-len", "256", "--random-output-len",
                    "256", "--num-prompts", "16", "--max-concurrency", "8"], cwd="/app", capture_output=True)  # warmup
res = []
for C in CONCS:
    P = max(8, C)
    for rep in range(1, REPS + 1):
        peak, stop = {"run": 0.0}, threading.Event()

        def samp():
            while not stop.is_set():
                try:
                    peak["run"] = max(peak["run"], harness.metrics("vllm:num_requests_running").get("vllm:num_requests_running", 0.0))
                except Exception:
                    pass
                time.sleep(0.25)
        th = threading.Thread(target=samp, daemon=True); th.start()
        a = counters()
        log = os.path.join(OUT, f"cohort_c{C}_rep{rep}.log")
        with open(log, "w") as f:
            subprocess.run(B + ["--dataset-name", "custom", "--dataset-path", "bench/prompts_real.jsonl",
                                "--custom-output-len", "1024", "--num-prompts", str(P), "--max-concurrency", str(C),
                                "--temperature", "0"], cwd="/app", stdout=f, stderr=subprocess.STDOUT)
        b = counters()
        stop.set(); th.join()
        txt = open(log).read()
        tpot = num(r"Mean TPOT \(ms\):", txt)
        row = {"conc": C, "rep": rep, "prompts": P, "decode": C * 1000 / tpot, "mean_tpot_ms": tpot,
               "e2e": num(r"Output token throughput \(tok/s\):", txt), "ttft_ms": num(r"Mean TTFT \(ms\):", txt),
               "tok_per_step": 1 + (b[1] - a[1]) / max(1, b[0] - a[0]), "preempt": b[2] - a[2], "peak_running": peak["run"]}
        res.append(row)
        print(f"COHORT {TAG} C{C} rep={rep} prompts={P} decode(C/meanTPOT)={row['decode']:.1f} e2e={row['e2e']:.1f} "
              f"tok/step={row['tok_per_step']:.2f} meanTTFT={row['ttft_ms']:.0f}ms preempt={row['preempt']:.0f} "
              f"peak_running={row['peak_running']:.0f}", flush=True)
    d = [r["decode"] for r in res if r["conc"] == C]
    print(f"SPREAD {TAG} C{C} decode min={min(d):.1f} max={max(d):.1f} spread={(max(d)-min(d))/min(d)*100:.1f}% "
          f"preempt_total={sum(r['preempt'] for r in res if r['conc'] == C):.0f}", flush=True)
json.dump(res, open(os.path.join(OUT, f"cohort_{TAG}.json"), "w"), indent=1)
