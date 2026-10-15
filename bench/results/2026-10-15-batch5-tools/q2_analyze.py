"""Batch 5 Q2 table: per arm, the launcher's printed pair (the `[start_qwen] note:` line), the pool lines, and per
probe wave the engine's own Running: lines inside the held window, the /metrics peak (num_gpu_blocks from
vllm:cache_config_info, kv_cache_usage_perc x (num_gpu_blocks - 1) = blocks in use), and the preemption delta.
  python q2_analyze.py <dir>:<label> ...
"""
import collections, json, re, sys

for arg in sys.argv[1:]:
    D, L = arg.split(":")
    boot = open(f"{D}/raw-bootlines-{L}.log", encoding="utf-8", errors="replace").read()
    m = re.search(r"holds about (\d+) resident requests on short .*?about (\d+) at MAX_LEN=(\d+), at DFLASH_TOKENS=(\d+) "
                  r"\(one request needs (\d+)\+2\*ceil\(ctx/864\) of the pool's (\d+) pages", boot)
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", boot)
    print(f"## {L} ({D})")
    print(f"printed: {'1k=%s MAX_LEN=%s (MAX_LEN %s, k=%s, base %s, pool %s pages)' % m.groups() if m else 'NO PRINT LINE'}")
    print(f"engine: {kv.group(0) if kv else 'no KV line'}")
    for w in (1, 2, 3):
        try:
            j = json.load(open(f"{D}/raw-probe-{L}-w{w}.json"))
        except Exception as e:  # noqa: BLE001
            print(f"  w{w}: no probe ({e})"); continue
        lines = [l for l in open(f"{D}/raw-running-{L}-w{w}.log") if l.startswith("HELD")]
        c = collections.Counter(re.search(r"Running: \d+ reqs, Waiting: \d+ reqs, GPU KV cache usage: [\d.]+%", l).group(0)
                                for l in lines if "Running:" in l)
        runs = [int(re.search(r"Running: (\d+)", k).group(1)) for k in c]
        s = open(f"{D}/raw-metrics-{L}-w{w}.log").read()
        nb = int(re.search(r'num_gpu_blocks="(\d+)"', s).group(1))
        mc = re.search(r'kv_cache_max_concurrency="([\d.]+)"', s)
        u = float(re.search(r"^vllm:kv_cache_usage_perc\{[^}]*\} (\S+)", s, re.M).group(1))
        r = float(re.search(r"^vllm:num_requests_running\{[^}]*\} (\S+)", s, re.M).group(1))
        g = [x[3] for x in j.get("gauges_running_waiting_preemptions_kvperc", []) if len(x) > 3 and x[3] is not None]
        pre = (max(g) - min(g)) if g else None
        errs = sum("error" in q for q in j.get("requests", []))
        ptok = [q.get("prompt_tokens") for q in j.get("requests", []) if q.get("prompt_tokens")]
        print(f"  w{w} n={len(j.get('requests', []))} prompt_tokens~{min(ptok) if ptok else '?'}-{max(ptok) if ptok else '?'}"
              f" HELD max Running={max(runs) if runs else None} lines={dict(c.most_common(3))}")
        print(f"     /metrics peak: num_gpu_blocks={nb} kv_cache_max_concurrency={mc.group(1) if mc else None} usage={u} "
              f"blocks={u*(nb-1):.4f} running={r:.0f} blocks/req={u*(nb-1)/max(r,1):.2f} preemptions={pre} errors={errs}")
