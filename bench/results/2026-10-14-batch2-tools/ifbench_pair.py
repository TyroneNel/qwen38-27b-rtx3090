"""S3b: score the two IFBench efforts on the same prompts.

run_eval.py scores every prompt, so a response cut at max_tokens while still thinking comes back with empty content
and counts as a fail. xhigh thinks far longer than low, so its raw score carries the token cap as well as
instruction following. This reads the official per-prompt eval output of each effort (eval_results_{strict,loose}
.jsonl) and the generation record (finish_reason and content per prompt), and reports, per effort and per mode:
  raw          the official score over all prompts
  answered     over prompts where that effort returned non-empty content
  shared       over prompts where BOTH efforts returned non-empty content (the like-for-like set)
plus the per-prompt disagreements on the shared set.

  python ifbench_pair.py <S3b dir> <xhigh label> <low label>
"""
import glob, json, os, sys

D, LX, LL = sys.argv[1], sys.argv[2], sys.argv[3]


def load(e, L):
    gen = {r["prompt"]: r for r in json.load(open(os.path.join(D, f"raw-ifbench-{e}-{L}.json")))}
    ev = {}
    for mode in ("strict", "loose"):
        f = glob.glob(os.path.join(D, f"raw-ifbench-eval-{e}-{L}", f"*eval_results_{mode}.jsonl"))[0]
        ev[mode] = {r["prompt"]: r for r in map(json.loads, open(f))}
    return gen, ev


gx, ex = load("xhigh", LX)
gl, el = load("low", LL)
empty = {e: {p for p, r in g.items() if not r["response"].strip()} for e, g in (("xhigh", gx), ("low", gl))}
trunc = {e: {p for p, r in g.items() if r["finish"] == "length"} for e, g in (("xhigh", gx), ("low", gl))}
prompts = list(ex["strict"])
shared = [p for p in prompts if p not in empty["xhigh"] and p not in empty["low"]]
for e in ("xhigh", "low"):
    print(f"{e}: truncated (finish=length)={len(trunc[e])} empty content={len(empty[e])} "
          f"(empty and finish=stop: {len(empty[e] - trunc[e])})")
print(f"prompts={len(prompts)} empty in both={len(empty['xhigh'] & empty['low'])} "
      f"shared (non-empty in both)={len(shared)}")
for mode in ("strict", "loose"):
    for e, ev in (("xhigh", ex), ("low", el)):
        r = ev[mode]
        acc = lambda ps: (sum(r[p]["follow_all_instructions"] for p in ps), len(ps))  # noqa: E731
        raw, ans, sh = acc(prompts), acc([p for p in prompts if p not in empty[e]]), acc(shared)
        ia = sum(sum(r[p]["follow_instruction_list"]) for p in shared)
        ib = sum(len(r[p]["follow_instruction_list"]) for p in shared)
        print(f"IFBPAIR {mode:6s} {e:5s} prompt-level raw={raw[0]}/{raw[1]}={raw[0]/raw[1]:.4f} "
              f"answered={ans[0]}/{ans[1]}={ans[0]/max(1, ans[1]):.4f} shared={sh[0]}/{sh[1]}={sh[0]/max(1, sh[1]):.4f} "
              f"| instruction-level shared={ia}/{ib}={ia/max(1, ib):.4f}")
    a, b = ex[mode], el[mode]
    xo = sum(1 for p in shared if a[p]["follow_all_instructions"] and not b[p]["follow_all_instructions"])
    lo = sum(1 for p in shared if b[p]["follow_all_instructions"] and not a[p]["follow_all_instructions"])
    print(f"IFBPAIR {mode:6s} shared-set disagreements: xhigh-only pass={xo} low-only pass={lo}")
