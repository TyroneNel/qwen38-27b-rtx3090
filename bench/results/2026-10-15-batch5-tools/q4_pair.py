"""Batch 5 Q4: score the two IFBench arms on the same prompts (batch 2's ifbench_pair.py, generalised to two arm
directories; the method is unchanged). Reads each arm's generation record (raw-ifbench-<label>.json: finish_reason
and content per prompt) and the pinned IFBench run_eval output (raw-ifbench-eval-<label>/eval_results_{strict,loose}
.jsonl), and reports per arm and mode: raw (official, all prompts), answered (non-empty content), shared (non-empty
in BOTH arms), plus the per-prompt disagreements on the shared set and a two-sided exact sign test on them.

  python q4_pair.py <dirA>:<labelA> <dirB>:<labelB>
"""
import glob, json, math, os, sys

arms = [a.split(":") for a in sys.argv[1:3]]


def load(D, L):
    gen = {r["prompt"]: r for r in json.load(open(os.path.join(D, f"raw-ifbench-{L}.json")))}
    ev = {}
    for mode in ("strict", "loose"):
        f = glob.glob(os.path.join(D, f"raw-ifbench-eval-{L}", f"*eval_results_{mode}.jsonl"))[0]
        ev[mode] = {r["prompt"]: r for r in map(json.loads, open(f))}
    return gen, ev


def sign_p(k, n):  # two-sided exact binomial(n, 1/2) p-value for k successes
    if n == 0:
        return 1.0
    c = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(c + 1)) / 2 ** n)


(DA, LA), (DB, LB) = arms
(gA, eA), (gB, eB) = load(DA, LA), load(DB, LB)
empty = {L: {p for p, r in g.items() if not r["response"].strip()} for L, g in ((LA, gA), (LB, gB))}
trunc = {L: {p for p, r in g.items() if r["finish"] == "length"} for L, g in ((LA, gA), (LB, gB))}
prompts = list(eA["strict"])
assert set(prompts) == set(eB["strict"]), "the two arms scored different prompt sets"
shared = [p for p in prompts if p not in empty[LA] and p not in empty[LB]]
for L, g in ((LA, gA), (LB, gB)):
    toks = [r["completion_tokens"] for r in g.values()]
    print(f"{L}: n={len(g)} truncated (finish=length)={len(trunc[L])} empty content={len(empty[L])} "
          f"mean completion tokens={sum(toks)/len(toks):.0f}")
print(f"prompts={len(prompts)} empty in both={len(empty[LA] & empty[LB])} shared (non-empty in both)={len(shared)}")
for mode in ("strict", "loose"):
    for L, ev in ((LA, eA), (LB, eB)):
        r = ev[mode]
        acc = lambda ps: (sum(r[p]["follow_all_instructions"] for p in ps), len(ps))  # noqa: E731
        raw, ans, sh = acc(prompts), acc([p for p in prompts if p not in empty[L]]), acc(shared)
        ia = sum(sum(r[p]["follow_instruction_list"]) for p in shared)
        ib = sum(len(r[p]["follow_instruction_list"]) for p in shared)
        print(f"IFBPAIR {mode:6s} {L:4s} prompt-level raw={raw[0]}/{raw[1]}={raw[0]/raw[1]:.4f} "
              f"answered={ans[0]}/{ans[1]}={ans[0]/max(1, ans[1]):.4f} shared={sh[0]}/{sh[1]}={sh[0]/max(1, sh[1]):.4f} "
              f"| instruction-level shared={ia}/{ib}={ia/max(1, ib):.4f}")
    a, b = eA[mode], eB[mode]
    ao = sum(1 for p in shared if a[p]["follow_all_instructions"] and not b[p]["follow_all_instructions"])
    bo = sum(1 for p in shared if b[p]["follow_all_instructions"] and not a[p]["follow_all_instructions"])
    print(f"IFBPAIR {mode:6s} shared-set disagreements: {LA}-only pass={ao} {LB}-only pass={bo} "
          f"sign-test p={sign_p(ao, ao + bo):.3f}")
    ra = sum(1 for p in prompts if a[p]["follow_all_instructions"] and not b[p]["follow_all_instructions"])
    rb = sum(1 for p in prompts if b[p]["follow_all_instructions"] and not a[p]["follow_all_instructions"])
    print(f"IFBPAIR {mode:6s} all-prompt disagreements: {LA}-only pass={ra} {LB}-only pass={rb} "
          f"sign-test p={sign_p(ra, ra + rb):.3f}")
