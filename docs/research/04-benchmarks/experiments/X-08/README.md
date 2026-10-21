# X-08 / H08: near-tie classifier for the greedy T=0 equality gate (EAGLE-4 E-18)

Status: **CPU part PASS** (14/14 tests, `test.out`). GPU part not run. Label: logic and plumbing are **measured on CPU against synthetic streams and a mock server** (`test.out`); every threshold is **estimated** (reasoning below); nothing here is a measurement of the real stack.

## Hypothesis

At the first divergence between two greedy runs (SPEC=off vs dflash2 vs mtp), the reference run's top-2 logprobs tell a numerical flip from a wrong token. A flip has a tiny gap on both sides, because both runs saw the same prefix. A real defect (KVarN adaptive + prefix) shows a large gap. The gate passes when every divergence is identical, a near-tie, or confirmed by a replay. It fails closed on missing evidence. Source: `02-eagle4/transferability.md` H08 / X-08 / E-18; the EAGLE-4 reference is `~/research-src/EAGLE-4/tools/fp32_arbiter.py` and `docs/support-matrix.md:159-163` (write-our-own: not copied).

## What `greedy_equal.py` already captures

Read `04-benchmarks/harness/greedy_equal.py`. It **does** request logprobs: `logprobs=True, top_logprobs=2, return_tokens_as_token_ids` (`:36`) and stores `ids` and `margins` (top1 - top2, 4 dp, `:43`). Its `compare` already flags near-tie vs REAL, but:

| Gap | Effect on X-08 |
|---|---|
| Only the margin is kept, not the top-2 token ids | cannot check that the other run's token is the runner-up; a margin-only "tie" can hide a third token |
| Default `--near-tie 0.1` (README says 0.1), the brief asks 1e-2 | the two disagree; see thresholds |
| Single pass per prompt | a prefix-cache hit never happens, so the KVarN case is unreachable |
| Skips `long_context` (rows have `question`, not `prompt`, `:29`) | the `copy` self-hit task is never recorded; short prompts only |
| No `messages` stored | nothing to replay a prefix from |
| No missing-logprob state: `None` margin counts as not a tie (`:68`) | fine, but silent; no `unknown` verdict |
| No fp32/replay step | n/a |

The harness file is untouched. The additions are separate scripts in this directory (the "minimal addition" asked for): `record_top2.py` writes a **superset** of `greedy_<label>.json` (same keys plus `top`, `messages`), so `greedy_equal.py compare` still reads the output. [review: confirmed for default flags; with `--passes 2` (`@p2` keys) or `--doc-ctx` the prompt keys differ, so do not mix such files with plain `greedy_equal.py` output in one compare.]

## Files

| File | Role | Where it runs |
|---|---|---|
| `classify_divergence.py` | `classify REF CAND... ` and `judge` (library + CLI) | CPU |
| `record_top2.py` | recorder: top-n ids and logprobs, `--passes 2`, `--doc-ctx` | GPU box (needs a server) |
| `rescore_replay.py` | replay interface for the GPU agent | GPU box |
| `mock_server_x08.py` | copy of the harness mock plus runner-up ids, `/tokenize`, `/v1/completions` | CPU test only |
| `test_classify_divergence.py`, `test.out` | tests and raw output | CPU |

## Design and thresholds

Per prompt, at the first differing index k (or the end of the shorter run), `gap` = how far the other run's token sits below each run's own top-1 (exact when that token is in the saved top-n; a lower bound when it is not; margin-only for legacy files). The larger side is used. Verdicts:

| Verdict | Rule | Gate |
|---|---|---|
| identical | same ids, same length | pass |
| near_tie | gap < `--near-tie` (default **0.01**) on **both** sides, other token seen in the top-n, no length divergence | pass |
| ulp_band | gap <= `--ulp-band` (default **0.13** [review (phase 5): code default now 0.25 per D18; `classify_divergence.py:4,46,176`, 14 tests pass]), or evidence one-sided / lower bound / length divergence | pending, needs replay |
| real | gap > 0.13 [review: EAGLE-4 does not call these real: gaps > 0.13 go to its fp32 arbiter, which passes d <= 2 ulp (0.25 on [16, 32)) and scales the ulp with magnitude (`EAGLE-4/tools/fp32_arbiter.py:8-12,69-70,85`). Set `--ulp-band` default to 0.25 so 0.13–0.25 goes to replay, matching `judge --tol 0.25`; D18 as written is stricter than EAGLE-4] | fail |
| unknown | no logprobs at k in either run | fail |
| replay_tie, replay_argmax | `judge` on an ulp_band entry (below) | pass |

Exit codes: 0 pass, 1 any real, 2 invalid input or any unknown, 3 no real but ulp_band still unjudged.

**Why two thresholds (estimated).** 1e-2 nats is what the task fixed and it is the default. But the EAGLE-4 gate treats a gap up to **0.13** as a tie because "one bf16 ulp for logits in [16, 32)" is 0.125 (`fp32_arbiter.py:7-8`, `greedy_parity.py` docstring). bf16 logits are quantised: two near-equal logits are either exactly tied (gap 0) or at least ~1 ulp apart. A flip that is one ulp (gap 0.125) is therefore an ordinary kernel-shape rounding difference that 1e-2 would call "real". Our logits' magnitude is not measured here (vLLM does not return raw logits); the [16, 32) range is EAGLE-4's statement for its Qwen target, so treat 0.125 as an assumption and re-derive it from the first real run's margin histogram (margins should cluster on multiples of the ulp). Gaps in [1e-2, 0.13] are not called real and not called a pass: they go to the replay. The harness's 0.1 default sits inside this band too.

**Replay (the "fp32 re-score" step, an interface, not run here).** True fp32 is not available on this box: the 27B target in fp32 is about 108 GB (27e9 x 4 B, estimated), and our weights are quantised. What the GPU agent can do is the interface of `rescore_replay.py`: boot **SPEC=off** with the same CTX, KV dtype and `PTH_FA` pin as the reference run, rebuild the exact prefix (chat-template tokens from `POST /tokenize`, then the reference's ids up to k), send it to `/v1/completions` as token ids with `max_tokens=1, logprobs=5`, and read the top-5. `judge` then decides: candidate token is the replay's top-1 -> `replay_argmax`; the two tokens within `--tol` 0.25 nats (2 x 0.125, EAGLE-4's 2-ulp rule) -> `replay_tie`; otherwise `real`; a token outside the top-5 with a provable gap over 0.25 is `real`, otherwise `unknown`. Caveat: this is a prefill-shaped bf16 forward, a third vote, not an oracle. It cannot separate a fragile argmax from a bug when the corrupted forward is also bf16-plausible, which is why the ulp band is kept narrow.

## Test results (`test.out`)

14 tests pass under pytest and as plain asserts. Synthetic streams: identical runs; benign near-tie flip (gap 0.004 both sides); real divergence (gap 2.0); divergence at token 0 (tie and real, `token0` note); at the last token (tie and real, `last_token` note); missing logprobs (both runs: unknown; one run: never near_tie, capped at ulp_band, large gap still real; no `top` at all); margin-only legacy files; candidate token not the runner-up (ulp_band, not near_tie); thresholds configurable; length divergence (EOS never a plain tie); ulp-band judge cases (tie, argmax, real, outside-top-n real, unprovable unknown, no replay); invalid input (exit 2); a synthetic KVarN-shaped pair with a three-way compare. One end-to-end test drives `record_top2.py --passes 2` -> `classify` -> `rescore_replay.py` -> `judge` against `mock_server_x08.py` for a tie (near_tie, exit 0), identical, a real divergence (exit 1) and a 0.1 band tie (ulp_band exit 3, replay, replay_tie exit 0). A mutation check (near_tie rule loosened) makes the suite fail. Not tested: real vLLM response shapes (see open points).

Pyright cleanup after the coordinator's report: `e['status'].upper()` and `__doc__.split` were flagged. The first is typing-only (every return path of `classify_pair`/`judge_one` sets a str `status`); fixed by `-> dict[str, Any]` annotations. The second is a real, minor edge: under `python -OO` the docstring is stripped and `__doc__` is None, so `--help` setup would raise; fixed with `(__doc__ or "")` in all three scripts (checked with `python -OO ... -h`). The two remaining pyright errors are `import harness` in `record_top2.py` and `rescore_replay.py`, resolved at run time by `PYTHONPATH=/app/bench` as `run_suite.sh:27` [review: `:28`] sets.

## What the GPU agent runs on bench-7

Preconditions: same prompts, `GREEDY_TOKENS`, CTX, KV dtype and `PTH_FA` pin on all arms (harness README "Spec vs non-spec greedy equality"); concurrency 1; `PYTHONPATH=/app/bench` (container; the harness mounts at `/harness`, put this directory next to it or `docker cp` it).

```bash
X=docs/research/04-benchmarks/experiments/X-08     # in the container: wherever it was copied
# per boot (SPEC=off, then dflash2, then mtp), instead of or in addition to run_suite.sh's `greedy` phase:
python3 $X/record_top2.py NS   --max-tokens 256 --passes 2 --out results/x08          # SPEC=off boot
python3 $X/record_top2.py DF2  --max-tokens 256 --passes 2 --out results/x08          # SPEC=dflash2 boot
python3 $X/record_top2.py MTP  --max-tokens 256 --passes 2 --out results/x08          # SPEC=mtp boot
# classify (CPU, anywhere); NS is the reference
python3 $X/classify_divergence.py classify results/x08/greedy_NS.json results/x08/greedy_DF2.json results/x08/greedy_MTP.json \
        --out results/x08/report.json --rescore-out results/x08/requests.json     # exit 0/1/2/3
# only if exit 3 (ulp_band pending): boot SPEC=off again with the NS settings, then
python3 $X/rescore_replay.py results/x08/requests.json --out results/x08/replays.json
python3 $X/classify_divergence.py judge results/x08/report.json results/x08/replays.json --out results/x08/final.json
```

KVarN arm (the known-real case; CTX=huge). Add `--doc-ctx 8000 --corpus /cache/bench/labd_corpus_long.txt` to the recorder so the `copy` task over the document is recorded, twice (`@p2` is the prefix-cache hit). Arms: (a) dflash2 `CTX=huge PREFIX_CACHE=1 DFLASH_TOKENS=15` with `LOOKUP_ADAPTIVE=1` (forced adaptive; the launcher pins 0 otherwise, `start_qwen.sh:422-428`); (b) the same with the pin left on. Reference for (a): arm (b) (same KV format, byte-identical to constant length per `start_qwen.sh:402-411`), because a SPEC=off boot at CTX=huge may not exist or may differ in KV path (confirm; `start_qwen.sh:261-268`). Record the reference pair as `greedy_NS`-style labels and classify (a) against (b).

Pass criteria (X-08): every non-near-tie divergence is reproducible (re-record the same pair once and get the same k and status), and the KVarN adaptive + prefix case is flagged `real`.

## How the KVarN adaptive + prefix case shows up (estimated; the case itself is measured: 38/793 self-hit wrong, clean at 794/794 pinned, `start_qwen.sh:402-411`, `01-project/dflash-specdec.md:470-473`, open ledger item `kvarn-varlen`)

- Pass 1 (cold, `copy`) is IDENTICAL to the reference. The comment says turn 1 is correct.
- Pass 2 (`copy@p2`, cache hit) diverges "after ~38 characters": at about k = 10-15 tokens (38 chars at the corpus's 3.6 chars/token, estimated), then stays wrong. The report line is `copy@p2: REAL at k/...`.
- The reference is confident on a verbatim copy, so its gap to the wrong token should be many nats (estimated; the real margins are the first thing the GPU run measures), far over 0.13: `real`, exit 1. A numerical flip would not be reproducible across pass 1 and pass 2 of the same prompt; this one is (deterministic at 5 of 6 residues). [review: "38/793" reads as 38 of 793 characters correct, not 38 wrong; and since it broke at 5 of 6 residues, record the KVarN arm at ≥2 doc lengths so one clean residue cannot hide it.]
- Arm (b), pinned length: both passes IDENTICAL, exit 0.
- Synthetic version: `test_kvarn_adaptive_prefix_shape`.
- Without `--passes 2` and `--doc-ctx`, bench-7 as currently specified (single pass, short prompts) **cannot** see this case at all.

## Open points

1. Real-server shapes unverified on the image: `POST /tokenize` with `messages`/`chat_template_kwargs`, `/v1/completions` `logprobs.top_logprobs` keyed `token_id:N`, and `top_logprobs=5` under the V2 runner. Run the harness smoke (bench-2 follow-up in the README: `greedy_equal.py record` with logprobs) plus a one-prompt replay first.
2. The bf16 ulp (0.125) assumption needs the real margin histogram.
3. Whether SPEC=off exists at CTX=huge (reference choice for the KVarN arm).
4. A tie in the first run does not prove the stack is deterministic: re-record the same boot once to measure run-to-run noise before trusting `near_tie` counts.
5. Proposed harness change (not made): have `greedy_equal.py record` take `--top-n`, keep top ids, and `--passes`; or make `run_suite.sh` greedy phase call `record_top2.py`. Also its `--near-tie` default 0.1 vs 0.01 here should be reconciled.

Re-run by: phase-4 reviewer, 2026-10-21, CONFIRMED WITH CORRECTIONS: 14/14 tests pass (pytest in place with no bytecode/cache, and as plain asserts). Corrections: `--ulp-band` vs EAGLE-4's 2-ulp arbiter, `@p2`/doc-ctx superset caveat, 38/793 reading, one line cite.
