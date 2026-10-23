# GPU results ledger

One row per item. The GPU agent keeps this file current and commits it with every result. Run instructions are in [../gpu-agent-prompt.md](../gpu-agent-prompt.md). Protocols and win criteria are in each item's "GPU validation" bullet in its appendix.

Status: `todo` · `running` · `done` · `blocked` · `skipped`. Verdict: `WIN` (clears the appendix's win criterion and the baseline spread) · `LOSS` · `NEUTRAL` (inside noise) · `INVALID` (correctness or lossless check failed) · `INFO` (measurement only, no win criterion).

## Phase 1: health and baselines

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| env | Box inventory: GPU, driver, power limit, WSL, Docker, models | – | done | INFO | 3090 at 250 W, driver 610.88, WSL2; bf16 DFlash2 fetched | [ENVIRONMENT.md](ENVIRONMENT.md) |
| health | `make verify` plus the repo's GPU-applicable `bench/test_*` | – | done | INFO | verify 0 FAIL; 39 tests pass/skip, 2 fail outside the serving path (stale 0.31 test API; WSL banner order) | [health.md](health.md) |
| cpu-checks | Re-run every script in `cpu-checks/`; compare with committed `.out` | – | done | INFO | 11/12 byte-identical; `quant_d_gptq` float noise only | [cpu-checks.md](cpu-checks.md) |
| baseline | Headline numbers per mode A–E on this box | health | done | INFO | A 814, B 110.4 e2e, C copy 358, D 99.8, D′ 55.7/48.8, E 57.8 tok/s; 6–14% under native docs (WSL2), A spills at 0.93 → 0.91 | [baseline.md](baseline.md) |

## Phase 2: measurement tools and probes

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| quant-2-check | `quant_real_ckpt_check.py` on the real checkpoint | – | done | INFO | 50.4% negative groups, 100% of them use −8; fold error 0.069% of weight energy | [quant-2-check.md](quant-2-check.md) |
| runtime-2 | `bench/step_curve.py`: ms/step vs verified tokens, per backend and depth | baseline | done | INFO | ms per verify token: FA2 ~0.03 (flat); pthfa 1.2 (q≤4), 2.6 at 150k for 4→8; KVarN 13.9 at 150k, 22.3 at 240k | [runtime-2.md](runtime-2.md) |
| evals-2 | `VLLM_GPU_SYNC_CHECK=warn` boot per mode | health | done | INFO | A, B, C, D, D′: 0 sites; E: 13 sites (KVarN metadata build, 3 syncs per step); negative control caught | [evals-2.md](evals-2.md) |
| evals-1 | Decode-vs-prefill KL probe per mode | health | done | INFO | k3 @4k: B 1.3e-4, D/D′ 1.4e-4, E 1.25e-3 (9.6×), A 4.3e-3 (33×), all-linear int8 1.1e-2 (88×); deterministic within a boot | [evals-1.md](evals-1.md) |
| evals-4 | Thinking-on speed and acceptance cohort | baseline | done | INFO | C1 thinking on (xhigh): B −14.8% (117.6 → 100.2), D −9.7%; 0 streams reached `</think>` in 1,024 tokens | [evals-4.md](evals-4.md) |
| evals-3 | Paired quality battery | health | done | INFO | known pair (all-linear int8): ppl +4.00% (docs +4.1%); greedy GSM8K b=2 c=3 p=1; sampled pooled b=14 c=6 p=0.115 | [evals-3.md](evals-3.md) |
| spec-5 | Σ min(p, q) acceptance telemetry | health | done | INFO | exact at T=0 (3.486 = 3.486); variance ratio across prompts only 1.03–1.07 | [spec-5.md](spec-5.md) |
| quant-1-calib | `bench/act_calib.py` per-128 and Hadamard-128 modes | health | done | INFO | per-128 ≥3× on 227/256 layers (down_proj 64/64 5.0×, in_proj_qkvz 48/48 5.7×); Humming int8 g128 0.26–0.41× Marlin W4A8 at M≤64 | [quant-1-calib.md](quant-1-calib.md) |
| attn-ncu | ncu on pthfa at 150k, T=8 (attention-kv open question 1) | health | blocked | INFO | ncu: ERR_NVGPUCTRPERM under WSL2 even after the Control Panel setting + Docker/WSL restart. Substitute microbench: T=8 @150k = 1,398 µs/layer at 24% DRAM BW (T≤4: 47–54%) | [attn-ncu.md](attn-ncu.md) |
| e-profile | Torch profile of E at 200k: attention share per step | health | done | INFO | at ~170k: attention 78% of a 151 ms step (KVarN stage1 117 ms, ~40× DRAM floor); E = 11.5 tok/s there | [e-profile.md](e-profile.md) |
| evals-5 | Special-token injection check (tokenizer on `<\|im_end\|>`), hardening if needed | health | done | INFO | vulnerable (typed markers → control ids, "PWNED" reply); escape patch: chat/tool markers 3/3 → 0/3, api_smoke 13/13, C1 −1.7% (noise); behavioural injection persists | [evals-5.md](evals-5.md) |
| evals-6 | Streamed tool-call test, tool case in `api_smoke` | health | done | INFO | parser 1,005 chunked streams 0 diffs; streamed tool call passes in all 6 modes | [evals-6.md](evals-6.md) |
| evals-7 | Golden greedy canary keyed by (GPU, pin, mode) | health | done | INFO | fresh boot, canary first: MATCH 8/8, Δlogp 0; after a session of other traffic: 3/8 diverge (B and D) | [evals-7.md](evals-7.md) |
| evals-8 | Per-component GPU time (nsys) | health | done | INFO | torch profiler (nsys absent): Marlin GEMMs 18–20 ms/step in every mode; attention 12% (B) → 30% (E @23k) → 78% (E @170k); GPU idle ≤10% (E worst) | [evals-8.md](evals-8.md) |
| evals-9 | Acceptance panel from DeepSpec's datasets (local copy, uncommitted) | health | done | INFO | B forced tok/step: humaneval 7.40 … arena-hard 3.36 (pooled 4.89); T=1 sampling −8% pooled | [evals-9.md](evals-9.md) |
| evals-10 | Throughput vs per-user Pareto sweep per mode | baseline | done | INFO | A knee N=16–32 (647 tok/s agg at N=32); at N=2–8 D (MTP) beats B (DFlash2): 423 vs 236 tok/s at N=8 | [evals-10.md](evals-10.md) |
| evals-11 | Automatic warmup detection | health | done | INFO | WARM after 3–4 runs; settling is thermal (C1 −9.4% over ~4 min, 62 → 83 °C), not JIT: the first run is the fastest | [evals-11.md](evals-11.md) |

## Phase 3: config and patch experiments

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| quant-1 | Per-128 int8 activation scales, route A (Humming) | quant-1-calib, evals-1 | done | LOSS | Humming int8 g128 at 0.26–0.41× of Marlin W4A8 at decode M≤64; +19% only at prefill M≥256; flag falls back to W4A16 | [quant-1.md](quant-1.md) |
| quant-2 | Lossless negative-scale fold, fix A | quant-2-check shows negative groups use −8 | done | LOSS (against the bar) | perplexity −1.0% (8.583 → 8.498, mode A), remap bit-exact; throughput −1.0 / −6.4 / −1.4% in 3 spill-free rounds (bar within 1%); remap runs ~3% lower clock at the 250 W cap | [quant-2.md](quant-2.md) |
| quant-7 | int4 embedding where `EMBED_UVA` can't be used | evals-1 | done | WIN (capacity) | KV pool +8.0% in D (207k → 224k), +16% in E (183k → 212k); PPL +0.14%, GSM8K PASS; decode −0.5 to −2% (drafter acceptance −1%), spill-free | [quant-7.md](quant-7.md) |
| quant-5 | Batch-mode lm_head as W4A8 | evals-1 | done | NEUTRAL | mode A 64-conc: +0.9 / +0.8 / −0.7% in 3 spill-free rounds (bar +2–3%); PPL −0.19%, GSM8K PASS | [quant-5.md](quant-5.md) |
| spec-0 | `DFLASH_TOKENS=3/5/7` ladder at 90k, 150k, 240k | runtime-2 | done | WIN | `DFLASH_TOKENS=3` vs 7 on E: +65% @90k, +56% @150k, +70% @235k (16.1 vs 9.5 tok/s); spill-free re-check +54% @150k; at 20k +7.8% total but copy −36%, code −11%; with attn-2 only +9% @170k | [spec-0.md](spec-0.md) |
| spec-1 | GDN opt-in to adaptive verification (= runtime-1) | runtime-2 | done | INVALID | greedy bit-identity fails at token 0–1 on all 6 tasks (0/23 clean chunks vs 18/23 off, off control identical); acceptance length 1.75 | [spec-1.md](spec-1.md) |
| kvarn-varlen | Root-cause KVarN corruption on verify-length change (`start_qwen.sh:399-428`) | – | running | | ~1 h of the 4 h timebox used: not reproduced (0 wrong at 6 residues); a hit inherits the previous request's verify schedule, and under the same schedule KV and GDN fingerprints are bit-identical (no corruption). Full 0–127 sweeps queued (see STATE.md) | |
| drafter-1a | Host-side confidence trim, stage 0 | runtime-2 | done | LOSS | +1.8–2.4% at 100k (bar +8%); copy −4–6% | [drafter-1a.md](drafter-1a.md) |
| spec-2 | Fresh-confidence device trim | spec-1 | skipped | | gate failed: spec-1 INVALID | [spec-1.md](spec-1.md) |
| spec-4 | Lookup-aware confidences | spec-1 | skipped | | gate failed: spec-1 INVALID | [spec-1.md](spec-1.md) |
| runtime-3 | Bound the tail of forced tool calls | – | done | WIN | runaways 6–10/40 → 0/40, valid calls 31–34/40 → 40/40, batch and single, required and named (`VLLM_TOOL_CALL_BOUND=1`) | [runtime-3.md](runtime-3.md) |

## Phase 4: kernel prototypes

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| attn-1.1 | pthfa unpadded 48-row tiles at T=8 (lossless) | attn-ncu | done | LOSS (alone, against the bar) | 1.23× per layer at 150k T=8 (bar 1.3×), lossless; the base that 1.2 and 1.3 build on | [attn-1.md](attn-1.md) |
| attn-1.2 | PV fp16 accumulate | attn-1.1 | done | WIN | 1.35× per layer at 150k T=8; D′ +9.2% at ~120k (45.85 → 50.05 tok/s), spill-free; pays only at T ≥ 6 | [attn-1.md](attn-1.md) |
| attn-1.3 | int8 QK | attn-1.2, evals-1 | done | WIN | 1.69× per layer at 150k T=8; D′ +10.6% at ~120k (45.85 → 50.7 tok/s); GSM8K n=600 94.8% vs 94.7%, needle 2/2 at 128k, PPL identical | [attn-1.md](attn-1.md) |
| attn-3 | int8-QK prefill on quantized-KV arms | evals-1 | running | WIN (mode D); mode E queued | D: cold TTFT −8.9% at 148k, −7.5% at 112k, −4.5% at 51k (bar ≥ 5%); 1.24× per layer-chunk; KL, PPL, GSM8K, needle unchanged; spill-free. E: kernel fixed after a shared-memory failure, runs queued | [attn-3.md](attn-3.md) |
| quant-6 | Fuse int8 activation quant into its producer | quant-1 | skipped | | gate failed: quant-1 is a LOSS (route A slower than Marlin); the appendix values fusion at ~0.5% of a 4k prefill | [quant-1.md](quant-1.md) |
| attn-5 | Hadamard-rotated int8/int4 per-token-head KV | evals-1 | done | LOSS | int8 in D: top-k KL vs bf16 −7.5% but PPL ratio ×1.012 (stock ×0.994); GSM8K, needle unchanged; decode −3% C1, −7% C4 (MTP acceptance −2.7%); int4 already rotated in vLLM | [attn-5.md](attn-5.md) |

## Phase 5: drafter tools

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| drafter-2 | Offline DFlash2 gate (`drafter/eval_dflash2.py`) | – | todo | | | |
| quant-3 | Better GPTQ for the int4 drafter and lm_head | drafter-2 | todo | | | |
| drafter-3 | TV-loss repair of the W4A16 DFlash2 | drafter-2 | todo | | | |
| drafter-1b | Learned confidence head, stage 1 | drafter-1a or spec-2 shows a win | skipped | | gate failed: drafter-1a LOSS, spec-2 skipped (spec-1 INVALID) | [drafter-1a.md](drafter-1a.md) |

## Phase 6: gated stretch

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| attn-2 | KVarN verify on a pthfa-family CUDA kernel | e-profile: attention ≥ 25% of E's step at 200k; attn-1.1 done | done | WIN | Triton shared-dequant verify (`KVARN_VERIFY_SHARED=1`): 4.7× per layer at 240k T=8; E +38% at 20k, +125% at 170k warm (16.0 → 36.0 tok/s), spill-free; GSM8K n=1,600 93.9% vs stock 94.75% (paired PASS), needle 3/3, KL and PPL unchanged | [attn-2.md](attn-2.md) |
| spec-3 | RecoverSSM for GDN | spec-1 WIN | skipped | | gate failed: spec-1 INVALID, not WIN | [spec-1.md](spec-1.md) |
| quant-4 | Block-Hadamard-128 rotation of linear inputs | quant-1 route A blocked or slower than Marlin by > 5% | todo | | | |
| attn-4 | Training-free top-k sparse attention | attn-2 done; needle and GSM8K gates ready | todo | | | |
| attn-6 | Device-side balanced split scheduler | appendix expects ~0; probe only if attn-ncu shows split imbalance | skipped | | gate not met: attn-ncu is blocked (no counters under WSL2), and the microbench shows uniform splits at batch 1 (21 splits/layer); imbalance needs mixed-length multi-stream pthfa traffic, which no mode runs | [attn-ncu.md](attn-ncu.md) |
| drafter-4 | Own-distribution fine-tune | needs an 80 GB GPU: out of scope here | skipped | | | |
