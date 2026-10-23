# DeepInfer expedition

What HyperQwen can take from deepseek-ai/DeepInfer, its submodules and its sibling DeepSpec, ranked by return on effort. Research and documentation only: nothing here changes serving. The research box has no GPU, so every gain below is an estimate, a simulation or a number quoted from HyperQwen's own docs. The GPU test plan at the end turns them into measurements.

Six lanes wrote the appendices; the lead verified citations, re-ran every CPU check from the committed copies and corrected numbers where needed (listed under [Lead corrections](#lead-corrections)).

| Appendix | Covers |
|---|---|
| [runtime.md](runtime.md) | Scheduler, CUDA graphs, determinism, structured outputs |
| [attention-kv.md](attention-kv.md) | Verify attention, KV formats, sparse attention |
| [spec-decode.md](spec-decode.md) | Verify-length policy, adaptive verification, GDN state |
| [drafter.md](drafter.md) | DSpark vs DFlash2, DeepSpec training, confidence head |
| [quant-heads.md](quant-heads.md) | Activation scaling, int4 GPTQ, heads, logits |
| [evals.md](evals.md) | Quality gates, sync checks, thinking-mode coverage |

## GPU results

Measured on one RTX 3090 at 250 W under WSL2 (Docker Desktop), 2026-10-20 to 10-23, against the stock image of main `ce9615c` (main `2cf50a0` before 10-22). Every experiment sits behind a default-off knob or a separately tagged image; the stock build is unchanged. Per-item numbers, protocols and raw-data paths: [gpu-results/LEDGER.md](gpu-results/LEDGER.md). Box and method notes: [gpu-results/ENVIRONMENT.md](gpu-results/ENVIRONMENT.md).

### Wins, ranked by what they buy

| Rank | ID | Mode | Knob | Measured |
|---|---|---|---|---|
| 1 | attn-2 | E | `KVARN_VERIFY_SHARED=1` | Decode **+125% at 170k** (16.0 → 36.0 tok/s), **+38% at 20k**; verify kernel 4.7× per layer at 240k. GSM8K n=1,600 within 1 point of stock, needle 3/3, KL and perplexity unchanged |
| 2 | spec-0 | E | `DFLASH_TOKENS=3` | **+54% at 150k**, +56–70% at 90–235k on the stock pool. At 20k +7.8% total but copy −36% and code −11%. With attn-2 in, only +9% at 170k and −12% at 20k |
| 3 | attn-1 stages 2–3 | D′ | `PTHFA_TILE48=1 PTHFA_MMA=pv16` / `s8pv16` | D′ decode **+9.2% / +10.6% at ~120k**; 1.35× / 1.69× per layer at 150k. Stage 3 quality unchanged (GSM8K n=600, needle at 128k). Nothing for D (T=4) |
| 4 | attn-3 | D, E | `PTHFA_PREFILL_QK=s8` (D), `KVARN_PREFILL_INT8=1` (E) | Cold TTFT **−8.9% (D) / −7.1% (E) at 148k**, −7.5% / −5.5% at 112k; under 5% below ~50k. Quality unchanged |
| 5 | quant-7 | D, E | int4-embedding checkpoint | KV pool **+8% (D), +16% (E)**; decode −0.5 to −2% |
| 6 | runtime-3 | A, single | `VLLM_TOOL_CALL_BOUND=1` | Forced tool calls that run to `max_tokens`: **6–10/40 → 0/40**; valid calls 40/40 |

The two E items stack differently than they look: attn-2 removes most of the verify cost that made a short draft (spec-0) pay, so with attn-2 on, k=7 is better at 20k and k=3 only edges it at depth.

### Neutral, losses, no-gos and invalid

| ID | Verdict | Why |
|---|---|---|
| quant-5 | NEUTRAL | int8 lm_head in A: +0.8% median, inside noise (estimate was +2–3%) |
| quant-2 | LOSS (vs bar) | lossless sign fold recovers 1.0% perplexity, but the zero-point path costs −1.4% throughput and ~3% clock at the power cap (bar: within 1%) |
| quant-1 | LOSS | Humming per-128 kernels run 0.26–0.41× Marlin at decode sizes |
| quant-3 | LOSS (vs bar) | signed scales + MSE clip + act-order cut the drafter's weight error 13% but move default-sampling acceptance only +0.12%; the whole int4 gap is 0.6% (drafter-2) |
| drafter-3 | LOSS (vs bar) | TV repair against the bf16 drafter: acceptance +0.17% at default sampling, greedy unchanged; its ceiling is the 0.6% gap |
| drafter-1a | LOSS | host confidence trim: +1–2.4% at 100k against an 8% bar (verify slope is 1.9 ms, not the assumed 3.6) |
| attn-5 | LOSS | Hadamard int8 KV: no quality gain on the real model (k_norm already flattens K), decode −3 to −7%; vLLM's int4 mode already rotates, and int4 KV costs ~16% perplexity |
| attn-1 stage 1 alone | LOSS (vs bar) | 1.23× per layer, bar 1.3×; it is the base the other two stages need |
| quant-4 | NO-GO | on real `down_proj` inputs the requant's second int4 rounding outweighs the activation gain (0.89×, gate 1.5×) |
| attn-4 | NO-GO | attention at 128k is diffuse: the top 10% of pages hold 0.34–0.85 of the mass, 0.61–0.94 even with oracle selection (gate 0.99) |
| spec-1 | INVALID | GDN opt-in to adaptive verification breaks greedy bit-identity at token 0–1; spec-2, spec-3, spec-4 skipped on its gate |

### What the measurements changed

- **Verify cost is 10× the simulation for KVarN.** 13.9–22.3 ms per verify token at 150–240k (runtime-2), not 1–2 ms. That is why spec-0 and attn-2 pay so much and why the trim items (drafter-1a) pay so little elsewhere.
- **The KVarN varlen corruption the launcher pins against does not reproduce.** 256/256 cache hits identical to cold across residues 0–127 on two corpora, adaptive length on; the only difference is schedule inheritance (kvarn-varlen).
- **The int4 drafter's sampling gap is 0.6%, not ~5%.** Paired on one capture, the W4A16 DFlash2 scores 0.994 of bf16 at default sampling; the old 3.2 vs 3.4 was two single runs inside ±5% noise (drafter-2). That removes most of the motivation for quant-3 and drafter-3.
- **Stock mode E spills to system RAM on this WSL2 box during long cold prefills** (+5% TTFT, ~10% decode right after), invisible to `nvidia-smi`. A 4.0 GB pool with 180k max length is spill-free. The probes that use `prompt_logprobs` (KL, perplexity) also spill in A and E; they ran in separate boots from all timings.
- **The new image is not slower.** A paired old-vs-new image A/B is equal within 0.5%; the apparent 4–5% dip came from spill and the box.
- **spec-1's "S patch" was wrong**: the device-side query length breaks bit-identity on GDN.
- **Thinking mode costs speed:** B −14.8%, D −9.7% with thinking on (evals-4).
- **ncu is unavailable under WSL2**; a microbenchmark stood in (T=8 at 150k is on-chip-bound, 24% of DRAM bandwidth).

### Decisions for you

1. **Ship attn-2 (`KVARN_VERIFY_SHARED=1`) as E's default.** Largest win of the run, quality unchanged on every gate. It is a Triton change; the CUDA-kernel route in the appendix was not needed.
2. **Retune E's `DFLASH_TOKENS` together with attn-2.** With attn-2 on, k=7 is better at 20k and k=3 is +9% at 170k. A context-dependent k (3 past ~100k) is the remaining spec-0 gain; without attn-2, k=3 is +54% at depth.
3. **attn-1: take stage 3 (`PTHFA_MMA=s8pv16`) or stop at stage 2 (`pv16`) for D′.** Stage 3 is lossy in principle but measured unchanged; stage 2 is near-lossless. D (MTP, T=4) gains nothing either way.
4. **attn-3 prefill knobs for D and E:** −7 to −9% cold TTFT at ~150k, nothing below ~50k.
5. **quant-7 (int4 embedding) for boxes that can't use `EMBED_UVA`:** +8–16% KV pool for −0.5 to −2% decode.
6. **runtime-3 (`VLLM_TOOL_CALL_BOUND=1`)** to end runaway forced tool calls.
7. **quant-2: keep the fold off, or fund fix B** (a kernel that doesn't cost clock); the 1.0% perplexity it recovers doesn't pay for −1.4% throughput.
8. **evals-5 escape patch** for typed `<|im_end|>` markers: closes control-token injection at −1.7% (noise) on C1; behavioural injection remains.
9. **Lift the kvarn-varlen launcher pin** in `single-user/start_qwen.sh`: the corruption did not reproduce. Re-check on the native box first if it was seen there.
10. **Multi-seat (2–8 streams) favours MTP mode D over DFlash2 B** (423 vs 236 tok/s at N=8, evals-10).
11. **Mode E on WSL2: use the spill-free pool** (`KV_MEM=4000000000 DFLASH_MAX_LEN=180224`), or accept +5% TTFT from paging on long cold prefills.
12. **quant-5 is harmless but not worth shipping** (+0.8%).

## Executive summary

1. **DeepInfer is built for H100/B200 MoE serving: FP8/FP4, MLA, data-parallel lockstep.** None of its kernels run on sm86. What transfers is method: cycle budgeting for attention, confidence-scheduled verify length, fine-grained activation scaling, a decode-path KL gate, and DeepSpec's TV loss and confidence head.
2. **There is no runtime overhead to win.** Host time is under 2% of a step, and mode A's 942 tok/s is fully explained by arithmetic (runtime).
3. **DFlash2 stays.** It beat DSpark on this 3090 (3.83 vs 3.37 tok/step, 150 vs 114 tok/s). Its improvements come from around it, not from replacing it:
   - **Verify length.** vLLM 0.31 already ships DeepInfer's confidence-scheduled verify. It is off for this model only because GDN refuses device-side query lengths, and upstream's KDA opt-in shows the unblock is small (spec-1, S patch). It pays at a few concurrent streams (C4 +6–7% over `DYNAMIC_K`). At one stream it pays only at long context, and only if the trim uses *fresh* confidence (spec-2, drafter-1).
   - **The int4 drafter's sampling gap.** The W4A16 DFlash2 keeps greedy acceptance but loses about 5% at default sampling. Better GPTQ (quant-3) and a TV-loss repair of its bf16 parameters (drafter-3) both target it; an offline tau_prob gate (drafter-2) measures it without ±5% run noise. *Measured: 0.6% (drafter-2).*
4. **Deep-context verify is bound by the tensor pipe, not DRAM.** Cheaper MMAs save about 4.6 ms per step at 150k in D′ (attn-1, inferred; ncu first). The first two stages, removing row padding at T=8 (lossless) and accumulating PV in fp16 (near-lossless), get most of it at boost clock. Moving KVarN's verify onto the same CUDA kernel is the largest item for E: about +21% at 240k (attn-2, inferred, needs a profile first).
5. **Mode A's lever is activation scaling.** Per-token × per-128 int8 activation scales cut activation error 4–7× on heavy-tailed inputs; vLLM 0.31 can already serve it through the Humming kernels behind one flag. If quality allows int8 on all layers: A up to +10.6%, single-user prefill +27–30% (quant-1).
6. **Mode D stays on MTP.** DFlash2's long-context loss is verify cost and a larger fixed step; even perfect trimming leaves it at about 41 vs MTP's 62–73 tok/s at 112k (drafter Q3).
7. **The biggest unknown behind the headline numbers is thinking mode.** Every speed and acceptance number in the repo is measured with thinking off, while the chat template defaults to xhigh effort (evals-4).
8. **Two things to check cheaply first.** The negative-scale fold in `marlin-int8-negative-scales` may drop an LSB on the largest weight of about half of all groups, not "extremely rarely"; one CPU script on the real checkpoint settles it (quant-2). And there is no recorded step-cost curve, which is the single number most of the verify-length items depend on (runtime-2).

## Roadmap, ranked by ROI

Modes: A batch; B DFlash2 64k; C quoting (T=16); D `SPEC=mtp CTX=long`; D′ `SPEC=dflash2 CTX=long PTH_FA=1`; E `SPEC=dflash2 CTX=huge` (KVarN). "Lossless" means output is unchanged in distribution.

### Do first: measurement that gates everything else

| # | ID | What | Effort | Why first | Measured |
|---|---|---|---|---|---|
| 1 | quant-2 (check) | `quant_real_ckpt_check.py` on the real AutoRound checkpoint | minutes, CPU | Confirms or closes quant-2 | Confirmed: 50.4% of groups negative, all use −8; fold error 0.069% of weight energy |
| 2 | runtime-2 | `bench/step_curve.py`: ms/step vs verified tokens per backend and depth | S | Settles the verify cost per token (3.6 vs 1.2 ms at 112–128k) and the stairs; spec-0/1/2, drafter-1 and runtime-1 all hinge on it | FA2 flat (~0.03 ms/token); pthfa 1.2 ms (q≤4), 4→8 stair 2.6 ms at 150k; KVarN 13.9 ms at 150k, 22.3 at 240k |
| 3 | evals-2 | Boot every mode with `VLLM_GPU_SYNC_CHECK=warn` | S | Catches hot-path syncs; already in the pin | A–D′ clean; E 13 sites (KVarN metadata, 3 syncs/step) |
| 4 | evals-1 | Decode-vs-prefill logprob KL probe per mode | S (4–16k) | The quality gate for every lossy item below | k3 at 4k: B/D 1.3–1.4e-4, E 9.6×, A 33×, all-linear int8 88×; used as the gate throughout |
| 5 | evals-4 | Thinking-on speed and acceptance cohort | S | Tests the headline numbers | Thinking on: B −14.8%, D −9.7%; no stream reached `</think>` in 1,024 tokens |
| 6 | evals-3 | Paired quality battery (per-item JSONL, McNemar/Wilson) | S | Makes 1-point quality changes visible | Reproduces the known +4.0% perplexity pair; used as the gate throughout |
| 7 | spec-5 | Σ min(p, q) expected-acceptance telemetry | S | Lower-variance acceptance A/Bs | Exact at T=0, but variance across prompts only 1.03–1.07× lower |

### Speed and capacity

| Rank | ID | What | Modes | Gain | Effort | Conf. | Lossless | Measured |
|---|---|---|---|---|---|---|---|---|
| 1 | quant-1 | Per-token × per-128 int8 activation scales (Humming route first) | A; prefill B–E | A up to +10.6%; prefill +27–30% | M | Medium | No (quality gate) | LOSS: Humming 0.26–0.41× Marlin at decode |
| 2 | spec-1 = runtime-1 | Opt GDN into vLLM's adaptive verification (mirror #52988) | C at 2–4 streams; E | C4 +6–7% over `DYNAMIC_K`; E +0.7 to +15% over k=7 | S patch, M validation | Medium | Yes | INVALID: breaks greedy bit-identity on GDN |
| 3 | attn-1 | pthfa verify: unpadded 48-row tiles at T=8, PV fp16 accumulate, then int8 QK | D′ (E via attn-2) | −4.6 ms/step at 150k; stages 1–2 alone reach the DRAM floor at boost clock | M | Medium | Stage 1 yes; 2 near; 3 no | WIN (stages 2–3): D′ +9.2% / +10.6% at ~120k; stage 1 alone 1.23× (bar 1.3×) |
| 4 | quant-7 | int4 embedding where `EMBED_UVA` can't be used | D, E | KV pool +8.4% | S | High | No (small) | WIN (capacity): pool +8% D, +16% E; decode −0.5 to −2% |
| 5 | spec-0 | Context-aware fixed `DFLASH_TOKENS` at 240k | E | −2.5 to +12.5% | S | Medium-low | Yes | WIN: k=3 +54% at 150k, +70% at 235k; +9% at 170k with attn-2 on |
| 6 | quant-3 | Better GPTQ for HyperQwen's own int4 requants | B, C, E (drafter); D (MTP) | DFlash2 +2–3% at default sampling | S | Medium | Drafter yes | LOSS vs bar: drafter acceptance +0.12% (weight error −13%) |
| 7 | attn-2 | KVarN verify on a pthfa-family CUDA kernel | E | ~+21% at 240k, ~+2% at 20k | L | Medium-low | Yes | WIN (Triton shared dequant): E +125% at 170k, +38% at 20k |
| 8 | drafter-3 | TV-loss repair of the W4A16 DFlash2's bf16 params | B, C, E | ~+3% (ceiling +6%) at default sampling | M | Medium | Yes | LOSS vs bar: acceptance +0.17%; the bf16 ceiling is +0.6%, not +6% |
| 9 | spec-2 + drafter-1 | Per-step verify trim on fresh confidence | D′/`CTX=long`, E | E +4 to +10%; 112k +4 to +20% (linear-cost ceiling) | S (host stage 0) to M | Low-medium | Yes | drafter-1a LOSS (+1–2.4% at 100k); spec-2 skipped |
| 10 | quant-5 | Batch-mode lm_head as W4A8 | A | +2–3% | S–M | Low-medium | No | NEUTRAL: +0.8% |
| 11 | attn-3 | int8-QK prefill on quantized-KV arms | D, E | cold TTFT −8 to −9% at 150k | M | Medium | No | WIN: TTFT −8.9% D, −7.1% E at 148k |
| 12 | spec-3 | RecoverSSM for GDN | DFlash2 residency | residents ~6 → 20+; `CTX=long` reach 114k → 132k | L | Medium | Yes | skipped (spec-1 INVALID) |
| 13 | attn-4 | Training-free top-k sparse attention | E | up to +30% at 240k | L | Low | No | NO-GO: top 10% of pages hold 0.34–0.85 of the mass (oracle ≤ 0.94) |
| 14 | quant-4 | Block-Hadamard-128 rotation of linear inputs | as quant-1 | as quant-1, on unchanged Marlin | L | Medium | No | NO-GO: activation-error gain 0.89× (gate 1.5×) |
| 15 | spec-4 | Lookup-aware confidences in the verify budget | C | C1 0 to +3% | M | Low | Yes | skipped (spec-1 INVALID) |
| 16 | quant-6 | Fuse the int8 activation quant into its producer | prefill | ~0.5% of a 4k prefill | M | Medium | Yes | skipped (quant-1 LOSS) |
| 17 | drafter-4 | Own-distribution, thinking-mode DFlash2 fine-tune | B, C, E | unknown | L + 80 GB rental | Low | Yes | skipped (needs an 80 GB GPU) |

### Quality, robustness and tooling

| ID | What | Effort | Notes | Measured |
|---|---|---|---|---|
| quant-2 | Lossless negative-scale fold (load-time remap onto the existing `kS8 × kU4` kernel) | S | Only if check 1 confirms it | LOSS vs bar: perplexity −1.0%, throughput −1.4% (bar within 1%) |
| drafter-2 | Offline DFlash2 gate: tau_prob and accept@position per domain | M | Required before drafter-3 or drafter-4 | Built and validated; the int4 drafter's sampling gap is 0.6%, not ~5% |
| attn-5 | Hadamard-rotated int8/int4 per-token-head KV | S–M | Quality for D; int4 would halve D's DRAM floor | LOSS: no quality gain on the real model, decode −3 to −7% |
| runtime-3 | Bound the tail of forced tool calls | S–M | 5–10% of forced calls run to `max_tokens` | WIN: runaways 6–10/40 → 0/40 (`VLLM_TOOL_CALL_BOUND=1`) |
| evals-5 to evals-11 | Special-token hardening, streamed tool calls, golden canary, per-component GPU time, broader acceptance panel, Pareto sweep, warmup detection | S–M each | See [evals.md](evals.md) | evals-5: typed `<\|im_end\|>` injects control tokens (escape patch fixes it); evals-6 passes; evals-7 canary matches on a fresh boot only; evals-8 attention 12% (B) → 78% (E at 170k); evals-10 D beats B at 2–8 streams; evals-11 settling is thermal |

## Threads that cross lanes

- **Verify length is one feature in four pieces.** runtime-2 measures the cost → spec-0 picks a better fixed width at E → spec-1 unblocks vLLM's adaptive verification on GDN → spec-2 (device mechanism) plus drafter-1 (confidence signal) trim per step. Two open issues shape all of it:
  - The cost per verify token is uncertain by 3×: drafter's 3.6 ms at 112k comes from the Triton int8 arm and is an upper bound; spec's ~1.2 ms at 128k is scaled from a 3080. Costs also come in stairs: Marlin is flat to 16 query tokens, and pthfa is flat from T=6 to T=10, so trimming 8 → 6 saves no attention there.
    - **Measured (2026-10-21, runtime-2):** the answer depends on the backend. FA2-fp8 is flat (~0.03 ms per extra verify token at 64–150k). pthfa costs ~1.2 ms per token for q 1→4, then a 4→8 stair of 1.9 ms per token at 112k and 2.6 at 150k (trimming 8 → 4 saves 18–21% of the step; 8 → 6 saves nothing). KVarN is linear and steep: 5.8, 10.3, 13.9 and 22.3 ms per token at 64k, 112k, 150k and 240k (8 → 4 is −41% of the step at 240k). Serving pads verify width to 1, 2, 4 or 8. [gpu-results/runtime-2.md](gpu-results/runtime-2.md)
  - **Blocker for E:** any step-to-step change in verify length corrupts a KVarN prefix-cache hit (`single-user/start_qwen.sh:399-428`, 38/793 wrong). Root-causing it unblocks spec-1, spec-2, spec-4 and drafter-1 in E.
- **The int4 DFlash2 sampling gap** (3.2 vs 3.4 tok/step) is attacked from two sides, quant-3 and drafter-3, and should be measured with drafter-2 and spec-5 rather than 8-prompt runs.
  - **Measured (2026-10-23, drafter-2):** paired on one capture, the gap is 0.6% at default sampling (W4A16/bf16 tau_chain 0.994) and 0.3% greedy. The 3.2 vs 3.4 came from single unpaired live runs inside their ±5% noise. [gpu-results/drafter-2.md](gpu-results/drafter-2.md)
- **Hadamard rotation helps integer grids and hurts float grids.** attn-5 and quant-4 reached this independently: Hadamard plus int8/int4 beats NVFP4 or MXFP8, which have no MMA on sm86 anyway.
- **Thinking mode** is unmeasured for speed (evals-4) and for drafter acceptance (drafter-4); 45% of the drafter's own training prompts are thinking-on.
- **Batch speculation** (A with a drafter) is blocked by KV residency, not policy; it needs spec-3's RecoverSSM first.
- **Measurement noise** (±3% greedy, ±5% sampled per run) hides most single-digit wins; evals-3, spec-5 and drafter-2 exist to fix that.

## GPU test plan

In order; each step names the items it decides. Commands are in the appendices. Run 2026-10-20 to 10-23 on one RTX 3090 under WSL2; the outcome follows each step (details in [gpu-results/LEDGER.md](gpu-results/LEDGER.md)).

1. **Real-checkpoint fold check** (CPU only, minutes): `cd docs/research/deepinfer/cpu-checks && python3 -E -P quant_real_ckpt_check.py <path to Qwen3.8-27B-W4A16-AutoRound>`. Decides quant-2. *Done: confirmed (50.4% of groups); the fold then lost on throughput.*
2. **Step-cost curve** (runtime-2): write `bench/step_curve.py`; ms/step vs total query tokens 1–32 under each launcher with CUDA graphs on, at 8k, 25k, 64k, 112k, 150k and 240k, on FA2-fp8, pthfa and KVarN Triton. Decides spec-0, spec-1, spec-2, drafter-1 and runtime-1. *Done: KVarN 13.9–22.3 ms per verify token at 150–240k; pthfa has a 4→8 stair.*
3. **Sync check** (evals-2): boot all modes with `VLLM_GPU_SYNC_CHECK=warn`; record the allowlist. *Done: only E syncs (13 sites).*
4. **Decode KL probe** (evals-1) at 4–16k per mode, two runs for spread. Becomes the gate for steps 6, 7 and 9. *Done; used as the gate.*
5. **Thinking-on cohort** (evals-4): `labd_bench` and `labd_accept` with `enable_thinking: True`, split by reasoning vs answer phase. *Done: B −14.8%, D −9.7% with thinking on.*
6. **Activation error per layer** (quant-1 step 1): extend `bench/act_calib.py` with per-128 and Hadamard-128 modes using the quantizers in `quant_a_act_quant.py`. *Done: per-128 ≥ 3× better on 227/256 layers.*
7. **Humming probe** (quant-1 route A): is `humming` importable, does int8 `DynamicGroup` run on sm86, speed vs Marlin with `--linear-backend humming`, and does it capture the int8 lm_head and embedding correctly. *Done: 0.26–0.41× Marlin at decode; quant-1 LOSS.*
8. **ncu on pthfa** at 150k, T=8, on a 3090: tensor-pipe vs DRAM utilisation. Decides attn-1. *Blocked: no performance counters under WSL2; a microbenchmark stood in (24% of DRAM bandwidth at T=8).*
9. **E profile at 200k** (torch profile): the real attention share per step. Decides attn-2 and spec-0's sign. *Done: attention is 78% of the step at ~170k.*
10. **spec-1 boot and greedy bit-identity** on `bench/labd_accept.py`, then `bench/conc_ladder.py` at C1, C2, C4 against `DYNAMIC_K`. Also answers whether FA2-fp8 accepts device q_len below the host bound. *Done: INVALID, bit-identity breaks at token 0–1.*
11. **spec-0 ladder**: `DFLASH_TOKENS=3/5/7` at 90k, 150k and 240k, three boots per arm. *Done: k=3 wins at depth (+54–70%).*
12. **Drafter work** once drafter-2 exists: quant-3 requant, then drafter-3, each measured by tau_prob before any tok/s run. *Done: the int4 gap is 0.6%, not ~5%; quant-3 recovers +0.12% and drafter-3 +0.17%, both far below their bars, so no tok/s run.*

## CPU checks run on the research box

All in [cpu-checks/](cpu-checks/), run from that directory. Scripts that need torch use `python3 -E -P` because torch is in the user site-packages, which `-I` hides. The lead re-ran every script from its committed copy; outputs match the lanes' byte for byte.

| Script | Lane | Checks | Command |
|---|---|---|---|
| `runtime_topk_sim.py` | runtime | Global top-k verify vs best uniform k (N=1 is 0% by construction) | `python3 -I runtime_topk_sim.py` |
| `attn_roofline.py` | attention | Cycle budget and roofline per mode | `python3 -I attn_roofline.py` |
| `attn_kvq_error.py` | attention | KV formats: int8/int4, NVFP4, MXFP8, Hadamard | `python3 -I attn_kvq_error.py` |
| `attn_mma_precision.py` | attention | PV fp16 accumulate, int8 QK error | `python3 -I attn_mma_precision.py` |
| `spec_varlen_sim.py` | spec | Fixed k vs adaptive, stale vs fresh, per mode | `python3 -I spec_varlen_sim.py` (~60 s) |
| `spec_calib.py` | spec | Grid search for the acceptance parameters | `python3 -E spec_calib.py` |
| `quant_a_act_quant.py`, `quant_a2_gauss.py` | quant | Activation quant error by scheme | `python3 -E -P <script>` |
| `quant_b_signfold.py` | quant | Negative-scale fold cost and lossless remap | `python3 -E -P quant_b_signfold.py` |
| `quant_c_logits_bf16.py` | quant | bf16 vs fp32 logits KL | `python3 -E -P quant_c_logits_bf16.py` |
| `quant_d_gptq.py` | quant | GPTQ variants, importing `drafter/gptq_utils.py` | `python3 -E -P quant_d_gptq.py` |
| `quant_real_ckpt_check.py`, `quant_mk_fake_ckpt.py` | quant | Fold check for a real checkpoint (GPU-box step 1), with a fake checkpoint to self-test | `python3 -E -P quant_real_ckpt_check.py <dir>` |
| `drafter_checks.py` | drafter | Sizes, DeepSpec cache, DSpark heads and loss, vocab slice, 112k step anatomy, confidence trim | `python3 -E -P drafter_checks.py` |

## Lead corrections

Changes the lead made to lane findings before committing them:

- **E's DRAM floor** at 240k is 3.45 ms, and E's 67 tok/s was measured at `--ctx 20000`; every E gain grows with depth from there, and E's attention cost at 200k+ on a 3090 is unmeasured (attention-kv).
- **attn-1 stage 1** (unpadded 48-row tiles at T=8) is the lead's addition; with PV fp16 accumulate it reaches D′'s DRAM floor without a lossy step.
- **spec's linear cost model ignores the stairs**; on pthfa, k=7 → 5 saves no attention, k ≤ 4 does (spec-decode).
- **spec-1 is S, not M**, following #52988, with a caveat that KDA's metadata also carries `RecoverSSMMetadata`; validate GDN bit-identity first. runtime-1 is the same item and is amended to match.
- **runtime-1's "zero at N=1"** holds only for a budget fixed from stale rates; the simulation's N=1 result is 0% by construction (runtime, amended).
- **drafter-1's 3.6 ms slope** is an upper bound from the Triton int8 arm and conflicts with spec's ~1.2 ms; runtime-2 settles it (drafter).
- **quant-2's wording**: the patch already expects ~50% negative scales; the question is whether the clamp of code −8 is rare, and under the lane's reconstruction it isn't (quant-heads).

## Rejected, across lanes

- **DeepInfer kernels and formats:** FlashMLA, DeepGEMM, DeepSelect and TileLang paths are SM90/SM100 only. MXFP8 KV equals per-tensor e4m3 in precision on sm86; NVFP4 KV has no MMA here and no edge over KVarN's Hadamard-int4. Power-of-two scales and fp8 activations lose to int8 per-128.
- **Runtime:** porting DeepInfer's zero-overhead scheduler (0.7–1.7% host time to hide), prefill/decode separation, full batch invariance (GDN backends reject it, and Marlin's tiling varies).
- **Drafters:** DSpark as a replacement (lost on this card), grafting its Markov head onto DFlash2 (+0.95 ms/step), EAGLE3 (MTP's niche), the 40,960-row vocab slice for DFlash2 (−2.6 to +2.2%), a long-context retrain to rescue D, training a full drafter on the 3090.
- **Heads and numerics:** fp32 logits (KL 1–6e-4 vs the int4 head's 2.4e-3), W8A16 or bf16 DFlash2 drafter (+1.36 ms/step), per-channel int4 lm_head, sub-4-bit body.
- **Evals:** exact golden-decode perplexity as the main gate (needs batch invariance), AIME/GPQA/HLE per change (hours per run), LLM judges.
- **attn-6**, DeepInfer's balanced split scheduler: about 0 as HyperQwen is configured.
- Everything multi-node, MoE, EP, DP, PD-disaggregated or Ascend.

Each appendix lists its own rejections with reasons.

## Open questions for you

Decisions:
1. Do you want multi-seat speculation (DFlash2 at 2–16 concurrent users)? It decides how much spec-1/runtime-1 is worth beyond C4.
2. Do you want DFlash2 at 64k+ (`CTX=long` or E)? drafter-1 and spec-2 pay there and almost nowhere else.
3. How much real traffic runs thinking mode, and at what effort? It decides evals-4's priority and whether drafter-4 is worth a look.
4. Will you carry more GDN patches across vLLM pins (spec-1, spec-3)?
5. Would you rent one 80 GB GPU for a few hours (drafter-4 only)?
6. Does the forced-tool-call runaway hurt real agent traffic? If so, runtime-3 moves up.
7. Can DeepSpec's eval datasets be redistributed inside HyperQwen (evals-9)? Does incoai publish DFlash2's training data or selector loss (drafter-3, drafter-4)?

Facts the GPU box answers (the test plan covers them): the real cost per verify token and its stairs; Humming on sm86; ncu on pthfa; E's attention share at 200k; whether FA2-fp8 takes device query lengths; how well vLLM's online estimator fits DFlash2; how autocorrelated acceptance is step to step; the KVarN verify-length root cause; `prompt_logprobs` memory at 64–100k; how Qwen3.8's tokenizer treats a typed `<|im_end|>`.

## Method and sources

| Lane | Model, effort | Appendix |
|---|---|---|
| Engine runtime | Sonnet, high | [runtime.md](runtime.md) |
| Evaluation and tooling | Sonnet, medium | [evals.md](evals.md) |
| Attention and KV cache | Opus, high | [attention-kv.md](attention-kv.md) |
| Speculative decoding | Opus, high | [spec-decode.md](spec-decode.md) |
| Quantization and heads | Opus, high | [quant-heads.md](quant-heads.md) |
| Drafters | Opus, high | [drafter.md](drafter.md) |

Lanes were read-only on all sources. Each wrote findings with file:line citations; the lead spot-checked citations, re-ran the CPU checks and reconciled conflicts between lanes.

Pinned sources:
- deepseek-ai/DeepInfer `774ce24`, with its `vendor/` submodules (DeepGEMM `1fc9c66`, FlashMLA `2e5429f`, DeepEP `93eb6eb`, DeepSelect `bfa4507`).
- deepseek-ai/DeepSpec `005e03b`.
- deepseek-ai/deepseek-recipe `935817c`.
- vLLM `v0.31.0` (HyperQwen's pin) and upstream main `4bd75e952a`.
- HyperQwen `main` at `95ac251`.

Mode numbers used throughout: A 942 tok/s batch; B 127 tok/s; C 381 tok/s; E 67 tok/s at `--ctx 20000`.
