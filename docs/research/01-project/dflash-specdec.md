# DFlash2 and the other speculative-decoding paths: mechanism, cost, bottlenecks

Lane: specdec-analyst (P1 speed/inference). Status: desk research, no GPU. Date: 2026-10-21.

## How to read the citations

- **`vllm/...` paths with no qualifier** refer to the vLLM tree this project actually runs: v0.31.0 plus
  `patches/series`. To rebuild it, run `git archive v0.31.0 vllm | tar -x -C X`, then
  `bash patches/apply.sh` against `X`. Line numbers are from that tree. ~~Most DFlash2 code is not
  upstream code. Our patches add it.~~ [review: wrong. The drafter model, candidate head, selector walk
  and speculator are upstream in v0.31.0 (`git ls-tree v0.31.0` lists `qwen3_dflash2.py` and
  `spec_decode/dflash2/speculator.py`, 265 lines). Our patches add `dflash2/lookup.py`, the lookup
  fusion, chain mode, the adaptive long block and prewarm (speculator grows 265 → 865 lines).]
- **`HEAD:vllm/...`** refers to the reference checkout `~/vllm` at v0.32.1rc0-68 (4bd75e9). It is only
  cited for hooks section 8 needs.
- **Repo files** (`single-user/start_qwen.sh`, `docs/*.md`, `drafter/README.md`) are cited from the
  repo root.
- **DeepInfer evidence** lives in `/home/tyrone/research-src/deepinfer-ref/docs/research/deepinfer/`.
  Paths below shorten this to `DI/`. Those runs were on a WSL2 RTX 3090 capped at 250 W. They read
  6-14% below native Linux (`DI/gpu-results/baseline.md`), so compare their ratios, not their
  absolute values, against repo numbers.
- **Number labels.** **[M]** means measured, with the path:line of the raw evidence. **[E]** means
  estimated or derived, with the arithmetic shown.

## 1 Summary

`SPEC=dflash2` drafts with **DFlash2**, a 5-layer, non-causal, block-parallel drafter. It is
conditioned on target hidden states from 5 target layers. Each step:

1. It runs one forward pass over `[bonus token, 7 mask tokens]`.
2. It takes the top-16 candidate tokens per slot from the target's own quantized lm_head.
3. It picks one chain of candidates with a learned low-rank transition score between neighbouring
   slots: a Markov walk with a Gumbel-argmax.
4. It fuses that chain with a **suffix-match lookup** over the request's history (LABD, on by
   default). Lookup tokens become point-mass draft distributions.
5. On copy-like steps, the block may be extended to 15 drafts ("long block").

The target verifies the whole chain in one forward pass. Verification is either greedy or
Leviathan-style rejection sampling against the drafter's cached candidate logits. Every path in
both vLLM trees is a **chain**: no tree attention exists in 0.31 or at HEAD. The hybrid target's
48 GDN layers handle speculation by writing one recurrent state **per verified position** into
`k+1` slots per request. The next step then reads the slot of the last accepted token.

The step is dominated by a fixed weight-streaming floor of about 22 ms [E, §6]. Speculation adds
about 4.8 ms [E]. [review: 22 ms is all non-attention, non-drafter target work (and is taken from a
2×3080 TP=2 profile), not weight streaming. The byte floor is 13.25 GB of target weights ÷ 936 GB/s
= 14.2 ms; e-profile measures Marlin + lm_head at 18.5 ms on WSL2. The 4.8 ms uses a 46 tok/s
baseline from an older config. The recomputed overhead is 4.5-6.5 ms [E]. See
`00-meta/reviews/phase1-dflash-specdec.md`.] So at short context the throughput lever is **accepted length**, not draft cost.
At long context the lever becomes **verify attention**, which is 78% of the step at about 170k [M].
The cost of concurrency is the **per-position recurrent-state residency**: 0.88 GiB per resident
and about 7 ms per resident [M]. The cheapest wins are ranked in §7.

MTP is the other production path: Qwen's 1-layer autoregressive head with k=4 (fast) or k=3
(long/huge). The proposers in the tree but not wired into our launchers are DSpark, EAGLE/EAGLE3,
draft_model, multi_module_mtp, LiLiCorr and Gemma4 (§5).

## 2 Drafting algorithm

### 2.1 Configuration surface (launcher → vLLM)

| knob | where | effect |
|---|---|---|
| `SPEC=dflash2` | `single-user/start_qwen.sh:382-398` | `method=dflash`, the DFlash2 checkpoint, `num_speculative_tokens=${DFLASH_TOKENS:-7}` (382), `draft_sample_method=probabilistic` (394). The verify query is `QMAX=k+1` (398). |
| `SPEC=mtp` | `start_qwen.sh:212,217,222,666-667` | `DRAFT_TOKENS` fast=4, huge=3, long=3. CUDA-graph cap 32 (667). |
| `LOOKUP` (default 1) | `start_qwen.sh:370` → `VLLM_DFLASH2_LOOKUP` | The env default in vLLM is **off** (`vllm/envs.py:212,1620`). The launcher turns it on. |
| `DFLASH_TOKENS=15` | `start_qwen.sh:382` | The scheduled maximum k is 15. The drafter still drafts its trained block of 7, and the lookup fills 8-15 (`dflash/speculator.py:62-91`). |
| `ADAPTIVE` | `start_qwen.sh:399-429` | The adaptive long-block trigger. Forced to 0 on `CTX=huge` with prefix caching, because of KVarN corruption (see §7 item 2). |
| async scheduling | `start_qwen.sh:430-435` | Turned off when the lookup is on and k>7. The draft-count feedback only runs on the synchronous path (§4.3). |
| `DYNAMIC_K` | `start_qwen.sh:678-692` | Adds `num_speculative_tokens_per_batch_size`. Refused without speculation (685) and on huge with prefix caching (687). |
| `PTH_FA=1` + dflash2 | `start_qwen.sh:280-335` | Refuses DRAFT_METHOD≠dflash, DFLASH_TOKENS≠7, LOOKUP≠1 and DYNAMIC_K (324-327). |

### 2.2 Drafter model (DFlash2)

Model code: `vllm/model_executor/models/qwen3_dflash.py` (the base) and `qwen3_dflash2.py` (the
candidate head).

- **Shape** [M, checkpoint config; summarised in `drafter/README.md:64-70`]: 5 Qwen3 decoder
  layers, hidden 5120, 32 query and 8 KV heads × 128, MLP 17408. The sliding window is 2048, and
  the layers are non-causal (`qwen3_dflash.py:86-97` `_dflash_layer_causal`). There are grouped
  dynamic convolutions in attention and MLP (`qwen3_dflash2.py:144-226`). 1.92B parameters:
  W4A16 1.19 GB, bf16 3.85 GB.
- **Context conditioning.** `fc` maps the concatenated target hidden states from layers
  5/19/33/47/61 to the drafter's hidden size (`qwen3_dflash.py:108-117,507-518`). The model runner
  registers these as the "aux hidden state" layers (`vllm/v1/worker/gpu/model_runner.py:391-403`).
- **Context KV is precomputed, not run through the layers.** For the new context rows of each step
  (bonus plus accepted tokens), `precompute_and_store_context_kv` (`qwen3_dflash.py:665-740`) runs
  RMSNorm, then **one fused K/V GEMM for all 5 layers** (`_build_fused_kv_buffers` 561-602,
  `_project_context_kv` 604-642), then K-norm, then RoPE, and inserts into each layer's own
  sliding-window paged KV cache. The drafter therefore has its **own KV cache**, as KV-cache
  groups named by `draft_attn_layer_names` (`model_runner.py:579,637`).
- **Query.** The bonus token plus `block_size-1 = 7` mask tokens, `num_query_per_req = 1+draft_block`
  (`dflash/speculator.py:91`). The mask tokens attend non-causally to each other and to the window.
  One forward pass gives 8 hidden rows, and the last 7 are draft slots.

### 2.3 Candidate head and selector walk (what DFlash2 adds over DFlash)

`DFlash2Speculator._generate_draft` (`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:723-777`)
runs these stages:

1. `_run_model`: the drafter forward pass above, run as a FULL CUDA graph when the batch is uniform
   (§4.1).
2. `compute_candidates` (`qwen3_dflash2.py:402-420`): the target's **shared int4 lm_head (248k
   vocab)** on the 7 draft rows, then top-16 per slot. This is a full-vocabulary GEMM per draft row.
   It is the reason the drafter reads the 0.65 GB lm_head every step (§6).
3. `_score_candidates` → `CandidateSelector` (`qwen3_dflash2.py:297-365`). `_score_edges` computes a
   unary score plus a low-rank bilinear `einsum` between each predecessor candidate and each
   successor candidate. That gives a **K×K = 16×16 transition matrix per slot**.
4. `candidate_sampler.sample` → `_selector_walk_kernel` (`dflash2/speculator.py:28-110`). This is a
   left-to-right Markov walk. At slot i it takes the row of the transition matrix indexed by the
   candidate chosen at slot i−1, applies optional top-k/top-p truncation, and takes a
   `gumbel_noised_argmax`. At temperature 0 this is plain argmax. The output is **one chain** of 7
   tokens.
5. `_cache_draft_logits` (`dflash2/speculator.py:113-146`) writes the walk's per-slot logits over
   the 16 candidates into the dense `draft_logits[req, slot, vocab]` buffer. The rest of the buffer
   stays at `-inf`, in float32 (`draft_logits_spec` 709-713). The rejection sampler uses this as
   q(x) (§3).
6. `_maybe_predict_acceptance` (769). This feeds the acceptance estimator used by `DYNAMIC_K`
   and the trim paths.
7. `_apply_lookup` (793-865) fuses the chain with the lookup (§2.4).

The selector exists because block-parallel slots are drafted independently. Without the edge score,
slot i+1 doesn't know which token slot i picked. The transition score is the cheap fix, and stays
O(K²) per slot.

### 2.4 Lookup (LABD) fusion and the long block

Kernels are in `vllm/v1/worker/gpu/spec_decode/dflash2/lookup.py`:

- **`suffix_lookup`** (38-106) finds the longest earlier occurrence of the current suffix anywhere
  in the request's token history. It scans in `BLOCK=1024` chunks, so it is **O(context)** per
  request per step. A match may overlap the suffix it matched (`docs/optimizations.md:306-307`).
- **`fuse_draft`** (109-193) merges the lookup into the drafter's chain:
  - A match of ≥ `NSTRONG` tokens is taken on its own.
  - A shorter one (≥ `NMIN`) is taken only if the drafter agrees on the first `AGREE` tokens.
  - **Defaults in code: NMIN=6, NSTRONG=6, AGREE=0** (`vllm/envs.py:215-218,1623-1626`). That
    means "take any match ≥6". The code comment at `dflash2/speculator.py:286-288` gives the
    reason: "measured 3.33 vs 3.27 at C1". `docs/optimizations.md:303-304` still says NSTRONG=8
    and AGREE=2. **The doc is stale** (open question 6).
- **`_point_mass_draft_logits_kernel`** (196-233) overwrites q(x) at each lookup-sourced position
  with a point mass at the lookup token. So under probabilistic verification a lookup token is
  accepted with probability p_target(token) (§3).

**Long block** (`next_num_draft_tokens`, `dflash2/speculator.py:404-457`): k is raised from 7 to
15 when the following conditions hold.

- (a) The lookup has a match long enough to fill the tail.
- (b) The step that just finished emitted at least a full short block.
- (c) Both held **two steps in a row**, because one saturated step happens inside ordinary prose
  (`docs/optimizations.md:286-293`).

Further details:

- The flags are read from a pinned host copy that is **one step stale**. Reading them synchronously
  costs 5% [M, `docs/optimizations.md:294-296`].
- The decision is **unanimous across the batch**, because there is one scheduled k per step.
- `STICKY` (keep the long block once triggered) only applies at `num_reqs==1`.
- `CHEAP_CTX` is a cheaper trigger variant gated in the same function.
- The docstring quotes "+6% per step at 1.5k against +27% at 25k" for the long block. The unit
  (step time or throughput) is not stated (open question 5).

**Chain mode** (`VLLM_DFLASH2_CHAIN`, default off, `envs.py:225`) proposes from the lookup alone
and skips the drafter forward pass when the lookup is confident (`dflash2/speculator.py:459-704`,
lookup call at 664, point mass at 689). It always forces the full block. Measured effect of
skipping the drafter on a copy run: 15.21 → 13.79 tok/step, saving about 3 ms of a 39 ms step
[M, `docs/gotchas.md:275-285`]. It does not pay off.

### 2.5 DFlash (v1) and the shared speculator plumbing

`DFlash2Speculator` extends the DFlash speculator in `dflash/speculator.py`:

- FULL_DECODE_ONLY graphs (155-179).
- `_generate_draft` (305-336).
- Graph context rows `num_reqs×(k+1)` (338-340).
- `_precompute_context_kv` (342-358).
- `propose` (366-553), which:
  - sets `last_num_emitted=num_sampled` (399);
  - combines aux hidden states (411-417);
  - dispatches the uniform-batch graph with `uniform_token_count=num_query_per_req` (487-497);
  - precomputes context KV inside the graph, or outside it for prefill rows beyond the graph
    (503-519);
  - calls `run_fullgraph` (540-542).
- The input-prep Triton kernel (556-753).
- Prewarm (812-823).

`init_speculator` (`vllm/v1/worker/gpu/spec_decode/__init__.py:8-70`) routes `method=dflash` to
LiLiCorr, DFlash2 or DFlash according to the checkpoint config.

## 3 Verification and acceptance

The target runs one forward pass over `[last token, d1..dk]` per request, a query of k+1 tokens.
`model_runner.py:1622-1627` calls the V2 `RejectionSampler` (`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:147-176`,
`__call__` 359-406) with `speculator.draft_logits`. The work happens in one Triton kernel,
`_rejection_kernel` (`rejection_sampler_utils.py:485`):

- **Greedy** (temperature 0, 572): accept d_i if `d_i == argmax p_i`. The stored output token is
  always the target argmax, so a rejection costs nothing extra.
- **Probabilistic** (661): Leviathan et al. Accept if `p(d_i) > u·q(d_i)`. On the first rejection
  at position i, resample from the residual `max(p−q, 0)` normalised (843). Here q is the drafter's
  cached candidate distribution:
  - q is zero outside the 16 candidates, so the residual there is p.
  - q is a point mass for lookup tokens.
- **One-hot draft** (855, also 187): if a proposer supplies no draft logits, q(d_i)=1. Acceptance
  is then `p(d_i) > u`. MTP runs this way unless `draft_sample_method=probabilistic` allocates
  logits (`speculator.py:177-181`).
- **Bonus** (`is_bonus` 787): when all k drafts are accepted, sample position k+1 from the target.
- **Block verification** (Sun et al., 597) and a synthetic-acceptance mode exist. Neither is used
  by our launchers.

Each step emits `1 + accepted` tokens per request. Under probabilistic verification the output
distribution equals the target's exactly. The int4 drafter costs acceptance, not correctness. Its
measured cost is about 5% acceptance under sampling: 3.2 vs 3.4 tok/step
[M, `docs/optimizations.md:228-238`]. Greedy acceptance is unchanged: 3.34-3.65 vs 3.54
[M, `drafter/README.md:100-103`].

**Measured acceptance profile:**

- Per-position survival at T=0.7 on a 3090 [M, `DI/gpu-results/spec-5.md`]:
  0.687 / 0.511 / 0.404 / 0.325 / … / 0.190 at position 7. Mean 3.611 tok/step.
  - Conditional acceptance rises from 0.69 to 0.84 along the chain.
  - Σ min(p,q) at each slot *falls* from 0.69 to 0.50.
  - Per-prompt spread is 1.58-5.55 tok/step.
- Repo `/metrics` at k=15 with the lookup [M, `docs/gotchas.md:1145-1162`]:
  - Per-position acceptance 0.958 … 0.793 over the drafter's 7, then about 0.15 for each of the
    8 lookup-only positions.
  - The lookup fills 4.3% of positions.
  - The long block is scheduled on 44% of rows.
- GSM8K accepted length 4.79-4.84 of 7 [M, `docs/long-context.md:449`].
- Greedy is **not** bit-deterministic across configs (±3%) [M, `docs/gotchas.md:178-182`]. Treat
  differences under about 3% as noise.

## 4 Scheduling

### 4.1 Batch shape and CUDA graphs

- **Verify query** = k+1 per request. Target decode graphs need an **exact uniform query length**
  (`vllm/v1/worker/gpu/cudagraph_utils.py:270-318`, capture loop 340-355; dispatch via
  `dp_utils.py:213 dispatch_cg_and_sync_dp`).
  - With `GRAPH_BOTH`, both 8 and 16 are captured (293-318), so the long block runs in a graph.
  - A ragged batch (requests at different k) falls back to piecewise execution. That costs 8%
    [M, `docs/gotchas.md:286-298`].
- Capture size `CG = min(64, MAX_SEQS×(k+1))` (`start_qwen.sh:560`). Graph memory is 1.82, 2.12 or
  2.27 GiB by config [M, `docs/gotchas.md:267-274`]. Capturing both lengths costs 27.9 vs 25.9 ms
  per step and 1.8 vs 1.45 GiB [M, `docs/gotchas.md:246-252`].
- **The drafter** runs FULL_DECODE_ONLY graphs with `uniform_token_count=num_query_per_req=8`
  (`dflash/speculator.py:155-179,487-497`). Rows beyond the graph, for example prefill context,
  run eagerly (503-519).
- **GDN** advertises `UNIFORM_BATCH` graph support only (`vllm/v1/attention/backends/gdn_attn.py:86`).

### 4.2 GDN / conv state under speculation (hybrid target)

- **Allocation.** Each GDN layer gets a `spec_state_indices` buffer with **k+1 columns** per request
  (`gdn_attn.py:140-141`), filled from `block_table[:, :num_spec+1]` (357-358). In mamba-align mode
  the cache manager reserves about `(2+k)` pages per group per request. The DFlash2 resident cost is
  0.88 GiB at k=7 vs 1.66 GiB at k=15, a 9/17 ratio, and 15.8% of the `CTX=fast` pool
  [M, `docs/gotchas.md:317-336`].
- **Write.** In the verify forward pass, `fused_recurrent` runs with `INPLACE_FINAL_STATE` and the
  spec indices. **Each of the k+1 positions writes its own state** into its own slot
  (`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:169-171`). Called from
  `qwen_gdn_linear_attn.py:1446-1463`.
- **Read.** The next step starts from slot `num_accepted−1` (`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:106`
  `i_t = num_accepted−1`). Rejected positions are never rolled back. They are simply not read.
- **Conv state.** This is the same pattern for `causal_conv1d_update` with `num_accepted_tokens`
  (`qwen_gdn_linear_attn.py:1335-1346`). The bounds fix from patch pr50021 is at
  `causal_conv1d.py:875-879`.
- **Prefix caching and align.** `mamba_hybrid.preprocess_state` (`vllm/v1/worker/gpu/mamba_hybrid.py:179-213`,
  called at `model_runner.py:1794`) copies or aligns state blocks before the step.
- **Consequence.** GDN cannot shorten a query after the CPU has fixed its length:
  `supports_device_cpu_query_lens_mismatch` returns `not is_ssm` (`backend.py:228`). That rules
  out device-side trimming of the verify length for a hybrid target, which is DeepInfer spec-1 /
  runtime-1.

### 4.3 Choosing k per step

- **Static.** `num_speculative_tokens` is fixed at launch.
- **DYNAMIC_K.** The scheduler looks up k by batch size (`vllm/v1/core/sched/scheduler.py:292-298,1446-1450`;
  table in `vllm/v1/spec_decode/dynamic/utils.py`). Graph `decode_query_lens` gets one entry per
  distinct k (`cudagraph_utils.py:270-290`).
- **DFlash2 long block.** `next_num_draft_tokens` produces a per-step k. The model runner passes it
  to the scheduler at `model_runner.py:2242-2257`. vLLM only does this on the **synchronous**
  scheduling path, hence `--no-async-scheduling` when k>7 (cost under 1% at C1
  [M, `docs/optimizations.md:296-299`]).

### 4.4 Pipeline per decode step

The step at batch B, k drafts:

```
scheduler (sync, k for this step) ─► target forward on B×(k+1) tokens
   ├─ 48 GDN layers: spec conv + recurrent, k+1 state writes each
   ├─ 16 attention layers: spec-decode attention (split-KV patch) over ctx + k+1
   └─ lm_head on B×(k+1) rows; aux hidden states from 5 layers kept
─► rejection sampler (one Triton kernel) ─► num_accepted, output tokens
─► speculator.propose:
     precompute drafter context KV for the new rows (one fused GEMM, 5 layers)
     drafter forward pass (8 rows/request, FULL graph)
     lm_head top-16 on 7 rows/request ─► edge scores ─► selector walk
     suffix lookup (O(ctx)) ─► fuse ─► point-mass draft logits
     next_num_draft_tokens (pinned flags, one step stale)
```

## 5 MTP and the other paths compared

| path | drafter | drafts/step | q(x) for verification | own KV? | graphs | status here |
|---|---|---|---|---|---|---|
| **DFlash2** (`dflash2/`) | 5-layer block-parallel, int4, 1.19 GB, plus the shared 0.65 GB lm_head | 7 (15 with lookup) in **1** forward pass | cached top-16 candidate logits; point mass for lookup | yes, sliding window 2048 | FULL, uniform 8 | `SPEC=dflash2` |
| **MTP** (`mtp/speculator.py:12`, extends `AutoRegressiveSpeculator`) | Qwen's 1-layer MTP head, GPTQ int4, 40k draft vocab (97.5% coverage [M, `docs/optimizations.md:21,87`]) | 4 (fast) / 3 (long, huge) in **k** sequential passes (`autoregressive/speculator.py:501-551`, loop at 515) | one-hot unless probabilistic | yes, a full cache layer | captured per step | `SPEC=mtp` |
| DSpark (`dspark/`) | block-parallel variant | – | – | yes | – | measured worse on a 3090: 3.37 vs 3.83 tok/step, 114 vs 150 tok/s [M, `docs/long-context.md:619-632`] |
| EAGLE / EAGLE3 (`eagle/`) | 1-layer AR head on aux hidden states | k sequential passes | one-hot or probabilistic | yes | yes | in tree, no checkpoint for this target |
| draft_model, multi_module_mtp, LiLiCorr, Gemma4, extract_hidden_states | various | – | – | – | – | in tree, not wired |
| HEAD only: ngram, standalone_ar, target_dependent_ar; `DFlash2BeamDraftModel` | – | – | – | – | – | "Beam" is a target-model family (`HEAD:vllm/model_executor/models/beam/nvidia/dflash2.py`), **not beam search** |

**Head-to-head at CTX=fast, C1** [M, `docs/optimizations.md:239-241,384-398`]:

| | ms/step | tok/step | tok/s |
|---|---|---|---|
| MTP | 24.8 | 2.8-2.9 | 111-115 (greedy 115-124) |
| DFlash2 | 26.5 | 3.14-3.34 | 117.8 / 125.7 |
| DFlash2 + lookup | – | – | 130 |
| DFlash2 + lookup + long block | – | – | 133 |

The baseline without speculation is 46 tok/s. MTP's position-0 acceptance is 74%/77%, and it sits
at its chain ceiling with top-1 0.685 [M, `drafter/README.md:29-37`].

**Where MTP wins:**

- **Long context.** At 112k the all-six average is 78.6 vs 47.0 tok/s, and QA is 72.9 vs 34.6
  [M, `docs/long-context.md:199-207`]. The causes are the drafter's 2048-token window, which costs
  acceptance at 12-36k (2.3-2.6 vs 2.6-3.0 tok/step [M, `docs/optimizations.md:353-356`]), and a
  longer verify query at the same attention cost.
- **Concurrency.** MTP has 8 residents vs 6. Its aggregate at C8 is 383 tok/s, against DFlash2's
  309 [M, `docs/optimizations.md:349-366`]. [review: 309 is DFlash2's C4 decode aggregate. At C8
  DFlash2 holds 5 residents and has no steady state (`single-user/README.md:112-117`).]

**Where DFlash2 wins:** copying from context. At 112k the reproduce task gives 154.8 vs 101.4 tok/s
[M, `docs/long-context.md:201`], and at 25k it reaches 381 tok/s at 14.97 tok/step
[M, `docs/optimizations.md:317`].

## 6 Per-step cost model

### 6.1 Model

For batch B at context L with verify length q = k+1:

```
t_step(B, L, q) ≈ T_gemm(B·q)                    # Marlin W4A16 weight streaming, ~flat for B·q ≤ 16
               + T_attn(L, B·q)                    # 16 full-attn layers, spec-decode split-KV
               + T_gdn(B·q)                        # 48 GDN layers incl. (k+1) state writes
               + T_lmhead + T_norm/other
               + T_draft(B)                        # drafter fwd + ctx-KV GEMM + lm_head top-16 + walk + lookup
               + T_host                            # scheduler, sync, launch gaps
tok/s ≈ B · E[1 + accepted] / t_step
```

**Measured components** at about 170k, profile E, WSL2 [M, `DI/gpu-results/e-profile.md`]:

| component | ms/step |
|---|---|
| total step | 150.8 |
| full attention (KVarN stage 1 at 7.3 ms/layer, about 40× its DRAM floor) | 117.7 (78%) |
| Marlin GEMMs | 17.7 |
| drafter + lookup | 3.6 |
| norms | 2.2 |
| GDN | 1.3 |
| lm_head | 0.8 |
| other | 1.7 |
| host gap | 5.7 |

That run gave 1.86 tok/step and 11.5 tok/s.

**Checks against measured end-to-end numbers:**

- **[E] Short context.** 3.34 tok/step ÷ 26.5 ms = 126 tok/s, against 125.7 measured
  [M, `docs/optimizations.md:239-241`]. 3.14 ÷ 26.5 = 118.5, against 117.8 measured. The model
  closes at C1 with no hidden term. [review: this check is circular. 26.5 ms is the measured step,
  so tok/step ÷ 26.5 ms has to reproduce the measured tok/s. It does not test the decomposition.]
- **[E] Speculation overhead.** Without speculation the rate is 46 tok/s
  [M, `docs/optimizations.md:384-398`], which is a 21.7 ms one-token step. DFlash2's step is 26.5 ms.
  So drafting plus k=7 verification costs about **4.8 ms (18%)** on top of the non-speculative
  forward pass. MTP-4's step is 24.8 ms, so its overhead is about **3.1 ms**. DFlash2 needs
  3.34/2.85 ≈ 1.17× MTP's accepted length to break even with a step 1.07× longer. It has about
  10% headroom. [review: the 46 tok/s row is the first row of a cumulative table
  (`docs/optimizations.md:387`). It predates fp16 state, the GPTQ-int4 lm_head and the sampler and
  split-KV patches, so 21.7 ms is not the fast-variant no-spec step. Recomputed from e-profile
  components, the overhead is about 4.5-6.5 ms [E], so 4.8 ms is plausible but unsupported as a
  measurement.]
- **[E] Drafter's share.** Skipping the drafter saves about 3 ms
  [M, `docs/gotchas.md:275-285`]. That agrees with the e-profile's 3.6 ms for drafter + lookup. A
  bandwidth floor: 1.19 GB of drafter weights plus 0.65 GB of lm_head ≈ 1.84 GB at about 936 GB/s
  is **≈2.0 ms**. DeepInfer's "draft cost check a" (`DI/drafter.md`) says ≥1.76 ms. So the drafter
  runs at roughly 55-65% of its bandwidth floor, and about half its cost is the lm_head re-read
  needed for the candidates. [review: the "1.19 GB" is GiB (1.281 GB). About 0.254 GB of it is the
  selector's two bf16 248320×256 codebooks, which are gathered for 16 rows, not streamed
  (`qwen3_dflash2.py:306,314`). Streamed bytes are 1.02 GB of layers + fc plus 0.656 GB of lm_head,
  1.68 GB in all, a floor of ≈1.80 ms at 936 GB/s. That matches DI's ≥1.76 ms. The lm_head is ≈39%
  of the bytes, not half (`00-meta/reviews/phase1-dflash-specdec-weights.out`).]
- **[M] Stairs in verify length.** Marlin uses an M tile of 16 rows. Spec attention uses a 128-row
  tile, which holds 21 query tokens at G=6. Steps were 39.5 ms at 16 queries, 47.8 at 19 and 47.2
  at 21. Attention per layer was 250, 583 and 1132 µs [`docs/gotchas.md:253-266`]. [review: those
  per-layer times are at 8/16/32 query tokens, not 16/19/21 (`docs/gotchas.md:262`). The
  39.5 → 47.8 ms jump is the Marlin stair (16 → 17 rows). The attention stair at 22 was never
  measured as a step time (`docs/gotchas.md:264-266`).] **B·(k+1) = 16
  is the last cheap point at short context.** B=2 at k=7 sits exactly on it.
- **Long block at 25k:**
  - [M] Each extra verify position costs about 1 ms of attention [`docs/optimizations.md:288`].
  - [E] Derived steps from the 25k table [`docs/optimizations.md:317-322`]: reproduce at k=15 is
    14.97/381 → **39.3 ms**; at k=7 it is 7.83/260 → 30.1 ms; with no lookup it is 4.72/159 →
    29.7 ms. C1 chat is 25.6-25.7 ms in all three columns, so the long block rarely triggers on
    chat.
  - So 8 extra positions cost about 9 ms at 25k, close to the 1 ms/position figure.
- **[E] Long context, 112k** (`docs/long-context.md:199-207`):
  - DFlash2 k=15 reproduce: 14.19/154.8 → **91.7 ms** per step.
  - DFlash2 QA: 2.17/34.6 → 62.7 ms.
  - MTP-3 QA: 2.75/72.9 → 37.7 ms.
  - At 112k a DFlash2 step costs 1.7-2.4× an MTP step. Its acceptance only pays this back on copy
    tasks.
  - `docs/long-context.md:214-218` attributes the 91.7 ms (vs bf16 48.2 ms at 50k) to the Triton
    backend, the drafter's 5 layers, and the int8 kernel's padded stride (about 13%).
- **[M] TP=2 profile on a 3080, per rank** (`docs/long-context.md:534-545`):
  - Drafter attention: 0.878 → 0.098 ms after the fix.
  - Target attention: 0.746 ms at 8k, 5.631 ms at 128k.
  - Everything else: 21.47-22.22 ms.
  - Step: 23.49 / 22.73 / 29.03 / 28.40 ms.
  - So the non-attention floor is about 22 ms, and attention is the only term that grows with
    context.
- **[M] Concurrency.** Each resident adds about 7 ms (25.9 → 49.1 ms from 1 to 4 streams). MTP adds
  about 5 ms [`docs/optimizations.md:358-366`]. Per stream: 137 / 97 / 46 tok/s at 1/2/4. Aggregate
  137 → 309. [review: the table is `single-user/README.md:112-117` (`bench/conc_ladder.py`, ms/pass
  over the decode-only window, 4k-token prompts). Its steps are 25.9 / 32.8 / 49.1 ms: +6.9 ms for the
  2nd resident and +8.15 ms each for the 3rd and 4th, a mean of 7.7. At C4, B·q = 32 crosses the
  16-row Marlin stair, so part of the 2 → 4 growth is that stair, not residency. Raw data is not in
  the repo.]
- **[E] Recurrent-state write traffic.** One GDN state is 1.57 MB [M, DeepInfer spec-3 sizing in
  `DI/spec-decode.md`]. 8 positions × 48 layers × 1.57 MB ≈ **603 MB/step** of writes, about
  0.64 ms at 936 GB/s per resident. That is about 9% of the 7 ms per-resident cost. The rest of
  the 7 ms is unattributed (open question 3).

### 6.2 Marginal value of one more verify position

[E] Adding position i is worth `s_i · t_step / E[tokens]` of step time, where s_i is the survival
probability. At T=0.7, s_7 = 0.19 and E = 3.61 [M, spec-5]. On a 26.5 ms step the break-even
marginal cost is 0.19 × 26.5 / 3.61 ≈ **1.4 ms**.

- At short context a position costs well under 1 ms while B·q ≤ 16, so k=7 is justified at B=1.
- At 25k a position costs about 1 ms, so positions 8-15 are only worth it when s_i ≫ 0.15. That is
  the copy case, and it is exactly what the adaptive trigger gates.
- At 112k+ every position's attention cost grows, and the break-even moves to k≈3-4. DeepInfer
  spec-0, "lower fixed k at E", targets this. DFlash2 cannot draft a shorter block from the same
  forward pass for free: k=7 is its trained block, and fewer drafts only shortens the target query.

## 7 Likely bottlenecks, ranked

Each item gives the evidence, then what would confirm it: what to log and which tool to use.

1. **Accepted length at short and mid context.** The step has a fixed weight-streaming floor of
   about 22 ms, so tok/s scales almost linearly with E[1+accepted].
   - Contributing losses:
     - The int4 drafter: −5% under sampling [M].
     - The 2048 window: about −12% at 12-36k [M, 2.3-2.6 vs 2.6-3.0].
     - Σ min(p,q) decays to 0.50 by slot 7 [M, spec-5].
   - Any drafter or verification change that raises tokens/step 10% buys about 10% throughput.
   - *Confirm:* per-position acceptance from vLLM `/metrics`
     (`vllm:spec_decode_num_accepted_tokens_per_pos`) and per-prompt tok/step with
     `bench/labd_accept.py` and DeepInfer's `bench/spec5_accept.py`, split by context bucket.
     Compare the bf16 drafter and full-window ablations.
   - Overlaps DeepInfer drafter-3 (TV-loss repair of int4) and drafter-4 (fine-tune).
2. **Verify attention at long context.** This is 78% of the step at about 170k, with KVarN stage 1
   at about 40× its DRAM floor [M, e-profile]. It is 5.6 ms per rank at 128k on TP=2 [M]. Every
   verify position multiplies it.
   - It also carries a **correctness hazard**: with prefix caching, the adaptive long block on KVarN
     gives wrong tokens (38 of 793 self-hits wrong [review (phase 5): misread. `start_qwen.sh:399-411` says turn 2 tracks the source for about 38 characters and then diverges; "38/793" is not a count of wrong self-hits] at 12.71 tok/step, against clean output at
     14.71 with ADAPTIVE=0) [M, `start_qwen.sh:399-421`]. That is why ADAPTIVE=0 is pinned on huge.
   - *Confirm:* the torch profiler (`--profiler-config`) or DeepInfer `bench/trace_breakdown.py` at
     8k/32k/64k/128k × q∈{4,8,16}, giving ms per attention layer. Use `bench/spec_attn_ctx_scan.py`
     for the offline kernel curve.
   - Overlaps attn-2 (KVarN verify kernel), runtime-2 (`bench/step_curve.py`, not yet written) and
     spec-0.
3. **Recurrent-state residency and per-resident cost.** Each resident holds 1+k state slots in
   every GDN layer: 0.88 GiB, 15.8% of the pool, 6 residents vs MTP's 8 [M]. Each resident adds
   about 7 ms [M]. Only about 0.6 ms of that is explained by state writes [E].
   - *Confirm:* the profiler at B=1/2/4, split per kernel class (GDN recurrent, conv, attention,
     drafter, Marlin), to decompose the 7 ms. Use `bench/conc_ladder.py` for the throughput side.
     Log `num_running` and preemptions from `/metrics`.
   - Overlaps spec-3 (RecoverSSM: about −0.45 ms per resident, residents from about 6 to 20+) and
     spec-1 / runtime-1.
4. **Fixed draft cost of about 3-3.6 ms per step (about 11-14%).** About half of it is the full
   248k-vocab lm_head GEMM on 7 rows that produces top-16 candidates [E]. [review: by bytes it is
   ≈39% (0.656 of 1.68 GB streamed), see §6.] The rest is the drafter
   forward pass, the context-KV GEMM and the walk.
   - *Confirm:* profiler ranges around `compute_candidates`, `_run_model` and
     `precompute_and_store_context_kv`. Check whether e-profile's "lm_head 0.8 ms" includes the
     drafter's lm_head call (open question 4).
   - A restricted-vocabulary candidate head (MTP's 40k vocab covers 97.5%) would cut most of the
     lm_head term.
   - Overlaps evals-8 (per-component profiling) and drafter-1.
5. **Graph-dispatch exactness and stairs.** A ragged batch costs 8% piecewise [M]. Capturing both
   lengths costs +2 ms and +0.35 GiB [M]. Crossing B·q = 16 (Marlin) or 21 (attention tile) gives
   a step jump of about 8 ms [M, `docs/gotchas.md:253-266`]. [review: the measured ≈8 ms jump at
   25k is the Marlin stair only. The attention-tile stair has per-layer kernel times but no
   measured step jump.]
   - Tokens per graph replay are batch-unanimous, so one non-copy request blocks the long block for
     everyone.
   - *Confirm:* log the graph-dispatch mode per step (FULL / PIECEWISE / eager) and B·q, with a
     step-time histogram. Run `bench/conc_ladder.py` at C2-C4.
6. **Synchronous scheduling and the host gap.** `--no-async-scheduling` costs under 1% at C1 [M].
   The host gap is 5.7 ms at 170k [M, e-profile], about 4% of that step but likely a larger share
   at short context and higher B.
   - *Confirm:* an nsys or torch-profiler timeline of the GPU idle gap between steps at 1.5k and 25k,
     async vs sync.
   - Overlaps spec-2 (fresh-confidence device trim, which would remove the stale flag).
7. **O(context) lookup scan.** `suffix_lookup` scans the whole history per request per step in
   1024-token chunks. It is bounded above by the measured 3.6 ms "drafter + lookup" at 170k [M]. It
   is likely under 1 ms [E: 170k tokens × 4 B = 0.7 MB per scan].
   - *Confirm:* `bench/test_lookup_kernels.py` timing at 8k-200k, and a profiler range around
     `_apply_lookup`.
   - Overlaps spec-4.

## 8 Hooks for an alternative drafter (e.g. EAGLE-style with its own cache, or a tree)

This section covers only the integration surface; the EAGLE-4 analysis is in another lane. The
line numbers are for the HEAD tree, with the patched-0.31 equivalents in brackets.

| hook | HEAD file:line (0.31-patched) | what a new drafter must provide |
|---|---|---|
| registry | `HEAD:vllm/v1/worker/gpu/spec_decode/__init__.py:13` `init_speculator(vllm_config, device, req_states)`; DFlash2 dispatch 34-41 (`__init__.py:8-70`) | a `method` branch that constructs the speculator |
| speculator API | `HEAD:.../spec_decode/speculator.py:61` `BaseSpeculator`, `:106` `DraftModelSpeculator` (59-101, 103) | `num_query_per_req`, `num_extra_query_per_req`, `propose(...)`, `init_cudagraph_manager`, `capture` |
| draft distribution | `HEAD:.../speculator.py:181` probabilistic allocation (177-181; `draft_logits_spec` 376-381, `sample_draft` 402-436) | dense `draft_logits[req, slot, vocab]`, or one-hot. Vocab must match the target's 248k, or a mapping is needed. |
| aux hidden states | `HEAD:vllm/v1/worker/gpu/model_runner.py:413`; `HEAD:.../spec_decode/eagle/eagle3_utils.py:25,110` (`model_runner.py:391-403`) | the list of target layers to tap |
| own KV cache | `HEAD:model_runner.py:617,675-689` `draft_attn_layer_names` (579, 637) | attention layer names, so the KV-cache manager allocates draft groups. DFlash2's sliding-window group shows how a non-standard cache plugs in. |
| eagle model load | `HEAD:.../spec_decode/eagle/utils.py:80` `load_eagle_model`; `eagle/speculator.py:12` `EagleSpeculator(TargetDependentARSpeculator)` | reusing the AR loop: `autoregressive/speculator.py` `_prefill` 453, `_multi_step_decode` 501 |
| propose call site | `HEAD:model_runner.py:954,2261` (2221) | |
| k feedback | (`model_runner.py:2242-2257`, `next_num_draft_tokens`) | sync scheduling only |
| verification | `HEAD:.../rejection_sampler.py:416` `__call__`; HEAD runner call 1660 (1622-1627; kernel `rejection_sampler_utils.py:485-998`) | **chain only.** A tree needs a new acceptance kernel (longest accepted root-to-leaf path) and token compaction. |
| graphs | `HEAD:vllm/v1/worker/gpu/cudagraph_utils.py:300` `decode_query_lens` (270-318) | the uniform query length per graph. A tree with N nodes means q = N+1 per request. |
| GDN spec state | `HEAD:vllm/v1/attention/backends/gdn_attn.py:204,447,468` (140-141, 357-358); `HEAD:fused_recurrent.py:107` (106, 169-171); `HEAD:backend.py:214` (228) | **The hard part for a tree.** Today slot j holds the state after position j of a linear chain, and the reader takes slot `num_accepted−1`. A tree needs each node's state computed from its **parent's** state: a parent-index array in `fused_recurrent` and `causal_conv1d_update`, one state slot per node (memory ×N/(k+1)), and the reader selecting the accepted leaf's slot. |
| mamba prefix/align | `HEAD:vllm/v1/worker/gpu/mamba_hybrid.py:180`; runner 1833 (179-213; 1794) | page count `(2+q)` per group grows with the node count |
| target attention | the 16 full-attention layers' spec-decode path (split-KV patch, `docs/gotchas.md:161-177`) | a tree needs an ancestor mask instead of causal-within-block. Neither tree has one. |

**Summary.** A chain drafter with its own KV cache (EAGLE-style) plugs in with no target-side
change. It needs a speculator class, a registry entry, aux layers and draft attention layer names.
MTP is the template. A **tree** requires changes in four target-side places that exist in neither
vLLM version:

1. the tree mask in attention;
2. parent-indexed GDN recurrent and conv state;
3. a tree rejection sampler;
4. graph capture for the node count.

Point 2 is unique to this hybrid model, and it multiplies the residency cost in §7 item 3.

## 9 Open questions

1. What is the real ms-per-verify-position curve at 8k / 64k / 128k for q ∈ {4, 8, 16} on native
   Linux? Only the 25k "about 1 ms" point and the 170k WSL2 profile exist. Owner: runtime-2
   `bench/step_curve.py`.
2. What causes the KVarN wrong-token rate when the verify length changes between steps under
   prefix caching (`start_qwen.sh:399-421`)? Is it a KVarN verify-kernel bug or a GDN state-slot
   misalignment when k flips 7↔15?
3. How does the about 7 ms per-resident step growth decompose? State writes explain about 0.6 ms
   [E].
4. Does e-profile's "lm_head 0.8 ms" include the drafter's candidate lm_head call, or is that
   inside "drafter + lookup 3.6 ms"?
5. What unit is the "+6% at 1.5k / +27% at 25k" figure in the `next_num_draft_tokens` docstring:
   step time or throughput?
6. Which NSTRONG/AGREE values are intended? The doc says 8/2 and the code says 6/0, which the code
   comment justifies with "3.33 vs 3.27 at C1". The doc should probably be updated.
7. Does a tree drafter on this target need per-node GDN state slots, or can a cheaper
   recompute-from-parent work (cf. spec-3 RecoverSSM)? This decides whether trees are affordable
   at all with 6 residents.
8. How much acceptance do the int4 quantisation and the 2048 window each cost at 32k-128k,
   separately? Only the short-context int4 number and the 12-36k combined gap are measured.

## Reviewer additions

Added by the phase-1 review (`00-meta/reviews/phase1-dflash-specdec.md`).

- **vLLM had tree attention and then deleted it.** Commit `b1728c1e66` "[Attention][Cleanup] Remove
  tree attention (#42121)" is an ancestor of v0.31.0. Use `git show b1728c1e66^` in `~/vllm` as
  the reference for the §8 ancestor mask. It never covered GDN state.
- **Lookup tail thresholds.** Tail positions (≥ draft_block) use `NMIN_TAIL=4`, not `NMIN`. The
  long-block flag also needs `LONGMIN=6` (`envs.py:219-220`; `lookup.py` `_fuse_draft_kernel`
  `take_tail`/`has_tail`).
- **The adaptive long block is off at the default `DFLASH_TOKENS=7`.** `_adaptive` needs
  `draft_block < num_speculative_steps` (`dflash2/speculator.py:317-320`). The launcher pins
  ADAPTIVE=0 only when all of these hold: LOOKUP=1, DFLASH_TOKENS>7, CTX=huge, prefix caching on
  and `LOOKUP_ADAPTIVE` unset (`start_qwen.sh:422-423`).
- **The selector edge score is gated by the hidden state.** It is `unary + Σ_r pred[r]·h[r]·succ[r]`,
  with rank 256 (`selector_rank`) and the anchor token as slot 0's predecessor
  (`qwen3_dflash2.py:306-316`).
- **The e-profile's "1.86 tok/step, 11.5 tok/s" is the unprofiled cold pass.** The profiled pass
  gave 2.01 / 12.5 (`DI/gpu-results/e-profile.md:31-32`).
