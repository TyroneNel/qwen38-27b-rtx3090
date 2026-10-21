# Review: `01-project/dflash-specdec.md` (phase 1, specdec lane)

Reviewer: Opus, effort high. Date: 2026-10-21. Desk review, no GPU.

## Verdict

**PASS WITH CORRECTIONS.**

The mechanism sections (§2-§4, §8) are accurate. Nearly every code citation lands on the right
line of the patched 0.31 tree. The cost model (§6, and the §1 summary that quotes it) has four
problems:

- It mislabels its 22 ms "weight-streaming floor".
- It checks itself against the very number it was built from.
- It takes its no-spec baseline from an older config.
- It misattributes the stair timings.

There is also one factual error about code provenance. All of these are fixed inline, each marked
`[review: …]`.

**Method.** I built the patched tree in scratch: `git archive v0.31.0 vllm`, then
`bash patches/apply.sh`. All 51 patches applied with no fuzz, and every `vllm/...` line below is in
that tree. Weight bytes come from safetensors headers fetched over HTTP by another lane
(`scratchpad/hf/headers.json`, metadata only). The script is
`phase1-dflash-specdec-weights.py` and its output is `.out`, next to this file.

## Checked claims

All `vllm/...` paths are in the patched v0.31.0 tree, and `DI/` = `~/research-src/deepinfer-ref/docs/research/deepinfer/`.

| # | Claim | Doc § | Evidence | Status | Note |
|---|---|---|---|---|---|
| 1 | DFlash2 has 5 non-causal layers, taps 5/19/33/47/61, a sliding window of 2048 and its own KV | 2.2 | HF `syvai/Qwen3.8-27B-DFlash2-W4A16` config: `num_hidden_layers 5`, `is_causal false`, `target_layer_ids [5,19,33,47,61]`, `sliding_window 2048`. Code: `qwen3_dflash.py:86-97` | CORRECT | |
| 2 | One forward pass over the bonus token plus 7 mask tokens. With LOOKUP and k=15 the drafter still drafts 7 | 2.2, 2.1 | `dflash/speculator.py:67-91` (draft_block = block_size−1 = 7 when lookup is on and k > 7; `num_query_per_req = 1+draft_block`); config `block_size 8` | CORRECT | Without LOOKUP, draft_block = k, so the drafter is asked for all 15 |
| 3 | Context KV comes from one fused GEMM for all 5 layers | 2.2 | `qwen3_dflash.py:561-602` `_build_fused_kv_buffers` ("one fused GEMM for all layers at once"), `:665-740` | CORRECT | |
| 4 | Top-16 candidates come from the target's shared int4 lm_head (248k vocab) | 2.3 | `qwen3_dflash2.py:415-420` `get_top_k_tokens(self.lm_head, …, top_k)`; `dflash/utils.py:62-73` replaces the drafter lm_head with the target's; config `selector_top_k 16`, `vocab_size 248320`; the fast-variant `lm_head.weight_packed` is int4 (headers) | CORRECT | |
| 5 | A 16×16 low-rank edge score, then a Gumbel-argmax Markov walk that picks one chain | 2.3 | `qwen3_dflash2.py:297-317` (`einsum blpr,blcr->blpc`, rank 256); `dflash2/speculator.py:57-110` (row = `previous`, `gumbel_noised_argmax`, `previous = index`) | CORRECT | Omits that the score is gated by the hidden state (`predecessors * hidden`, :315-316). Added under Reviewer additions |
| 6 | Lookup fusion. Defaults NMIN=6, NSTRONG=6, AGREE=0; `optimizations.md` says 8/2 | 2.4, OQ6 | `vllm/envs.py:215-218` and `:1623-1626` (added by `patches/speed-knobs-envs.patch:39-40,78-79`); `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:286-290` (comment "3.33 vs 3.27", added by `patches/dflash2-lookup-drafting.patch:873`); `docs/optimizations.md:302-304` says "(8)" and "(2)". No launcher sets either env (`grep` of `*.sh`, compose, `.env.example`) | CORRECT | The doc is stale and the code defaults are what ships. Omits NMIN_TAIL=4 and LONGMIN=6 (added) |
| 7 | The adaptive long block raises k from 7 to 15 after two qualifying steps in a row, reading pinned flags one step stale, unanimous across the batch, STICKY only at 1 request | 2.4 | `dflash2/speculator.py:404-457` (439-457) | CORRECT | Inert at the default DFLASH_TOKENS=7 (`:317-320`). Added |
| 8 | Verification is chain-only, with no tree in 0.31 or HEAD | 1, 3, 8 | No tree mask or tree sampler in `vllm/v1/{spec_decode,worker/gpu/spec_decode,attention/backends}` in either tree. `~/vllm` `b1728c1e66` "Remove tree attention (#42121)" is an ancestor of v0.31.0 | CORRECT | Missed: tree attention existed upstream and was removed before 0.31 |
| 9 | GDN writes k+1 state slots and reads slot num_accepted−1 | 4.2 | `gdn_attn.py:140-141` (`num_spec + 1` columns), `:357-358`; `fused_recurrent.py:106` `i_t = num_accepted − 1`; `:170-171` `INPLACE_FINAL_STATE` stores per token at `ssm_state_indices[i_n, i_t]` | CORRECT | |
| 10 | GDN cannot take a device/CPU query-length mismatch | 4.2 | `vllm/v1/attention/backend.py:216-228` `return not cls.is_ssm()` | CORRECT | |
| 11 | Rejection kernel line anchors (greedy 572, block verification 597, ratio test 661, bonus 787, residual 843, one-hot 855) | 3 | `rejection_sampler_utils.py` at those lines; `model_runner.py:1622-1627` | CORRECT | The launcher also sets `draft_sample_method=probabilistic` for MTP (`start_qwen.sh:666`), so production MTP is not one-hot |
| 12 | "≈7 ms per resident [M]" | 1, 6, 7.3 | `single-user/README.md:112-121` (the table) and `docs/optimizations.md:357-362`. Measured with `bench/conc_ladder.py`, ms/pass over the decode-only window (`bench/conc_ladder.py:32-35`), 4k-token prompts | CORRECT (secondary source) | 25.9/32.8/49.1 ms gives +6.9, then +8.15 per resident; the mean is 7.7. C4 runs at B·q=32, past the 16-row Marlin stair, so "residency" is confounded. Raw logs are not in the repo |
| 13 | Weight-streaming floor ≈22 ms | 1, 6, 7.1 | Derived from the 2×RTX 3080 TP=2 "everything else" row (`docs/long-context.md:505,539`) | WRONG | See the cost model. The 3090 byte floor is 14.2 ms, and 22 ms is all non-attention, non-drafter work |
| 14 | 3.34/26.5 ms reproduces 126 tok/s, so "the model closes with no hidden term" | 6.1 | `docs/optimizations.md:239-241` | WRONG (as a check) | The arithmetic is right but circular, because 26.5 ms is the measured step |
| 15 | Speculation overhead 4.8 ms = 26.5 − 21.7 (46 tok/s) | 1, 6.1 | `docs/optimizations.md:387` is the first row of a cumulative table, before fp16 state, the GPTQ-int4 lm_head and the sampler and split-KV patches | UNSUPPORTED | Recomputed below at 4.5-6.5 ms [E] |
| 16 | Drafter floor: 1.19 + 0.65 = 1.84 GB, ≈2.0 ms; lm_head is "about half" | 6.1, 7.4 | Headers: drafter 1.281 GB (= 1.19 GiB). The 0.254 GB of bf16 codebooks are gathered, not streamed (`qwen3_dflash2.py:306,314`) | WRONG (minor) | 1.68 GB streamed, ≈1.80 ms (matches `DI/drafter.md:35,196` ≥1.76); lm_head ≈39% |
| 17 | Stairs: 39.5/47.8/47.2 ms at 16/19/21, attention 250/583/1132 µs at those lengths | 6.1, 7.5 | `docs/gotchas.md:253-266` gives the µs at **8/16/32** query tokens; the jump is the Marlin stair | WRONG | Fixed inline |
| 18 | "Aggregate at C8 383 vs DFlash2's 309" | 5 | `single-user/README.md:115-117`: 309 is DFlash2 at **C4**; at C8 it has no steady state | WRONG (minor) | Fixed inline |
| 19 | "Most DFlash2 code is not upstream; our patches add it" | citations | `git ls-tree v0.31.0` contains `qwen3_dflash2.py` and `dflash2/speculator.py` (265 lines; 865 after patches); the patches add `lookup.py` and the lookup, chain and adaptive code | WRONG | Fixed inline |
| 20 | Chain mode costs 15.21 → 13.79 tok/step and saves ~3 ms of 39 | 2.4 | `docs/gotchas.md:275-285` | CORRECT | |
| 21 | Per-position survival 0.687…0.190, mean 3.611 | 3, 6.2 | `DI/gpu-results/spec-5.md:33,40-44` | CORRECT | "Position 7" means 1-indexed (spec-5 calls it pos 6) |
| 22 | e-profile components (150.8 ms, attention 117.7, Marlin 17.7, …) | 6.1 | `DI/gpu-results/e-profile.md:34-47`; the components plus the host gap sum to 150.7 | CORRECT | "1.86 tok/step, 11.5 tok/s" is the unprofiled cold pass (`:31`); the profiled pass gave 2.01/12.5 |
| 23 | 0.88 GiB per resident at k=7, 1.66 GiB at k=15, ratio 9/17 | 4.2 | `docs/gotchas.md:317-336` | CORRECT | |
| 24 | GDN state 1.57 MB, ≈603 MB/step of writes, ≈0.64 ms | 6.1 | Target config: 48 value heads × 128 × 128 × fp16 (`--mamba-ssm-cache-dtype float16`, `start_qwen.sh:1036`) = 1,572,864 B | CORRECT | Adding the read of 1 slot/layer (75 MB) gives ≈0.73 ms per resident |

**24 claims checked: 16 correct, 1 correct from a secondary source, 5 wrong (2 of them minor), 1
unsupported and 1 circular.**

## Recomputed cost model (C1, short context, k=7, B=1)

Bytes come from the safetensors headers (`phase1-dflash-specdec-weights.out`). The effective target
is the base AutoRound shards 1-6 plus the fast variant's shard 7 (GPTQ-int4 lm_head) and its extra
tensors.

| Term | Basis | ms | Label |
|---|---|---|---|
| Target weights streamed per step | 0.865 attention + 2.906 GDN + 8.824 MLP + 0.656 int4 lm_head = **13.25 GB** (excludes embed, vision, MTP) | **14.2** at 936 GB/s | E (byte floor) |
| Same, as executed | e-profile Marlin 17.7 + lm_head 0.8 = 18.5 on WSL2 (≈76% of peak). Native, given WSL2 runs 6-14% slower (`DI/gpu-results/baseline.md`) | 16.3-17.4 | M (WSL2) → E (native) |
| Norms, GDN, other | e-profile 2.2 + 1.3 + 1.7 = 5.2 on WSL2 | 4.6-4.9 | M → E |
| Drafter (forward, context KV, candidate lm_head, walk, lookup) | e-profile 3.6; drafter-skip saves ≈3 | 3.0-3.6 | M |
| Drafter byte floor | 1.02 GB layers + fc + 0.656 GB lm_head = 1.68 GB | 1.80 | E |
| Target attention at short context | 0.75 ms per rank at 8k on TP=2 3080 (`docs/long-context.md:538`) | ≈0.5-1 | E |
| GDN state traffic (8 writes + 1 read × 48 layers) | 0.68 GB | 0.73 | E |
| **Sum excluding host gap** | | **24.4-26.9** | E |
| Measured step | `docs/optimizations.md:239-241` | 26.5 | M |

**Reading.** The decomposition closes with 0-2 ms left for the host gap. The doc's two-term split
should be read as follows:

- About 17 ms of weight streaming, running at about 85% of the 14.2 ms byte floor.
- About 5 ms of norms, GDN and elementwise work.
- About 3-3.6 ms of drafter.
- About 1 ms of attention at short context.

The "22 ms floor" is roughly the first two lines, not weight streaming. The doc's conclusion that
accepted length is the lever at short context still holds, because about 21 ms of the step does
not depend on k.

**Speculation overhead.** A fast-variant no-spec step reads the same 13.25 GB (≈16.3-17.4 ms
native) plus ≤4.6 ms of q=1 elementwise work plus small attention, so ≈20-22 ms [E]. That makes the
DFlash2 overhead 26.5 − (20…22) ≈ **4.5-6.5 ms** [E]. The doc's 4.8 ms falls inside that range, but
the 21.7 ms it subtracts is from an older config. Queue item: one `SPEC=off` run of the fast
variant at C1 settles it.

**Per resident.** The measured +6.9 ms for the 2nd stream (B·q=16, on the bottom Marlin stair) is
the cleanest residency number. State traffic explains ≈0.73 ms of it, and the second request's
attention at 4k is well under 1 ms [E], so about 5 ms is still unexplained. Open question 3 stands.
Profile B=1 vs B=2 at q=8, not B=4, because B=4 mixes in the Marlin stair.

## Missed items

1. **Tree attention existed in vLLM.** `~/vllm` commit `b1728c1e66` (#42121) removed a TreeAttention
   backend before v0.31.0. §8 says neither tree has an ancestor mask. That is true at the tips,
   but the history holds a reference implementation for item 1 of the §8 tree list. It did not
   handle GDN state.
2. **Provenance.** DFlash2 (model, selector and speculator) is upstream v0.31.0 code. Only the
   lookup, chain, adaptive and prewarm layers are ours. This matters for porting to HEAD: the
   patches ride on upstream code that may move.
3. **The selector is more than a pure bigram.** The edge score is gated by a 256-dim projection of
   each slot's hidden state, and slot 0's predecessor is the anchor token. Its bf16 codebooks
   (2 × 127 MB) are VRAM, but they are not bandwidth.
4. **NMIN_TAIL=4 and LONGMIN=6** govern the tail and the long-block flag. NMIN/NSTRONG/AGREE apply
   only to the head.
5. **At the default DFLASH_TOKENS=7 the adaptive path and the long block are dead code.** So the
   KVarN prefix-cache corruption (§7 item 2) only bites `DFLASH_TOKENS>7` deployments.
6. **MTP also runs probabilistic verification in production** (`start_qwen.sh:666`), so the §5
   "one-hot unless probabilistic" cell is the non-default case.
7. **The C4 per-resident cost is confounded by the Marlin 16-row stair** (B·q=32). The doc's
   "Confirm" step for §7 item 3 should hold B·q ≤ 16.
8. **The C1 closure is circular**, and the 46 tok/s baseline comes from a different config. Both
   are covered above.

## Corrections made to `01-project/dflash-specdec.md`

All are inline `[review: …]` notes, and no analysis was rewritten:

1. Citations section: struck "Most DFlash2 code is not upstream". Upstream has the model, selector
   and speculator; the patches add lookup, chain and adaptive.
2. §1 Summary: the 22 ms is non-attention, non-drafter work from a 3080 TP=2 profile. The byte
   floor is 14.2 ms, and the overhead is 4.5-6.5 ms [E].
3. §5 Concurrency: 309 tok/s is DFlash2's C4 aggregate, not C8.
4. §6.1 Short-context check: marked circular.
5. §6.1 Speculation overhead: the baseline is from an older config. Recomputed at 4.5-6.5 ms [E].
6. §6.1 Drafter's share: 1.19 is GiB, the codebooks are not streamed, the floor is 1.68 GB ≈ 1.80
   ms, and lm_head is ≈39%.
7. §6.1 Stairs: the attention µs figures are at 8/16/32 query tokens, and the jump is the Marlin
   stair.
8. §6.1 Concurrency: the source table, +6.9 / +8.15 ms per resident, and the stair confound at C4.
9. §7 item 4: lm_head is ≈39% by bytes.
10. §7 item 5: the ≈8 ms jump is the Marlin stair only.
11. I appended a short "Reviewer additions" section covering tree-attention history, the tail
    thresholds, adaptive being inert at k=7, the selector gating and the e-profile cold vs
    profiled pass.

## Files

- `00-meta/reviews/phase1-dflash-specdec.md` (this file)
- `00-meta/reviews/phase1-dflash-specdec-weights.py`, `.out`: byte counts from the safetensors
  headers
- Edited: `01-project/dflash-specdec.md`, inline `[review: …]` notes plus Reviewer additions
