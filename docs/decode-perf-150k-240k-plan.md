# Raising single-user decode tok/s at 150k–240k: ranked, evidence-backed plan

**Scope.** Static analysis + research only. No code, benchmarks, or profilers were run for this
document. Analyzed tree: `main` @ `1cf86656c26b7725743c41a0ad7b9de99f5d7844` (identical to
`upstream/main`, syv-ai/HyperQwen; the fork tip `f6a5436` differs only by a docs/auth merge).

**Number labels.** Every figure is one of:
- **[REPO-MEASURED]** — quoted from this repo's docs/issues, with the path.
- **[EXTERNAL-MEASURED]** — another project's published result, with link.
- **[ESTIMATE]** — my arithmetic, with the calculation shown. Never a measurement.

**Target.** The brief left `[TARGET] tok/s at [CONTEXT LENGTH]` as a placeholder. This plan is
written against an illustrative target of **≥ 115 tok/s at 150k** and **≥ 60 tok/s at 240k**
(mixed prose, single stream, one 24 GB card), with the copy/quote profile at 240k ≥ 250 tok/s.
Substitute the real target; the ranking does not change.

**Where the repo contradicts the brief, the repo wins.** The brief quotes Setup D as
"~95–100 tok/s at ~150k" and Setup E as "~67 tok/s mixed at ~240k". The repo's own
depth-resolved numbers say those are **short-prompt cohort figures measured on servers
configured for 150k/240k**, not decode rates at that depth:
- D (`SPEC=mtp CTX=long`, fp8): 92.6–101.8 tok/s on the C1 cohort of ~1–2k prompts
  [REPO-MEASURED: docs/vllm-0.29.md lines 271, 332], but **68.1 tok/s at 112k depth**
  [REPO-MEASURED: docs/long-context.md lines 48–56, issue #11] and 80.3 @ 25k / 70.7 @ 60k
  [REPO-MEASURED: docs/gotchas.md gotcha 40, lines 825–830]. At true 150k depth expect
  ~60–70 [ESTIMATE: extrapolation of those two series].
- E (`CTX=huge`, KVarN + DFlash2): 93.6–107.2 tok/s C1 [REPO-MEASURED: docs/vllm-0.29.md
  lines 290, 333], **73.7 tok/s at 25k, 38.6 at 90k** [REPO-MEASURED: docs/vllm-0.29.md
  lines 126–127], 32.0 at 112k [REPO-MEASURED: docs/long-context.md line 53]. The "67 mixed"
  is a moderate-depth number; at true 240k depth the same curve reads ~19–25 tok/s
  [ESTIMATE: 62 ms/step at 90k + the measured ~0.43 ms/step per +1k token slope → ~127 ms/step
  at 240k, at ~2.4 tok/step]. The "164 quoting" figure is likewise depth-dependent.

So the honest baseline is: **~65–75 tok/s at true 150k depth (D), ~19–25 tok/s at true 240k
depth (E)**, and the ceiling analysis below is done against those, with the short-context
numbers noted where they are what the repo publishes.

---

## 1. Executive summary — top 5 opportunities

| # | Opportunity | Expected gain [ESTIMATE] | Quality risk | Effort |
|---|---|---|---|---|
| 1 | **Enable KVarN's shared-dequant verify kernel** (`KVARN_SHARED_VERIFY`, currently default-off over an unresolved corruption) + batch-1 kernel-efficiency pass on the KVarN decode path. The fallback re-dequantizes the whole KV cache **once per query token** (4× at MTP k=3, 8× at DFlash2 k=7) every step. | **240k: +80–200%** (step ~127 → ~35–45 ms); 112–150k KVarN: +40–90% | none (kernel math unchanged; validated in isolation already) | **M** |
| 2 | **Land the vLLM 0.30.0 port (PR #189, open, verified on the reference 3090) and switch on its spec-decode wins**: adaptive verification (vllm#52228), DFlash AOT-schedule drop (vllm#54374), gc-freeze graph capture (vllm#54646), `FULL_DECODE_ONLY` graphs (vllm#55095), Mamba-state-at-EAGLE-resume (vllm#53945). | **+5–15%** at both depths | none (spec decode is exact) | **S–M** |
| 3 | **Recover FULL CUDA graphs on the fp8/FlashInfer 150k path and re-enable MTP k=4** (PIECEWISE costs −6.6% step [REPO-MEASURED: gotcha 40]; k=4 is worth ~+7% [REPO-MEASURED: docs/optimizations.md lines 397–403]). Both are gated on the #34 FlashInfer crash, which the 0.29/0.30 pin + FlashInfer 0.6.18.post1 may already fix — a soak test decides. | **150k: +8–14%** | none | **S–M** |
| 4 | **DFlash2 + fp8 at 150k via the FA2-fp8 plugin geometry relaxation** (issue #153: TP=1 needs `(256,4)`/`(128,8)` admitted to the adapter's tested set). Gives DFlash2's acceptance + the lookup lane at 150k with FULL graphs, instead of the TRITON/int8 tier that decays with depth (gotcha 40) or KVarN's per-token verify. | **150k copy/quote: +30–100%**; mixed prose at depth: ~0 (MTP keeps that workload) | none | **M** |
| 5 | **int4-per-token-head KV + MQ-3D split-KV verify as the hedge 240k path** (`single-user/alternative.sh` + `VLLM_INT4_MQ_3D=1`): measured 38–46 tok/s at 88k on the 3090 [REPO-MEASURED: docs/spec-decode-scratch-token-units.md lines 366–378] vs KVarN's 38.6 at 90k, pool 314,915 tokens, GSM8K 96.0 / 100k needle retrieved [REPO-MEASURED: docs/long-context.md lines 456–465]. | **240k: 1.5–2.5× current E** if #1 stalls | **medium** (PPL at depth and 240k needle not yet published; int4 pth is coarser than KVarN k4v2) | **M** |

**Read of the whole table:** per-step time at batch 1 is ~70–80% **weight bytes** (a fixed
17.1 GB) and the rest is KV-cache read+dequant and kernel efficiency. Lossless weight
reduction does not exist (the body is already W4A16 Marlin near its roofline; the heads are
already int4-GPTQ). Therefore the ranking is dominated by (a) **not re-reading/re-dequantizing
KV per draft token**, and (b) **tokens per step** (drafting quality), with kernel-efficiency
recovery third. Lossy ideas (int4 KV beyond KVarN, W3A16 weights, lower-precision state) are
ranked below the lossless ones and each carries a quality budget check against Hard
constraint 2.

**Two cheap honorable mentions** (not top-5 because they do not move the stated benchmark):
- **Rebuild the MTP draft vocabulary over a multilingual corpus** (issue #196: the shipped
  40,960-id list covers ~23% of one reporter's real outputs — 5–8% on Chinese prose — and
  acceptance collapses to 1.06 tok/step there). Effort **S**, lossless, huge for non-en/da/code
  users; ~0 for the repo's en/da/code cohort (its 97.5% coverage is already measured
  [REPO-MEASURED: drafter/README.md lines 17–22]).
- **Multilingual-or-full head A/B** (`MTP_DRAFT_VOCAB=0`): the same reporter measured the full
  lm_head *faster* than even a rebuilt truncated head at `CTX=long` (80.6 vs 61–64 tok/s) —
  contradicts repo doctrine, one box, no maintainer reply. Open question Q3.


---

## 2. Decode hot-path map for setups D and E, with roofline math

### 2.1 Model constants (for all math below)

From the checkpoint config (`text_config` of `dbirks/Qwen3.8-27B-W4A16-AutoRound`,
[EXTERNAL-MEASURED: huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound/raw/main/config.json])
and the repo's own figures:

- 64 layers: **48 Gated DeltaNet (GDN) + 16 full attention** (every 4th layer;
  `full_attention_interval=4`). hidden 5120; attention: 24 q heads, **4 KV heads, head_dim
  256**; GDN: 16 key heads ×128, 48 value heads ×128; vocab 248,320;
  `max_position_embeddings` 262,144.
- **Weights per step (read once per forward):** ~15.9 GiB loaded
  [REPO-MEASURED: docs/spec-decode-scratch-token-units.md line 343] ≈ **17.1 GB** — W4A16
  Marlin body ~13.2 GB, int4-GPTQ lm_head 0.64 GB, int8 embed (one row gathered per step,
  negligible), int4-GPTQ MTP module [REPO-MEASURED: docs/quality.md lines 49–58].
- **Recurrent state per request:** 48 × (48×128×128) elements = 37.7 M elements = **151 MB
  fp32 / 75.5 MB fp16**, read+written every step [REPO-MEASURED: docs/optimizations.md lines
  54–58 ("~150 MB per request" fp32); fp16 state is the shipped default, docs/quality.md
  line 33].
- **KV bytes per token** (16 attention layers × 4 heads × 256 dims × K+V):
  bf16 65,536 B [REPO-MEASURED: gotcha 30]; fp8 32,768 B [REPO-MEASURED: docs/long-context.md
  line 11]; int8-per-token-head ~33,280 B [REPO-MEASURED: gotcha 39, 2080 B/layer]; KVarN
  k4v2_g128 840 B/layer → **13,440 B** [REPO-MEASURED: docs/long-context.md line 15,
  kvarn/README.md line 51]; int4-per-token-head ~17,900 B [ESTIMATE: 6.96 GiB pool ÷ 405,948
  tokens, docs/vllm-0.29.md line 420].
- **Drafter weights per step:** DFlash2 W4A16 1.19 GB [REPO-MEASURED: docs/optimizations.md
  lines 220–227]; MTP int4 ~0.45 GB [ESTIMATE from "~850 MB bf16 → int8", prepare/README.md
  line 16, then int4 GPTQ fast variant].
- **Card:** RTX 3090, 936 GB/s spec, 82 SMs, sm86, 250 W cap. Sustained effective bandwidth
  ~85–90% at these read sizes [REPO-MEASURED: gotcha 10 — the GDN decode kernel "already runs
  at ~85% of the 3090's memory bandwidth"; docs/optimizations.md lines 405–410 — the Marlin
  gap "is the memory system's ramp on 16–92 MB reads, not the kernel"].


### 2.2 Setup D — `SPEC=mtp CTX=long` (fp8 KV, FlashInfer, k=3, MAX_LEN 150k)

Launcher mapping [REPO-MEASURED: single-user/start_qwen.sh lines 202–204]:
`--kv-cache-dtype fp8`, `DRAFT_TOKENS=3`, MAX_LEN 150,000. fp8 on sm86 has exactly one
backend — FlashInfer — because FLASH_ATTN refuses fp8 KV (needs FA3/SM90+) and TRITON
refuses fp8e4nv below SM89 [REPO-MEASURED: gotcha 40, lines 777–787]. FlashInfer's
spec-decode path is single-token-decode-only, so the verify step runs **PIECEWISE** (no
FULL CUDA graph) [REPO-MEASURED: patches/triton-spec-attn-fp8-kv.patch preamble].

Per decode step, in order:
1. **MTP drafter, 3 chained single-token forwards.** Each: one drafter layer + the truncated
   40,960-row lm_head slice (`patches/qwen3_5-mtp-draft-vocab.patch`). The drafter's own
   1-layer KV grows with context; 3 serial passes per step.
2. **Target verify forward, 4 query tokens, one pass:** 64 layers — W4A16 Marlin GEMMs
   (17.1 GB), GDN chunk-scan on fp16 state, and 16 layers of fp8 FlashInfer multi-query
   attention over the full context (KV read once per step).
3. **Rejection sampler** (sort-free top-k/top-p, `patches/sampler-small-topk-fast-softmax.patch`;
   kernels prewarmed by `patches/spec-sampler-prewarm.patch`).
4. On accept, k+1 tokens commit and the GDN state rolls forward; on reject it restores from
   the checkpointed position (backstopped by `patches/vllm-pr50021-gdn-spec-bounds.patch`).

Step-time budget at 150k [ESTIMATE, bandwidth-only, 100% efficiency]: weights 17.1 + fp8 KV
150,000×32,768 B = 4.92 + state 0.15 + drafter ~0.5 = **22.7 GB → 24.2 ms → 41.3 tok/s
unspeculated ceiling**; at the measured 2.6 tok/step [REPO-MEASURED: docs/vllm-0.29.md
line 271] → **speculated ceiling ≈ 107 tok/s**. Measured 93–102 at C1 (short), ~68–83 at
depth → **~63–77% of the byte roofline at depth, ~85% short**. The at-depth gap is the
PIECEWISE downgrade (−6.6% step [REPO-MEASURED: gotcha 40, lines 803–810]), the FlashInfer
fp8 batch-1 decode kernel, and per-step host/sampler overhead.

### 2.3 Setup E — `CTX=huge` (KVarN k4v2 KV + DFlash2, MAX_LEN 245,760)

Launcher mapping [REPO-MEASURED: single-user/start_qwen.sh lines 197–199, 219–223]:
`--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`, DFlash2 k=7 (QLEN=8 verify), FULL graphs
restored for DFlash2 (residue fix `a75ee4b`/`b356e31`; MTP stays PIECEWISE — gotcha 33 lines
435–446), `--no-async-scheduling` (the lookup lane needs per-step draft feedback,
docs/optimizations.md lines 285–289).

Per decode step:
1. **DFlash2 drafter, one non-autoregressive pass** (5 layers, 2,048-token sliding window —
   cost independent of context) + optional lookup fill from the request's own history
   (`patches/dflash2-lookup-drafting.patch`).
2. **Target verify forward, 8 query tokens:** weights (17.1 GB), GDN state, and 16 layers of
   KVarN attention: the fused Triton kernel walks 128-token tiles — load packed 4-bit keys /
   2-bit values + scales, dequantize, Hadamard-rotated q·k, online softmax
   [REPO-MEASURED: kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py lines
   1053–1120].
3. Sampler + commit, as D.

Step-time budget at 240k [ESTIMATE, bandwidth-only]: weights 17.1 + KVarN KV 240,000×13,440 B
= 3.23 + state 0.15 + drafter 1.28 = **21.7 GB → 23.2 ms → 43 tok/s unspeculated ceiling**;
at 2.5 tok/step → **speculated ceiling ≈ 107 tok/s**; in copy mode at ~8–15 tok/step (lookup
engaged [REPO-MEASURED: docs/optimizations.md lines 307–314]) the ceiling is 250–400+ tok/s.
Measured 73.7 @ 25k, 38.6 @ 90k, ~19–25 extrapolated @ 240k → **~35–55% of roofline at 25k,
~20% at depth**. That gap is finding F1, and it is enormous.

### 2.4 What the roofline says, plainly

| term | 150k fp8 (D) | 240k KVarN (E) |
|---|---|---|
| weights | 17.1 GB (75%) | 17.1 GB (79%) |
| KV read | 4.92 GB (22%) | 3.23 GB (15%) |
| state + drafter | 0.65 GB (3%) | 1.43 GB (6%) |
| **step bytes** | **22.7 GB** | **21.7 GB** |
| ceiling @2.5–2.6 tok/step | ~107 tok/s | ~107 tok/s |

Three consequences:
1. **Weights dominate everywhere.** Even a perfect (free) KV cache caps D and E near
   ~110–125 tok/s at current acceptance. Lossless gains must come from *efficiency*
   (closing the measured-vs-roofline gap) and *tokens per step*.
2. **KVarN has already won the byte war at 240k** (KV is 15% of step bytes). The 240k
   problem is not capacity and not bytes — it is that the kernel *executes* far above its
   byte time (finding F1).
3. **To beat ~110–135 tok/s at any depth you need more tokens per step**, not faster
   kernels — which is why drafting quality (opportunities 3, 4, and the lookup/suffix
   family) is a first-class lever even though kernels look like the story.


---

## 3. Bottleneck findings (code evidence + confidence)

**F1 — The 240k bottleneck: KVarN verify re-dequantizes the whole KV cache once per query
token, every step (and the shared-dequant fix already exists but is default-off).**
`kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py` `kvarn_verify_attention` (lines
877–960) documents two modes: a UNIFORM shared-dequant path where "the request's QLEN tokens
SHARE each block's dequant, so KV bytes and dequant ALU match single-token decode", and a
per-token fallback with "**QLEN-x redundant dequant**" (docstring, lines ~889–895). The
shared path is gated behind `envs.KVARN_SHARED_VERIFY`, **default OFF**, because "serving
with it corrupts the MTP drafter's proposals (invalid [-1,...] spec tokens, embedding index
asserts at temperature>0, degenerate greedy output) through a mechanism not yet isolated —
suspicion is an interaction with async scheduling / drafter metadata rather than kernel
math" (lines ~936–946). The kernel was "numerically validated in isolation (matches the
per-token kernel within fp32 reduction noise on live inputs)". So at HEAD, every verify step
on `CTX=huge` runs per-token: **4× redundant dequant at MTP k=3, 8× at DFlash2 k=7**, growing
linearly with context. Quantified [ESTIMATE]: at 90k the per-token path visits
8 × (90,000/128) × 16 ≈ 90,000 tile-dequants/step; at ~0.33 µs/tile-visit (implied by the
repo's own 25k/90k numbers [REPO-MEASURED: docs/vllm-0.29.md lines 126–127]) that is ~30 ms
of a ~52–62 ms step; sharing the dequant cuts it to ~3.8 ms → step ~25–35 ms → **~2–2.5× at
depth**. This also explains the observed steepening of the KVarN tax with context
[REPO-MEASURED: docs/long-context.md lines 57–59, issue #11: 1.22× batch at 100k → 2.13×
single-user at 112k] — the tax is per-query-token, and single-user verifies 4–8 queries.
**Confidence: HIGH** that this is the dominant 240k cost (code + repo's own depth slope
agree); MEDIUM that enabling the shared kernel is easy (corruption mechanism unresolved —
but the launcher already runs `--no-async-scheduling` at CTX=huge for the lookup lane, and
the suspicion named is async-scheduling/drafter-metadata, so enabling may be a validation
exercise, not a rewrite).

**F2 — fp8 verify on sm86 is structurally downgraded: PIECEWISE graphs and k=3 (not 4).**
Two independent constraints on D: (a) FlashInfer's spec-decode path is single-token-only →
PIECEWISE, measured −6.6% step (27.2 → 25.4 ms) [REPO-MEASURED: gotcha 40 lines 803–810;
patches/triton-spec-attn-fp8-kv.patch preamble]; (b) k=4 on FlashInfer crashed with an
illegal memory access on 0.28.0 ("n=4 eventually dies, n=3 stable"), so CTX=long gives up
~7% [REPO-MEASURED: docs/optimizations.md lines 397–403]. The repo's Triton fp8 split-KV
verify kernel exists but is sm89+ [REPO-MEASURED: gotcha 57, lines 1285–1299].
**Confidence: HIGH** (all repo-measured); whether 0.29/0.30 + FlashInfer 0.6.18.post1 lifts
either gate needs a GPU soak (open question Q4).


**F3 — DFlash2's drafter sees only a 2,048-token window, so its acceptance edge collapses on
non-copy text at depth; it wins at long context only when the lookup lane fills from the
prompt.** Repo doctrine: "DFlash2 past 64k is worth it only for context reproduction and
loses to SPEC=mtp CTX=long roughly 2:1 on everything else" [REPO-MEASURED: .env.example
lines 18–21]; 128k acceptance divergence measured in issue #60. The lookup lane restores it
for copy/quote (164 tok/s quoting at CTX=huge [REPO-MEASURED: single-user/README.md]).
**Confidence: HIGH.** Consequence: DFlash2-at-150k (opportunity 4) is a *copy-workload*
play, not a mixed-prose play.

**F4 — Weights are 75–79% of step bytes and are already near their roofline.** W4A16 Marlin
retuning measured +0.4% end-to-end [REPO-MEASURED: docs/optimizations.md lines 190–195,
405–410: "the remaining gap to peak bandwidth is the memory system's ramp on 16–92 MB reads,
not the kernel"]. lm_head/MTP/drafter are already int4-GPTQ [REPO-MEASURED: docs/quality.md
lines 49–58]. **Confidence: HIGH.** Consequence: no lossless weight lever remains; W3A16 is
the only further weight cut and is lossy (table row T7).

**F5 — The GDN/recurrent path is not the decode bottleneck.** The DeltaNet decode kernel
already runs at ~85% of memory bandwidth and every tuning variant lands within 3%
[REPO-MEASURED: gotcha 10, lines 149–153]; fp16 state read+write is ~0.16 ms/step
[ESTIMATE: 151 MB ÷ 936 GB/s]. State size bounds *concurrency* (pages scale with the verify
block, not slots [REPO-MEASURED: gotcha 29]), not single-user speed. **Confidence: HIGH.**
Consequence: a fused GDN decode kernel (the external pattern — FlashInfer SM100
`fused_kda_decode` 1.33× at 1 row [EXTERNAL-MEASURED: flashinfer v0.6.18 release notes],
vLLM #53835 SM110) buys little on sm86 here; deprioritized (rejected idea R6).

**F6 — CPU/host overhead is small: PIECEWISE on the fp8 path (F2), and
`--no-async-scheduling` at CTX=huge costs <1% at batch 1** [REPO-MEASURED:
docs/optimizations.md lines 285–289]. The sampler is already patched; the DFlash draft pass
is a captured graph [REPO-MEASURED: gotcha 20]. **Confidence: HIGH.**

**F7 — The int8-QK prefill kernel and Marlin int8 GEMM tunes are gated to exact geometry /
known to misfire on this checkpoint, but those are prefill/batch concerns, not single-user
decode** [REPO-MEASURED: patches/prefill-attn-int8.patch, marlin-int8-negative-scales.patch;
int8 activations buy nothing at batch 1, docs/quality.md line 49, and may cost ~9% decode
via acceptance — issue #62, unconfirmed]. **Confidence: HIGH** (not on the D/E decode path).

**F8 — Reliability cliffs on the exact D/E profile: the #34 FlashInfer+MTP Xid-31 at 28–34k
(cause unattributed [REPO-MEASURED: gotcha 40]), the #107 engine stepping-stalls at ~190k
MTP/fp8 (upstream candidate vllm#50021, still open), and Bug B residue corruption at
CTX=huge (mitigated: PIECEWISE for MTP, FULL for DFlash2 [REPO-MEASURED: gotcha 33]).** No
steady-state tok/s cost, but any change touching the verify path must re-run their sweeps
(`bench/residue_sweep.py`, `bench/verbatim.py`). **Confidence: HIGH.**


---

## 4. Ranked opportunity table

Gains are [ESTIMATE] at true depth (150k / 240k) vs the honest baselines (~65–75 tok/s D,
~19–25 tok/s E) unless noted. Quality risk is against Hard constraint 2. Evidence labels as
defined at the top.

| rank | idea | mechanism | evidence | gain @150k | gain @240k | quality risk (predicted impact) | effort | files / deps |
|---|---|---|---|---|---|---|---|---|
| **T1** | Enable + fix KVarN shared-dequant verify (`KVARN_SHARED_VERIFY`), then batch-1 kernel pass (splits/tile sweep) | stop re-dequantizing KV per query token; share dequant across QLEN | [REPO-MEASURED: triton_kvarn_decode.py 877–960; docs/vllm-0.29.md 126–127] + [ESTIMATE §3/F1] | n/a (KVarN not the 150k tier) | **+80–200%** | none — kernel math unchanged; needle re-check only | M | `kvarn/files/.../triton_kvarn_decode.py`, `kvarn_attn.py`; async-scheduling hypothesis test |
| **T2** | Land 0.30.0 (PR #189) + adaptive verification (#52228), DFlash AOT drop (#54374), gc-freeze (#54646), FULL_DECODE_ONLY (#55095), Mamba-resume (#53945) | fewer wasted drafts when acceptance dips; less scheduler/graph overhead | [EXTERNAL-MEASURED: vLLM v0.30.0 release notes; PR #189 verified pools on 3090] | +5–15% | +5–15% | none (spec decode exact; IFBench/PPL unchanged by construction) | S–M | merge PR #189; launcher flags; re-run acceptance |
| **T3** | FULL graphs on fp8 verify + MTP k=4 at CTX=long (post-soak) | remove PIECEWISE downgrade; deepen drafts | [REPO-MEASURED: gotcha 40; optimizations.md 397–403] | +8–14% | n/a | none | S–M | `start_qwen.sh` DRAFT_TOKENS, `cudagraph_mode`; gated on #34 soak |
| **T4** | DFlash2 + fp8 at 150k (FA2-fp8 geometry relaxation, #153) | DFlash2 acceptance + lookup lane at 150k, FULL graphs | [REPO-MEASURED: issue #153; gotcha 40; #194] | copy +30–100%; mixed ~0 | n/a | none | M | plugin adapter gate `(256,4)`/`(128,8)`; port to 0.29/0.30 image; battery |
| **T5** | int4-per-token-head KV + MQ-3D (`VLLM_INT4_MQ_3D=1`) as hedge 240k path | proper split-KV verify over int4 cache; pool 314,915 tokens | [REPO-MEASURED: spec-decode-scratch doc 366–378; long-context.md 435–465; issue #86] | n/a | 1.5–2.5× current E | **medium** — PPL at depth + 240k needle unpublished; GSM8K 96.0 & 100k needle OK | M | `alternative.sh` + mq3d patches; quality battery |
| T6 | Multilingual draft-vocab rebuild (#196) | acceptance on non-en/da/code traffic | [REPO-MEASURED: issue #196] | ~0 on cohort; large for multilingual users | same | none | S | `prepare/build_draft_vocab.py --corpus` |
| T7 | W3A16 (~3bpw) body, EXL3-style, mixed W3/W4 | cut the dominant 17.1 GB weight term ~20–25% | [EXTERNAL-MEASURED: ExLlamaV3 Qwen3.8-27B@4bpw 48.6 t/s 3090Ti; Flash-Next@3bpw 62.8 t/s] + [ESTIMATE] | +15–20% | +15–20% | **high** — likely exceeds PPL +2% budget; fallback W3-on-MLP-only | M–L | new quant in `prepare/`; Marlin W3 or EXL3 kernel |
| T8 | Suffix/prompt-lookup proposer for MTP at 150k (backport vLLM main suffix decoding) | copy-mode drafts for the MTP tier (today lookup is DFlash2-only) | [EXTERNAL-MEASURED: SuffixDecoding ~5.3× on agentic/copy; vLLM main spec-decode docs] | copy +20–60% | n/a | none | M | backport `suffix_decoding`; conflicts with DFlash2 lane — pick per profile |
| T9 | KVARN_NUM_KV_SPLITS / tile-size autotune at batch 1 | occupancy of the fused kernel | [REPO-MEASURED: triton_kvarn_decode.py 37–78] | n/a | +10–30% (stacked on T1) | none | S | env sweep only |
| T10 | int8 recurrent state | halve state bytes | [ESTIMATE: 0.08 ms/step] | +0.3% | +0.3% | medium (state precision) for ~0 gain | — | rejected (see R5) |


---

## 5. Top 5 — implementation sketches and validation plans

### T1 — KVarN shared-dequant verify (the 240k fix)

**Mechanism.** At HEAD, `CTX=huge` verify runs the per-token fallback kernel: the whole
quantized KV cache is dequantized once per query token (4× MTP, 8× DFlash2) every step. The
shared-dequant kernel (`_kvarn_fused_verify_stage1` + stage2 combine) exists, is numerically
validated in isolation, and is gated off by `KVARN_SHARED_VERIFY` default-0 because serving
with it corrupted the MTP drafter's proposals "through a mechanism not yet isolated —
suspicion is an interaction with async scheduling / drafter metadata"
(triton_kvarn_decode.py lines ~877–960).

**Implementation sketch.**
1. Reproduce the corruption minimally: boot `SPEC=mtp CTX=huge` with
   `KVARN_SHARED_VERIFY=1`, async scheduling ON vs OFF (`ASYNC_SCHED=0`), and run
   `bench/residue_sweep.py` + a tool-calling soak. The named suspect is async scheduling;
   single-user `CTX=huge` already forces `--no-async-scheduling` for the DFlash2 lookup, so
   there is a real chance the shared kernel is already safe on the shipped profile.
2. If the corruption is async-only: gate the shared path on `not async_scheduling` (or on
   the V2-runner uniform-decode shape) rather than on a debug env var; promote the env to a
   registered knob with a safe default.
3. If it corrupts even without async scheduling: bisect the vq plan (`Seq_lens_ptr` CPU-side
   build vs the device `seq_lens` — the builder's own comment says the device tensor "can
   disagree" under async spec decode; that disagreement is the likely invalid-spec-token
   source). Pin the plan from the same snapshot the scheduler used for the step.
4. Then the batch-1 efficiency pass (T9): sweep `KVARN_NUM_KV_SPLITS` (16/32/64) and the
   stage-1 tile `BLOCK_N` at 90k/150k/240k; check SM occupancy of the `(B, Hk, splits)` grid
   at batch 1 (4 KV heads × splits on 82 SMs).

Diff outline: in `kvarn_verify_attention`, replace the `envs.KVARN_SHARED_VERIFY` clause with
a predicate on scheduler mode + validated geometry; in `kvarn_attn.py` plumb the flag;
document in `kvarn/README.md`. ~50–150 lines + tests.

**Benchmark + quality validation (Setup E, `CTX=huge SPEC=dflash2 PREFIX_CACHE=1`).**
- Decode vs depth: `bench/labd_bench.py <tag> --ctx 25000,90000,150000,240000 --tasks
  qa,summary` plus a verbatim-copy cell (`bench/labd_bench.py <tag> --ctx 20000` with the
  document-copy task). Arms: flag off vs on, same boot. **Pass:** step time at 90k drops
  ≥ 30% (target decode ≥ 55 tok/s at 90k, ≥ 45 at 240k from the ~19–25 baseline); no arm
  slower than baseline by >3%.
- Correctness/quality: `bench/quality_battery.py huge-shared --gsm-n 200` (GSM8K ≥ baseline
  −1.0 pt → ≥ 95.0–95.5); perplexity vs E baseline (8.236 → ≤ +2% → ≤ 8.40);
  `bench/needle_test.py 150000 0.9` and `bench/needle_test.py 240000 0.9` retrieved;
  `bench/residue_sweep.py` + `bench/verbatim.py` clean at all 128 residues (Bug B/F8);
  30-min `bench/labd_soak.py` without degenerate output.
- IFBench (299 prompts, per docs/quality.md) on the best arm if the kernel path changed any
  numerics: ≥ 77.3 (baseline 78.3 − 1.0).

**Strongest reason it might NOT help:** the corruption may not be async-scheduling but a
real kernel/metadata bug that only shows under live drafter traffic, and the fix could
require re-plumbing the verify plan — in which case T1 slips from M to L. Even then, the
T9 sweep alone (splits/tile) is a cheap partial.

---

### T2 — vLLM 0.30.0 (PR #189) + adaptive verification

**Mechanism.** The port PR is open and verified on the reference 3090 with identical KV
pools on every shipped mode [REPO-MEASURED: PR #189 description]. 0.30.0 carries: acceptance
estimation for adaptive verification (vllm#52228, merged 2026-09-14) — the engine shortens
the draft when acceptance is low, which matters most exactly where verify cost is highest
(long context); DFlash drafters dropping FlashAttention's AOT schedule (vllm#54374);
gc-freeze during graph capture (vllm#54646); `FULL_DECODE_ONLY` graphs (vllm#55095); Mamba
state cached at the EAGLE resume position (vllm#53945, a correctness enabler for EAGLE-class
drafters on hybrid models).

**Implementation sketch.** Land PR #189 under the repo's two-box bar (docs/vllm-0.29.md
lines 143–166 describes the procedure). Then: enable adaptive verification for the D and E
profiles (confirm #52228 covers MTP and DFlash2 on the V2 runner; it is documented as "every
draft-model speculator"), keep the launcher's explicit prefix-cache retention (PR #189
already handles the 0.30 default change), and re-run the mode acceptance tables.

**Benchmark + quality validation (Setups B, D, E).** `bash bench/run_benchmarks.sh single`
twice per mode (keep second run), C1–C8 + tok/step, 0.29 vs 0.30 on the same box in one
session; plus `bench/labd_bench.py --ctx 90000,150000` on D and E for the depth rows.
**Pass:** no mode regresses >3% decode; D or E gains ≥ 5%; `bench/quality_battery.py` per
mode within the constraint-2 budget (GSM8K ≥ −1.0 pt, PPL ≤ +2%); Bug-B and needle sweeps
clean on E. **Fallback:** if adaptive verification measurably cuts tok/step (estimator
mispredicts at depth), keep 0.30 but leave it off — the other 0.30 wins stand alone.

**Strongest reason it might NOT help:** adaptive verification's estimator could under-draft
at depth (acceptance at long context is lower and spikier), trading verify cost for tok/step
and netting ~0; and the MRv1↔MRv2 divergence may put some 0.30 spec-decode wins out of this
model's reach. The pools being identical means even a wash costs nothing but the port effort.


---

### T3 — FULL CUDA graphs + MTP k=4 on the fp8 150k path

**Mechanism.** D runs its verify PIECEWISE because FlashInfer's spec-decode path is
single-token-only (−6.6% step [REPO-MEASURED: gotcha 40 lines 803–810]) and at k=3 because
k=4 crashed on 0.28.0's FlashInfer ("n=4 eventually dies, n=3 stable", −~7%
[REPO-MEASURED: docs/optimizations.md lines 397–403]). Both gates are version-sensitive:
the 0.29→0.30 pin bumps FlashInfer to 0.6.18.post1 [REPO-MEASURED: PR #189 description].

**Implementation sketch.**
1. On the 0.30 image: soak `SPEC=mtp CTX=long DRAFT_TOKENS=4` with the F8 watch items
   (concurrent-garbage check from #121, long multi-turn traffic, `dmesg` watch for Xid).
   The crash signature was "one request finishes while another is mid-generation", so the
   soak must mix finishing/starting requests at 28–34k context.
2. If stable: ship `DRAFT_TOKENS=4` at CTX=long and test `--cudagraph-mode FULL` (or
   FULL_DECODE_ONLY from #55095) on the verify path; measure step time and tok/step.
3. If k=4 still crashes: keep k=3 and pursue only the graph-mode half; the #34 issue stays
   the tracker.

**Validation (Setup D).** `bench/run_benchmarks.sh single` (C1/C2 at default + greedy,
tok/step), plus `bench/labd_bench.py --ctx 60000,112000,150000 --tasks qa,summary` for the
depth curve; 4 h stability soak with request churn. **Pass:** decode +≥ 8% at C1 and at
112k, zero Xid/EngineDeadError, quality within budget (`bench/quality_battery.py long-k4
--gsm-n 200`: GSM8K ≥ 95.5; PPL ≤ +2%). **Fallback:** `DRAFT_TOKENS=3` + FULL graphs only.

**Strongest reason it might NOT help:** the #34 crash is unattributed (FlashInfer workspace
vs async-scheduling window) — it may reproduce on 0.30 too, in which case only the smaller
graph-mode gain (~+6.6% step, partially offset by the measured int8-acceptance give-back if
the KV dtype has to change) survives.

---

### T4 — DFlash2 + fp8 at 150k (FA2-fp8 geometry relaxation, issue #153)

**Mechanism.** Today DFlash2 at >64k must take the TRITON/int8 tier (decays with depth:
−34% decode at 60k [REPO-MEASURED: gotcha 40 lines 825–832]) or KVarN (F1). The FA2-fp8
plugin proved DFlash2+fp8 on sm86 at TP=2 (103/192 tok/s [REPO-MEASURED: issue #153]) but
its adapter refuses TP=1 geometries: the model needs `(256,4)` target / `(128,8)` drafter,
the adapter tested `(256,1/2)` and `(128,4)`. The maintainer's read: the per-head work at
TP=1 is identical (GQA ratios 6 and 4 at every TP), so this is "a gate relaxation plus a
correctness run, not new cubins" [REPO-MEASURED: issue #153 maintainer comment].

**Implementation sketch.** Patch the plugin adapter's geometry set to admit `(256,4)` /
`(128,8)`; run its `check_full.py` on one 3090 at those shapes; port the plugin to the
0.29/0.30 image (it builds against a pinned 0.27.1 image); serve `SPEC=dflash2` +
`--kv-cache-dtype fp8` on FLASH_ATTN with FULL graphs at MAX_LEN 150000. Then measure.
This creates a "Setup D-prime": DFlash2 acceptance + the lookup lane at 150k.

**Validation (Setup D-prime, copy-focused).** `bench/run_benchmarks.sh single` (C1 cohort),
plus `bench/labd_bench.py --ctx 60000,112000,150000` with the document-copy/verbatim tasks
(the workload this tier is for) and qa/summary for the mixed check. **Pass:** copy-task
decode ≥ MTP-at-150k +30% with identical verbatim fidelity (`bench/verbatim.py` coverage
≥ baseline); mixed-task decode ≥ MTP − 5%; quality per budget
(`bench/quality_battery.py dflash2-fp8 --gsm-n 200`; PPL — fp8 KV is already the D-tier
dtype with published PPL parity, docs/long-context.md line 33).
**Fallback:** if the adapter author will not take the shapes, maintain the two-line gate as
a repo patch against the plugin with CI verifying against pinned upstream (option (a) in the
maintainer's comment).

**Strongest reason it might NOT help:** DFlash2's advantage at depth is concentrated on copy
work (F3); on mixed prose at 150k it may still lose to MTP ~2:1, so D-prime could end up a
second niche tier rather than the 150k default. The plugin also self-describes incomplete
soak/quality validation, so the correctness run is real work, not a formality.

---

### T5 — int4-per-token-head KV + MQ-3D split-KV verify (hedge 240k path) — LOSSY

**Mechanism.** vLLM's stock `int4_per_token_head` cache now works with DFlash2 on this repo
(314,915-token pool at 256k [REPO-MEASURED: docs/long-context.md lines 441–449]) and the
MQ-3D multi-query 3D split-KV verify kernel (`patches/spec-decode-int4-kv-mq3d.patch` +
`VLLM_INT4_MQ_3D=1`) is the difference between ~9 and 94–108 tok/s [REPO-MEASURED: issue
#60]. The captured-path A/B on the reference 3090: 2D→3D at 24k/49k/88k = 119→205 /
75→160 / 19→46 tok/s degenerate-repeat and 44→79 / 35→47 / 15→38 prose [REPO-MEASURED:
docs/spec-decode-scratch-token-units.md lines 366–378], with PPL/GSM8K parity between the
two kernel orders (8.2079/94.5 vs 8.2087/95.5, same file lines 358–364). KVarN at the same
90k reads 38.6 [REPO-MEASURED: docs/vllm-0.29.md line 127] — i.e., int4+MQ-3D is already at
KVarN's depth rate, with a flatter slope (2.42–2.5× split-KV win held at 88k) and 1.23× the
pool. If T1 stalls, this is the 240k path.

**Implementation sketch.** `bash single-user/alternative.sh` with `VLLM_INT4_MQ_3D=1`,
`DFLASH_TOKENS=7` (15 does not fit at 256k — gotcha 47), MAX_LEN 240000. The work is
validation, not code: produce the missing quality-at-depth evidence, then decide KVarN vs
int4 as the shipped CTX=huge default on quality grounds (int4 pth has no Hadamard/rotation —
it is coarser than KVarN k4v2 on outlier channels).

**Validation (Setup E-prime).**
- Quality first (it gates everything): `bench/quality_battery.py int4kv-240 --gsm-n 200`
  (GSM8K ≥ 95.0 vs the 96.0–96.5 band → within −1.0 pt); perplexity vs the E baseline
  (KVarN reads 8.236 [REPO-MEASURED: docs/long-context.md line 33]; pass ≤ +2% → ≤ 8.40;
  note int4 KV PPL at depth is currently unpublished, so this is a measurement, not a
  formality); `bench/needle_test.py 150000 0.9` and `240000 0.9` retrieved (100k already
  retrieved [REPO-MEASURED: docs/long-context.md lines 456–465]); IFBench ≥ 77.3 if the PPL
  move is > 1%.
- Then speed: `bench/labd_bench.py --ctx 90000,150000,240000 --tasks qa,summary` + copy
  cells. **Pass:** decode ≥ KVarN-at-HEAD ×1.5 at 150k+ with the quality gates above met.
- **Fallback if quality exceeds budget:** keep KVarN for ≤128k and int4 only >128k (a
  context-tiered default the launcher can already express), or hold int4 K and raise V to
  8-bit (k4v8-style mixed mode — values are the 2-bit weak link; KVarN's own ablations
  point at V [EXTERNAL-MEASURED: KVarN repo/paper, github.com/huawei-csl/KVarN]).

**Strongest reason it might NOT help:** the unpublished perplexity at depth may fail the
+2% budget (int4 pth quantizes per token per head with no rotation; long-context retrieval
leans on outlier keys), in which case the whole path is a capacity feature, not a speed
feature, and T1 (lossless) remains the only 240k route.

---

## 6. Ideas rejected, and why (including maintainer-rejected)

- **R1 — Marlin tile tuning / int8 activations at batch 1.** Measured, not assumed: +0.4%
  end-to-end ("the remaining gap to peak bandwidth is the memory system's ramp on 16–92 MB
  reads, not the kernel") [REPO-MEASURED: docs/optimizations.md lines 190–195, 405–410];
  int8 activations buy nothing at batch size 1 [REPO-MEASURED: docs/quality.md line 49] and
  may cost ~9% decode via acceptance (issue #62, unconfirmed).
- **R2 — Fine-tuning the MTP head.** Done and rejected with data: KL halves, greedy top-1
  unchanged; Qwen's head is "already at the ceiling of a single-layer chain drafter"
  [REPO-MEASURED: drafter/README.md lines 29–37; docs/optimizations.md lines 405–407].
- **R3 — Skipping the drafter when the lookup overwrites all its proposals.** Tried twice,
  loses net 6% ("the drafter is covering the positions past the end of the match")
  [REPO-MEASURED: gotcha 26].
- **R4 — n-gram chains at sampling temperature.** Fundamental (−8% C1): a point-mass
  proposal accepts with p(token); greedy-only gate shipped [REPO-MEASURED: issue #38].
- **R5 — int8 / lower-precision recurrent state.** Speed gain ~0.08 ms/step (state is 0.7%
  of step bytes) for a real precision risk — backwards by any measure.
- **R6 — Fused single-launch GDN decode kernel (the SM100/SM110 pattern).** The GDN decode
  kernel already runs at ~85% of memory bandwidth; variants land within 3%
  [REPO-MEASURED: gotcha 10]. Launch overhead is hidden by CUDA graphs on the shipped
  profiles. The external 1.33× figure is on SM100 where the baseline is different.
- **R7 — TurboQuant KV backend (in vLLM 0.30).** KVarN's own comparison: ~2.4× TurboQuant's
  throughput, and vLLM's blog numbers show 40–52% lower throughput for the capacity
  [EXTERNAL-MEASURED: kvarn README references; vllm.ai/blog/2026-05-11-turboquant]. KVarN
  already wins this slot.
- **R8 — Eviction-based long-context methods (SnapKV/H2O etc.).** Forbidden by Hard
  constraint 2/3 (they drop tokens from the context). Not evaluated further.
- **R9 — Approximate/sparse attention (training-free).** Retrieval-at-depth is a hard
  requirement; training-free sparse attention routinely degrades needles, and the KV read is
  only 15–22% of step bytes at 150k fp8 anyway (§2.4), so the upside is capped. High risk,
  low ceiling.
- **R10 — EAGLE-3 / Medusa retrain.** Needs a training pipeline and Mamba-state-at-resume
  (only fixed in 0.30, vllm#53945); DFlash2 already occupies the block-drafter slot with
  better measured acceptance (4.80 vs 4.28 tok/step on bf16 [REPO-MEASURED:
  docs/optimizations.md lines 205–208]). Revisit only if T2 frees the hybrid-EAGLE path.
- **R11 — More draft depth at k=5+.** Measured: "going deeper (k=5) loses again: 106 / 105.
  k=4 is the knee" [REPO-MEASURED: docs/optimizations.md line 397].
- **R12 — Bigger prefill chunks / TTFT work.** Out of scope (prefill, not decode); also
  measured not to boot above 2048 [REPO-MEASURED: gotcha 43].



---

## 7. Open questions only a real GPU run can answer (as specific experiments)

- **Q1 (T1 enabler).** Boot `SPEC=mtp CTX=huge KVARN_SHARED_VERIFY=1` under
  `ASYNC_SCHED=0` and under async, one request at a time, `bench/residue_sweep.py` +
  30-min tool-call soak. Does the corruption reproduce with async off? If not, T1 is a
  validation-and-gating task; if yes, the vq-plan snapshot bisect in T1 step 3 is the next
  experiment.
- **Q2 (the honest depth curve).** What are D and E decode tok/s at *true* 150k and 240k on
  the reference 3090 today, at fixed tok/step? Run `bench/labd_bench.py --ctx
  25000,60000,90000,112000,150000` (D) and `--ctx 90000,150000,200000,240000` (E) on one
  boot each, reporting step ms and tok/step separately. The published 95–100 / 67–164 are
  short/moderate-depth numbers (see the baseline note at the top); every target in this plan should be re-baselined on
  this curve.
- **Q3 (#196 contradiction).** At `CTX=long`, A/B `MTP_DRAFT_VOCAB=1` (shipped list) vs a
  rebuilt multilingual list vs `=0` (full head) on the reference 3090, en/da/code cohort
  plus a Chinese-prose cell. The reporter measured full head *faster* than the truncated
  head at 150k (80.6 vs 61–64 tok/s) — if that reproduces, the draft-vocab truncation is a
  pessimization at long context and the launcher should stop shipping it there; if it does
  not, the fix is just the corpus rebuild (T6).
- **Q4 (T3 enabler).** On the 0.30 image with FlashInfer 0.6.18.post1: does
  `DRAFT_TOKENS=4` at `CTX=long` still Xid/IMA under request churn at 28–34k (#34), and
  does FULL (or FULL_DECODE_ONLY) graph mode on the fp8 verify pass the concurrent-garbage
  check from #121? Both are boot-and-soak, no code.
- **Q5 (KVarN kernel occupancy).** Profile one decode step at 90k and 240k on E
  (`KVARN_SPEC_DEBUG=1` + nsys): what fraction of step time is `_kvarn_fused_verify_*`
  stage-1 vs stage-2 vs the drafter vs the target GEMMs, and what is the achieved
  bytes/tile-visit vs the 0.114 µs/tile byte time? This decides whether T1 alone suffices
  or the T9 tile/split retune is also needed.
- **Q6 (T2 scope).** Does vllm#52228's adaptive verification actually engage for MTP and
  DFlash2 on this model under the V2 runner at batch 1 (log the per-step draft count under a
  mixed workload), and does it help or hurt tok/step at 112k depth?
- **Q7 (int4 KV quality).** Perplexity of the int4-per-token-head cache at 100k+ depth, and
  needles at 150k/240k (T5's gate). Unpublished today; the answer decides KVarN vs int4 as
  the shipped CTX=huge dtype.
- **Q8 (#107 stalls).** Do the stepping-stalls at ~190k MTP/fp8 reproduce on 0.30, and does
  the still-open upstream vllm#50021 (or the repo's backport) cover the sawtooth shape?
  Reliability, not throughput, but it gates any "run at 190k+" recommendation.

---

## Appendix A — multi-GPU (outside the primary single-card plan)

Not part of the plan (Hard constraint 3), recorded because the repo has measured it:
- **2× 3090 TP=2:** +16–35% decode at batch 1 with P2P/PCIe [REPO-MEASURED: issue #40];
  168.6–172.7 tok/s C1 on 2× 3090 NVLink [REPO-MEASURED: issues #159/#164]; **−24% without
  P2P** [REPO-MEASURED: issue #190]; 182.8 tok/s C1 after `expandable_segments:False`
  [REPO-MEASURED: docs/multi-gpu.md lines 122–128]. TP=3 is invalid for this model (4 KV
  heads, 64 layers) [REPO-MEASURED: docs/multi-gpu.md lines 15–26].
- **MTP at TP>1:** int8 KV on TRITON_ATTN beats fp8/FlashInfer 1.43–1.89× on sm120
  [REPO-MEASURED: docs/multi-gpu.md lines 156–164]; unmeasured on Ampere TP>1 — the doc
  names it "the most useful A/B left".
- The second card buys a second memory system, which is exactly what batch-1 decode is
  bound by (§2.4) — hence the outsize gains; but it is not the 24 GB plan.

## Appendix B — sources

Repo (this tree): README.md; PATCHES.md; docs/{optimizations,long-context,benchmarks,
quality,gotchas,vllm-0.29,spec-decode-scratch-token-units,main-track,multi-gpu}.md;
single-user/README.md; single-user/start_qwen.sh; .env.example; kvarn/README.md;
kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py; drafter/README.md;
prepare/README.md; patches/spec-decode-attn.patch; patches/triton-spec-attn-fp8-kv.patch;
bench/{quality_battery.py,run_benchmarks.sh}; model config from
huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound/raw/main/config.json.

GitHub (syv-ai/HyperQwen): issues #11, #25, #34, #38, #40, #52, #57, #60, #62, #73, #86,
#103, #105, #107, #121, #153, #159, #160, #164, #174, #190, #192, #194, #196; PR #42, #46,
#148, #188, #189. Upstream vLLM PRs: #50021 (open), #52228, #53945, #54374, #54646,
#55041, #55095, #55450, #55760, #58024–#58028 (open), #54282, #52789; vLLM v0.30.0 release
notes (2026-09-22).

External: FlashInfer v0.6.18/v0.7.0 release notes (flashinfer-ai/flashinfer); SGLang
releases (sgl-project/sglang); llama.cpp (ggml-org/llama.cpp) and BeeLlama
(Godl1nk/beellama.cpp, 3090-measured KV-quant KLD ladder); ExLlamaV3
(turboderp-org/exllamav3: Qwen3.8-27B@4bpw 48.6 t/s 3090Ti; Flash-Next@3bpw+ngram 62.8
t/s); TensorRT-LLM (NVIDIA/TensorRT-LLM); LMDeploy (InternLM/lmdeploy); KVarN
(huawei-csl/KVarN); SuffixDecoding; vllm.ai/blog/2026-05-11-turboquant; ninfer-3090
(Don-Chad/ninfer-3090, comparison baseline).

