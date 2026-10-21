# EAGLE-4: overview, claims, support and licence

Role: eagle-analyst (architecture, inference, build). Sources:
- EAGLE-4 checkout `~/research-src/EAGLE-4` at `7b7b5a5`.
- The paper `EAGLE4_Paper.pdf`: 12 pages, PDF page = printed footer.
- The HF repo `EAGLE-Corp/Qwen3.8-27B-EAGLE4` at sha `f58a598`, read through the HF API and a safetensors-header range read. No weights were downloaded.
- vLLM reference checkout `~/vllm`: tags `v0.31.0` and `HEAD` = `v0.32.1rc0-68-g4bd75e9`.

Labels:
- **paper-reported**: a number from the paper.
- **repo-reported**: a number from EAGLE-4 docs or profiles.
- **measured**: I ran it here on CPU or git, with the command or file given.
- **estimated**: my arithmetic, with the reasoning given.

Nothing here ran on a GPU. Companion files:
- [architecture.md](architecture.md)
- [build-and-kernels.md](build-and-kernels.md)
- [training-recipe.md](training-recipe.md) (other lane, T-xx ids)

## What EAGLE-4 is

EAGLE-4 is an EAGLE-3-style drafter: one decoder layer, autoregressive, fed with fused low, middle and high target hidden states. It adds EAGLE-3.1's norms (paper §2.2, p. 5) and three new pieces (paper §3, pp. 5–6):

1. **EAGLE-Harness** (§3.1, p. 5; Algorithm 1, p. 6). A cost-model-driven dynamic tree.
   - Every decision is one comparison, ΔA/ΔC ≥ τ, made at three points:
     - Draft one level deeper? The threshold rises with depth.
     - How many drafted tokens to verify?
     - How wide may a node branch? Up to 8 when lightly loaded, at most 4 under heavy load.
   - ΔA comes from products of draft probabilities.
   - ΔC comes from profiled latency tables keyed by batch size, context length and token count.
2. **EAGLE Efficient LM Head** (§3.2, pp. 5–6). Two stages:
   - A rank-r low-rank head (r = 128 for this drafter) picks K candidates from the full 248,320 vocab.
   - The full lm_head then rescores only those rows exactly.
3. **Draft attention and cache** (§3.3, p. 6).
   - Sparse (DSA) or sliding-window (SWA) draft attention.
   - Draft tokens are appended to the draft context without evicting old entries during a drafting phase.
   - The draft FFN type follows the target's: dense for dense targets, latent MoE for MoE targets (§3.4, p. 6).

The repo ships only the **inference side**: the `eagle-harness` package, engine plugins, kernels, cost profiles and calibration. It does not ship the training code; see training-recipe.md. The repo describes the package as "EAGLE-Harness: cost-model-driven dynamic-tree speculative decoding (out-of-tree sglang, vLLM and TokenSpeed plugin)" (`pyproject.toml:8`).

## Claims with numbers

All numbers in this section are **paper-reported** unless marked otherwise. The paper reports throughput, not acceptance length. It calls acceptance length "insufficient to evaluate" these methods (p. 7). Every GPU result is on 4×B300 TP4 or MI355X TP1; none is on Ampere.

### Headline claims (abstract, p. 1; Fig. 1)

- Batch size 1:
  - 1.28× DFlash2's throughput.
  - 1.62× MTP's throughput.
- At 512 concurrency:
  - 1.70× DFlash2.
  - 1.99× DSpark.
- The LM head change cuts draft overhead by more than 40%.
- Fig. 1 speedups over vanilla decoding:

  | Platform | Speedup |
  |---|---|
  | 4×B300 | 3.59× |
  | Razr 70 Ultra phone | 3.62× |
  | c5.xlarge CPU | 2.71× |

- Per drafted position (Table 1, p. 3):

  | Drafter | Layers | Params |
  |---|---|---|
  | EAGLE-4 | 1 | 0.44B |
  | MTP | 1 | 0.37B |
  | DFlash2 | 5 | 1.66B |
  | DSpark | 5 | 1.65B |

- EAGLE-4 uses 3.8× fewer FLOPs per drafted token (p. 2).

### Qwen3.8-27B on 4×B300 TP4, SPEED-Bench (Table 2, p. 9; MegaKernel disabled)

Each cell is tok/s, with the speedup over vanilla in brackets.

| T | Conc. | EAGLE-4 | DFlash2 | DSpark | MTP | Vanilla |
|---|---|---|---|---|---|---|
| 0 | 1 | 863.8 (3.59×) | 672.6 (2.80×) | 618.6 (2.57×) | 534.5 (2.22×) | 240.5 |
| 0 | 512 | 28,340 (1.34×) | 16,641 (0.78×) | 14,229 (0.67×) | 21,518 (1.01×) | 21,203 |
| 1 | 1 | 608.6 (2.59×) | 560.3 (2.39×) | 519.4 (2.21×) | 369.1 (1.57×) | 234.9 |
| 1 | 512 | 21,005 (1.03×) | 14,217 (0.70×) | 11,723 (0.58×) | 18,869 (0.93×) | 20,307 |

Configuration (p. 7):
- EAGLE-Harness "12-4-16" (depth-width-verified) at concurrency 1, and "15-1-16", a chain, at 512.
- MTP 7-1-8 at concurrency 1 and 1-1-2 at 512.

Reading:
- At T=1 the margin over DFlash2 at concurrency 1 shrinks to 1.09× (608.6 / 560.3).
- At 512 the gain over vanilla is only 1.03×.

### Concurrency sweep on HumanEval (p. 7, Fig. 3 p. 8)

- Tree for C ∈ {1, 2, 4}, chain above that.
- At C=1: EAGLE-4 929, DFlash2 759, DSpark 701, vanilla 240 tok/s.
- At C=512: 32,935 tok/s, which is:
  - 1.67× DFlash2
  - 1.95× DSpark
  - 1.43× vanilla
- DFlash2 and DSpark fall below vanilla at C256 and C512.

### Nemotron-3.5-Super 120B-A12B (Table 3, p. 9)

| T | C1 | C512 |
|---|---|---|
| 0 | 647 tok/s (2.74×) | 33,480 (1.69×) |
| 1 | 563 (2.40×) | 29,633 (1.51×) |

### MI355X TP1, Qwen3.8-27B, 117-turn SPEED-Bench, tree 12-4-32 (Table 4, p. 8)

| Conc. | Speedup | Detail |
|---|---|---|
| 1 | 3.49× | 75.27 → 262.89 tok/s |
| 2 | 3.32× | |
| 4 | 2.90× | |
| 8 | 2.54× | |
| 16 | 2.30× | |
| 32 | 1.70× | 1390.52 → 2360.83 tok/s |

### Long context, TokenSpeed, single request (Table 7, p. 10)

| Context | Throughput | Round time | TTFT |
|---|---|---|---|
| 32K | 759 tok/s | 5.57 ms | 0.55 s |
| 128K | 822 tok/s | 5.74 ms | 2.1 s |
| 250K | 761 tok/s | 6.04 ms | 5.7 s |

The paper credits the flat round time to SWA draft attention (§3.3.1, p. 6). [review: unsupported as worded. The Table 7 text (p. 10) gives no cause; §3.3 (p. 6) motivates SWA by long-context draft cost. Treat the link as an inference.]

### LM head (§3.2, pp. 5–6; Fig. 2, p. 4)

- Draft forward time and the LM head's share:

  | Head | Draft forward | LM head share | Cut |
  |---|---|---|---|
  | Full | 574 µs | 65.4% | |
  | Rank 512 | 326 µs | 37.9% | −43.2% |
  | Rank 128 | 255 µs | 17.9% | −55.6% |

- 128 candidates contain the full head's top-1 in 98.2% of drafting steps. The SVD initialisation reaches 91.5%.
- The SVD initialisation alone costs 0.48 accepted tokens per verification. The trained head shows "no acceptance-length loss beyond run-to-run noise".
- Draft time per verify cycle at batch size 1 halves, from 9.1 to 4.8 ms.

### CPU and phone (Tables 5–6, pp. 9–10)

These come from llama.cpp on a c5.xlarge with Gemma-4-E2B 4-bit, and from a phone.
- CPU speedups: GSM8K 2.87×, MATH 2.92×, HumanEval 2.76×, MT-Bench 2.28×. EAGLE-3 reached 1.25–1.65×.
- Phone:
  - HumanEval 22.8 → 82.6 tok/s (3.62×).
  - Energy 356 → 166 mJ/token.
  - Draft step 8.7 → 3.9 ms.
  - Resident memory +0.27 GB. [review: the paper also says 0.26 GB on the same page (p. 10); 134 MB of it is the draft model.]
- **Repo check (measured):** neither the llama.cpp nor the phone integration is in the repo. `grep -rli "llama.cpp|gguf|android|razr"` over `*.py` and `*.md` returns nothing. These results cannot be reproduced from the release.

### Repo-reported serving numbers (vLLM and SGLang on B300 and MI355X)

From `README.md:89-93` and `docs/support-matrix.md`:

| Engine | Hardware | Conc. | Baseline → EAGLE-4 (tok/s) | Speedup |
|---|---|---|---|---|
| vLLM | 4×B300 TP4 | 1 | 241.6 → 722.8 | 2.99× |
| vLLM | 4×B300 TP4 | 8 | | 1.70× |
| vLLM | 1×B300 TP1 | 1 | 104.0 → 279.9 | 2.69× |
| vLLM | 1×B300 TP1 | 8 | 715.5 → 1389.1 | 1.94× |
| SGLang | 4×B300 TP4 | 8 | 1479.4 → 2820.6 | 1.91× |
| SGLang | 1×MI355X | 8 | 552.7 → 1145.9 | 2.07× |

- vLLM is behind SGLang: TP1 C1 gives 2.69× on vLLM against 3.33× on SGLang (support-matrix).
- One vLLM high-load tree schedule measured **0.970×** a no-spec control at C64 (support-matrix). A tree can be a net loss under load.

## Supported engines and hardware

- **Engines** (`pyproject.toml:15-22`, `docs/release.env`):
  - SGLang `0.5.23.dev252+eagle.4`. This is the primary engine: live cost calibration, a per-round decision, CUDA graphs and prelaunch.
  - vLLM `0.32.0rc1.dev283+eagle.4`. It needs Model Runner V2, `--attention-backend CUSTOM` and a fixed width per batch-size schedule (`docs/quickstart-vllm.md:56-71`).
  - TokenSpeed `0.1.1+eagle.4`, experimental. Its sampled distribution is not qualified (`docs/known-limits.md`, TokenSpeed section).
  - All three are **eagle-hq forks** (github.com/eagle-hq/{sglang,vllm,tokenspeed}). No upstream release has the hooks: "sglang v0.5.22 and vLLM v0.32.0 are the last without them" (`pyproject.toml:16`).
- **GPUs** (`README.md:190`, `docs/kernels.md:18-24`):
  - NVIDIA sm_80+ and AMD CDNA2–4 through Triton.
  - "The kernel tests run on B300", and other architectures are compile-checked.
  - Serving is qualified only on B300 and MI355X. Ampere consumer cards (sm_86) are not mentioned anywhere. Details are in build-and-kernels.md.
- **Request limits:**
  - Text only.
  - vLLM tree verify "needs an unquantized KV cache" (`README.md:193`, `docs/known-limits.md:48`; enforced at `eagle_harness/vllm/tree_attention.py:69-70`).
  - No pipeline parallelism (`README.md:197`).
  - Grammar-constrained batches on vLLM emit only the root token per round (`docs/known-limits.md`, Requests).
- **Model coverage:**
  - Qwen3.8-27B (hybrid GDN): a dense draft, measured.
  - Nemotron-3.5 (Mamba2 plus MoE): a latent-MoE draft.
  - Adapters exist on SGLang and vLLM (`README.md:184`).

## Release state: v0.1.0 against main

- Tag `v0.1.0` = `7543671` (2026-10-20 12:11 −0500). HEAD = `7b7b5a5`, 17 commits later, the same day.
  - Measured with `git log --oneline v0.1.0..HEAD`.
  - The CHANGELOG has only 0.1.0 (2026-10-20). `pyproject.toml:7` still says `0.1.0`.
- Post-tag changes that matter for us:
  - `6465fb4`: answers vLLM's optional `suspend_drafting` / `verify_budget` hooks. **The 0.1.0 release engine lacks them**: "empty on the 0.1.0 release engine: neither hook below is called there" (`eagle_harness/vllm/speculator.py:69`). On the release image the vLLM width per batch size is therefore not enforced per batch, and native-OFF batch sizes do not run. [review: partly wrong. The release engine still honours native-OFF ranges through `SpeculativeConfig.native_decode_batch_sizes` (`eagle_harness/vllm/config.py:242-245`, comment "the 0.1.0 release engine"; `vllm/batch_policy.py:16-17`), and the schedule still sets width. Only per-batch verify trimming (`verify_budget`) and in-step drafting suspension (`suspend_drafting`) are missing.]
  - `e9baddd`: refuses chain widths and tree rounds measured on another setup. Cost data is keyed to GPU, TP, config and dtype.
  - `f6149b0`: an fp32 arbiter for greedy-parity divergences above the near-tie gap.
  - `7b7b5a5`: removes the `eval`, `ragged` and `moe_gate_acc` options. `docs/migration.md` lists them as "Removed in 0.1.1".
- Published artefacts (`docs/release.env`):
  - Wheels at `eagle-hq.github.io/EAGLE-4/whl/cu130/`, and PyPI per `1c4cb1c`.
  - Images `ghcr.io/eagle-hq/eagle-4-{sglang,vllm,tokenspeed}:0.1.0`. The ROCm image is unpublished (`dd83ac3`).
  - The vLLM Dockerfile says "Sketch only. Base and final image need clean-build qualification" (`docker/vllm.Dockerfile:5`).
- HF drafter `EAGLE-Corp/Qwen3.8-27B-EAGLE4`:
  - Created 2026-10-07, modified 2026-10-19. Not gated. Licence "other" (eagle-4). 31 downloads (measured, HF API 2026-10-21).
  - One `model.safetensors` of 6,178,152,384 bytes (measured from the API listing).

## Licence terms that affect adoption

The source is `LICENSE`, the "Eagle 4 Source-Available License". This is a summary, not legal advice.

- **Acceptance by conduct** (preamble `LICENSE:8-13`; "Acceptance and Repository Access" `LICENSE:15-33`).
  - "By downloading, cloning, forking, copying, accessing, modifying, using, hosting, distributing, or otherwise exercising any right in the Software, Licensee accepts."
  - The clone in `~/research-src/EAGLE-4` has therefore already accepted it for whoever made it.
- **Weights are covered.** "Software" includes "model architecture, model weights and parameters, source code … documentation" (§1(j), `LICENSE:90-93`). The HF card points to the same licence.
- **The technique itself is covered, not only the code.**
  - "Eagle 4" means "the Software and the technology embodied in it, including EAGLE-Harness and EAGLE Efficient LM Head technology, but excluding independently developed technology that merely interoperates" (§1(e), `LICENSE:73-75`).
  - "Eagle Files" are files that "contain, reproduce, modify, implement, port, or are derived from any portion of the Software" (§1(f), `LICENSE:77-80`).
- **Free uses** (§2(a), `LICENSE:97-104`): research, evaluation, education, personal use, internal noncommercial experimentation, academic and nonprofit research, and "other noncommercial purposes".
  - §1(c), `LICENSE:55-59`: genuine internal evaluation is not Commercial Use while it is not customer-facing, earns no revenue and "does not materially support a revenue-generating product or service".
- **Competing Offering** (§1(d), `LICENSE:61-71`; §3(a)–(b), `LICENSE:122-131`).
  - A Competing Offering is a product or service that substitutes for Eagle 4 or "incorporates material Eagle 4 technology into a model, inference, training, acceleration, or serving framework that competes with Eagle 4".
  - The licensee "may not develop, offer, sell, license, host, provide, or otherwise commercialize a Competing Offering … regardless of Aggregate Revenue" (§3(a)).
  - The licensee may not evade this "by copying, forking, renaming, … rewriting, porting … or by placing material Eagle 4 technology in another framework" (§3(b)).
  - Excluded (`LICENSE:67-71`): a product that merely interoperates with Eagle 4, uses its outputs, or uses it in an incidental or immaterial way.
- **Revenue cap** (§3(c)–(f), `LICENSE:133-159`).
  - Commercial use that is not a Competing Offering is allowed only while the licensee's and its affiliates' Aggregate Revenue has not exceeded **US$20 million over any consecutive 12 months**.
  - Once the cap is exceeded, a separate agreement is needed first. There is no grace period (§3(d)). Splitting revenue among affiliates or others to avoid the cap is barred (§3(f)).
- **Distribution conditions** (§4, `LICENSE:161-226`):
  - Ship `LICENSE` and `NOTICE`, and keep the source-file notices (§4(a)).
  - Mark every materially modified file with what changed, when and by whom (§4(b)).
  - Make the complete corresponding source available under this licence for any distributed Modification, including "patches, or weight deltas" (§4(c)).
  - Every public repository or description of a derivative must state: "This project includes or is derived from Eagle 4 technology developed by Eagle AI Corp…" (§4(d)).
  - Ported code keeps its obligations wherever it moves (§4(g), `LICENSE:216-221`). Only a file the owner expressly designates as Apache-2.0 is Apache (§1(f), §4(h)).
- **Trademarks** (§5, `LICENSE:228-252`): a distributed fork may not use "Eagle" or "Eagle 4" as its primary name (§5(d)).
- **Patents** (§6(a), `LICENSE:254-262`).
  - The patent licence covers only claims "necessarily infringed by the unmodified Software" when used as the licence permits.
  - **An independent re-implementation of the harness or the low-rank head gets no patent grant from this licence.** §6(b) ends the patent grant for anyone who sues over it.
- **Records** (§8(a)): keep compliance records for three years after each use or distribution.
- **Termination** (§9): 30 days to cure. Immediate termination is possible for wilful circumvention of §3.
  - On termination, copies must be deleted within 30 days (§9(b), `LICENSE:305-320`).
  - Quirk: §9(b) lists "Sections 1, 5, 7, 8, 9, 10, 11, and 12" as surviving, but `LICENSE` has no Section 12.
- **Law** (§11(a)): New York law, New York County courts.

What this means for this repo (estimated, not legal advice):
- Our repo is Apache-2.0 (`LICENSE:1-3`).
- **Local, personal or evaluation use** of the EAGLE-4 drafter and plugin on a 3090 falls under §2(a).
- **Vendoring or porting `eagle_harness` files** into this Apache-2.0 public repo would make them Eagle Files inside our tree. Each would need the notices, modification records and the attribution statement (§4). They could not be relicensed.
- **Re-implementing the Efficient LM Head or the cost-driven tree in our own code** is still "Eagle 4 technology" under §1(e).
  - Whether a free public serving stack that does this is a "Competing Offering" that one may not "develop" (§3(a)) is ambiguous.
  - There is no patent grant for a re-implementation (§6(a)).
  - The maintainers should decide this before E-03, E-04 or E-05 below land in a public branch.
- The generic ideas the paper credits to earlier work are not Eagle 4 technology as such: EAGLE-2 dynamic trees, EAGLE-3 multi-layer fusion, SWA and DSA. But §1(e) does not draw the line, and §3(b)'s "rewriting" clause leans against the reader.

## Candidate techniques

These are inference and build items. Training items are T-xx in training-recipe.md. The legal review in E-01 gates E-03 to E-05.

- **E-01** Legal gate: before porting harness or low-rank-head ideas into this public Apache-2.0 repo, decide the §1(e)/§3(a)–(b)/§6(a) question. A local-only evaluation stays inside §2(a).
- **E-02** Serve the released Qwen3.8-27B EAGLE-4 drafter as-is for an offline acceptance baseline on our prompts. This is evaluation use under §2(a). Mechanics are in build-and-kernels.md, E-20.
- **E-03** Efficient LM Head on our own drafter head (MTP or DFlash2): low-rank top-K retrieval, then exact rescoring of K rows. Details in architecture.md, E-10.
- **E-04** Cost-table-driven draft depth and verify width per batch size and context. Overlaps deepinfer spec-0, spec-1 and spec-2. Details in architecture.md, E-12 and E-13.
- **E-05** Cost-model-sized tree verify on the GDN target. Details in architecture.md, E-14.
