# attn-4: training-free top-k sparse attention (offline study, go/no-go)

| Field | Value |
|---|---|
| Date | 2026-10-23 17:16–17:56 SAST |
| Commit | `bench/attn4_dump.py` (GPU dump), `bench/attn4_mass.py` (CPU analysis, `--self-test` passes) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`sha256:0cf71b42d301…`, main `ce9615c`), stock |
| Dump | Mode D's KV setup (int8 per-token-head, pthfa, no drafter, prefix cache off), one 128,044-token cold prefill (repo docs plus a question) and 64 greedy decode tokens. Post-norm, post-RoPE K for every position, captured before the int8 cache write; Q for the last 32 prompt positions, all 64 decode positions and 16 depth-spread positions; all 16 full-attention layers. Consistency check passed (positions = prompt + gen − 1). |
| Analysis | fp32 softmax over every causal key. Page sizes 16 and 64; budget k = max(2, ⌈pct × visible pages⌉) for 2, 5 and 10%; sink page and own page always included. Selectors: oracle (exact page mass) and bounds (per-page, per-channel K min/max upper bound, Quest-style); granularity per KV head (the 6 query heads of a KV head share one page set, as a kernel would) and per query head. |
| Gate (appendix) | captured attention mass ≥ 0.99 (mean over tail queries) in all 16 layers, bounds selector, per KV head |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\attn-4\{dump,mass}\` (JSON `attn4_mass.json`) |

## Captured attention mass, tail queries, per KV head (bounds selector, oracle in brackets)

| Layer | P16 2% | P16 5% | P16 10% | P64 10% |
|---|---|---|---|---|
| best (L19) | 0.735 (0.832) | 0.798 (0.882) | 0.849 (0.919) | 0.813 (0.896) |
| typical (L47) | 0.606 (0.807) | 0.714 (0.875) | 0.796 (0.922) | 0.728 (0.891) |
| worst (L27) | 0.143 (0.348) | 0.227 (0.487) | 0.340 (0.608) | 0.265 (0.582) |

Over depth-spread queries, the worst layer captures 0.10 / 0.18 / 0.28 of the mass at P16 with 2 / 5 / 10%.

## Verdict: NO-GO (closed)

Attention at 128k in this model is diffuse. Even exact (oracle) page selection keeps at most 0.94 of the mass with 10% of pages in the best layer and 0.61 in the worst, and the training-free bounds selector does worse still. No tested budget meets the 0.99 gate in any layer, so a kernel is not worth building. DeepSeek's DSA works with a trained indexer and dense-to-sparse training (the appendix estimates the warm-up alone at ~21 days on one 3090), which is out of scope here.
