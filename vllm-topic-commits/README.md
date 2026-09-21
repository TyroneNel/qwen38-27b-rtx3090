# The five [qwen38] topic commits — the vLLM serve-fixes

These are the source commits the five new `patches/series` entries on `main`
were exported from (findings 2–6 of the tokenize/404/auth review). They were
cut on top of `cpuchip/vllm` branch `qwen38/0.28` (vLLM v0.28.0 + the
HyperQwen patch series), whose tip is `02f1236`:

| commit | subject | exported as |
|---|---|---|
| `813321b` | `[qwen38] bench-probe-errors` | `patches/bench-probe-errors.patch` |
| `1c5d490` | `[qwen38] serve-404-served-names` | `patches/serve-404-served-names.patch` |
| `1a60b33` | `[qwen38] serve-model-path-match` | `patches/serve-model-path-match.patch` |
| `95147c3` | `[qwen38] tokenize-v1-route` | `patches/tokenize-v1-route.patch` |
| `5dc5131` | `[qwen38] auth-deny-default` | `patches/auth-deny-default.patch` |

Those SHAs are the ones the `--- exported from cpuchip/vllm <sha> ---`
provenance lines in the patch files reference.

This branch is a transport branch — it is **not** merged to `main` (main
already carries the applied form: the five `patches/*.patch` files, their
`patches/series` rows and their `PATCHES.md` rows).

## Getting the commits back out

**1. `git am` — recreates the commits anywhere (new SHAs, identical content,
messages and authorship):**

```bash
git am 00*.patch
```

- On `cpuchip/vllm` `qwen38/0.28` they apply as-is (that is where they were cut).
- On pristine `vllm-project/vllm` `v0.28.0` every hunk applies too — the only
  file shared with an earlier series patch is `entrypoints/openai/cli_args.py`
  (auth-deny-default edits its `--api-key` help text ~150 lines away from
  sse-keep-alive's region, so the context is identical either way).
- On current `vllm-project/vllm` `main`, re-check the context first: these
  entrypoint files have been moving between releases.

**2. The bundle — preserves the exact SHAs** (`qwen38-serve-fixes.bundle`,
6.8 KB, thin: its basis is `cpuchip/vllm`'s public `02f1236`):

```bash
git clone --branch qwen38/0.28 https://github.com/cpuchip/vllm.git vllm
cd vllm
git fetch /path/to/qwen38-serve-fixes.bundle \
    refs/heads/qwen38/0.28:refs/heads/qwen38-serve-fixes
git log --oneline -6 qwen38-serve-fixes   # 5dc5131 and the four below it
```

**3. Already applied in this repo:** `main` applies the five patches to the
installed vLLM wheel at image build (Dockerfile reads `patches/series`), and
`patches/check_vllm_series.sh` validates the whole series against pristine
`vllm-project/vllm` `v0.28.0`.
