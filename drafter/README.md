# drafter/ — self-distillation data, calibrated int4 requant, MTP fine-tuning, DFlash2 requant

Tooling used to build the single-user "fast" variant of the model
(`models/Qwen3.8-27B-W4A16-AutoRound-fast`, prebuilt on the Hub as
`syvai/qwen3.8-27b-3090-fast-variant`; `prepare/fetch_fast_variant.py` assembles it). It also
contains a complete MTP-head fine-tuning pipeline that, honestly, did **not** move the
needle — kept because the negative result is informative and the same data feeds the
things that did work.

Everything here runs on the 3090 in the serving venv; ~6 h of GPU time end to end.

`gen_data.py` resumes complete records in `data/gen.jsonl`. It removes an interrupted
final record before appending and adds a missing final newline to complete records;
corruption earlier in the file stops the run instead of being silently skipped.
The stdlib-only recovery helper is `generation_state.py`.
A nonblocking OS lock on `data/gen.jsonl.lock` excludes a second generator before
resume repair or model loading, and stays held through all writes. The sidecar
file remains, but a process exit releases its lock automatically. Do not delete
the sidecar while a generator is running.

## What actually mattered (in order)

1. **The draft-head vocabulary.** The MTP drafter scores a 40,960-row slice of `lm_head`
   (`prepare/build_draft_vocab.py`); a token outside that slice can never be drafted, so every
   such token is a guaranteed rejection *and* truncates the chain. The originally shipped id
   list (counted over Danish web text, Wikipedia, Python, and 8.8M tokens of older outputs)
   covered 92.1% of what the model actually generates — 83% on code. A list counted over
   5.4M tokens of the model's own outputs (this pipeline, `gen_data.py`) covers 97.5%
   (96% on code). Nothing else changed: 98.0 → 108.6 tok/s greedy, 90.0 → 107.4 at
   default sampling. Coverage barely improves past 40k rows (49k: 98.2%; the model only
   ever emits ~54k distinct tokens), and 49k measured no faster.
2. **GPTQ-calibrated int4 for lm_head and the MTP module.** Round-to-nearest int4 costs
   +1.5% perplexity on lm_head (KL to the bf16 head 0.0068) and ~2% acceptance on the
   MTP module. GPTQ with a Hessian from 300k captured hidden states (`gptq_lm_head.py`)
   halves the lm_head KL (0.0029, +0.6% PPL, GSM8K 96.5% unchanged) and the MTP
   Hessians from `train_mtp.py --dump-hessians` keep acceptance intact
   (`requant_mtp_gptq.py`). Together: −1.8 ms per decode step (108.6 → 118.8 greedy).
3. **Fine-tuning the MTP head: no.** Distilling the target's own distribution into the
   drafter (KL over the draft vocab, unrolled depth-2 chains, 7M tokens, one epoch)
   halves the KL and looks great on the naive metric — until you score only response
   tokens against the *actual* next token, where top-1 agreement is unchanged
   (0.685 → 0.685) and vLLM's acceptance moves within noise. Qwen's head is already at the
   ceiling of a single-layer chain drafter for greedy top-1 on this data; the KL gains
   were on prompt tokens and on positions whose true token is outside the draft vocab.
   `train_mtp.py --eval-only` with `--depths 4` prints the greedy chain simulation that
   matches vLLM (2.5 vs 2.6 tok/step for the original head).

## Pipeline

```bash
V=venv/bin/python
$V drafter/collect_prompts.py                 # 6.8k prompts: UltraChat, Magicoder, syvai/da-instruction,
                                              #   syvai/reasoning-v1, skolegpt-instruct, GSM8K; 45% thinking on
VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_MARLIN_INT8_INCLUDE_RE=mlp $V drafter/gen_data.py   # 2.2 h, 5.4M output tokens
$V drafter/capture.py                         # 1.7 h: hidden states of every token (74 GB memmap), in-process
                                              #   vLLM hook on GPUModelRunner._model_forward
# draft vocab from the model's own outputs -> prepare/draft_vocab_ids.json (the shipped list)
$V drafter/train_mtp.py --out drafter/runs/e --eval-only 1 --draft-ids prepare/draft_vocab_ids.json \
     --max-seqs 400 --val-frac 0.4 --depths 2 --dump-hessians drafter/runs/e/mtp_hessians.pt
$V drafter/gptq_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound models/tmp-lm4 --bits 4 --calib-rows 300000
$V prepare/build_draft_vocab.py models/tmp-lm4 --ids prepare/draft_vocab_ids.json   # int4 draft head from the int4 lm_head
$V drafter/requant_mtp_gptq.py models/tmp-lm4 models/Qwen3.8-27B-W4A16-AutoRound-fast drafter/runs/e/mtp_hessians.pt --bits 4
```

Optional fine-tune (for the record): `train_mtp.py --out runs/r --depths 2 --depth-weights 1,0.5
--epochs 1 --lr 3e-5 --micro-tokens 4096` (~30 min/epoch at 4k tok/s), then `export_mtp.py`.
The trainer reproduces vLLM's drafter to 1% (checked by replaying captured drafter calls
with their KV history) so its numbers are trustworthy; use `--eval-only` first, response
tokens only, true-token criterion.

## DFlash2 drafter: W4A16 requantization

`SPEC=dflash2` single-user mode uses [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
(5 Qwen3-style layers, hidden 5120, 8 KV heads × 128, MLP 17408, an `fc` that projects the
target's layer 5/19/33/47/61 hidden states, dynamic convs, a candidate selector; 1.92B
params, 3.85 GB bf16). Read once per decode step that is ~5 ms on a 3090 and a 21k-token
KV pool, so it ships requantized to W4A16 compressed-tensors (Marlin), 1.19 GB:
[syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)
(`prepare/fetch_dflash2.py`). To rebuild it:

```bash
V=venv/bin/python
$V prepare/fetch_dflash2.py --bf16                               # models/Qwen3.8-27B-DFlash2 (3.85 GB)
# 1. Hessians from the drafter's OWN inputs: vLLM in-process, eager (hooks), the bf16 drafter
#    speculating on 400 prompts of data/gen.jsonl at model-default sampling, ~20 min;
#    hooks on qkv_proj / o_proj / gate_up_proj (GPU fp32 Hessians), down_proj / fc (rows
#    dumped to memmaps, reduced on the GPU in a re-exec'd process), plus the input of the
#    fused context-KV precompute (the k/v rows are applied to those too). ~56 GB of scratch.
DRAFT=models/Qwen3.8-27B-DFlash2 $V drafter/capture_dflash2.py --prompts 400 --max-tokens 384
# 2. GPTQ int4 g128 for the 35 layer matrices + fc (k/v blend the context Hessian in),
#    compressed-tensors export with the vLLM-prefix ignore list (~40 s, GPU must be free:
#    the fc Hessian is 25600^2)
$V drafter/quant_dflash2.py models/Qwen3.8-27B-DFlash2 models/Qwen3.8-27B-DFlash2-W4A16 drafter/runs/dflash2/hessians.pt
```

Research options (off by default; the commands above rebuild the shipped drafter):

- `quant_dflash2.py ... --q3` (= `--signed-scale --mse-clip --act-order`): signed ±amax/8
  scales, MSE clip and act-order with static groups in `gptq_utils.gptq_quantize`
  (quant-3 in `docs/research/deepinfer/quant-heads.md`; −50% synthetic output error,
  `bench/test_gptq_q3.py`). Same stored layout, no `g_idx`; scales may be negative.
- `train_dflash2_tv.py`: TV-loss repair of the W4A16 drafter's bf16 tensors (convs,
  selector, norms; int4 linears frozen) against the bf16 drafter on a drafter-2 capture
  (`capture_dflash2.py --dump-target`, forward in `dflash2_ref.py`); drafter-3 in
  `docs/research/deepinfer/drafter.md`. `--smoke` runs a CPU self-test.

What the measurements said (8 realistic prompts × 1,024 tokens, fast-variant target):

- int4 GPTQ keeps greedy acceptance (3.34-3.65 vs 3.54 tokens per step for bf16) and loses
  ~5% at the model's default sampling (3.2 vs 3.4): noise in q hurts the acceptance
  *probability*, not the argmax. Per step it reads 2.7 GB less (31.4 → 28 ms with the base
  target, 26.5 ms with the fast variant), which is what turns DFlash2 from a wash into a
  win on this card.
- `fc` in bf16 instead of int4 (+0.26 GB): no acceptance difference (3.17 vs 3.17).
- Blending the context-KV input distribution into the k/v Hessians (the k/v rows are also
  applied to the context rows by the fused precompute, so on paper this is the right
  calibration): **7% worse** greedy acceptance, 3.34 → 3.12 tokens per step, 126 → 118 tok/s.
  It looked equal on a single default-sampling run, which is how it nearly shipped; greedy is
  the reproducible signal here (four repeats land within 1.5 tok/s, step counts identical).
  `quant_dflash2.py` still supports it — pass a Hessian file containing `ctx_kv` — but the
  shipped drafter does not use it.
- Applying the request's top-k/top-p to the selector walk's 16-candidate proposal (cached
  truncated, so the verify stays lossless — the DFlash2 analogue of the MTP draft
  truncation): +2%, inside the noise; on by default (`VLLM_DFLASH2_DRAFT_TOPK_TOPP=0` off).
- Relative weight error of the int4 matrices: 0.147 mean (Frobenius), like the MTP module.

## DFlash2 offline gate (`eval_dflash2.py`)

Drafter A/Bs on 8 prompts × 1k tokens sit inside ±3% (greedy) and ±5% (default sampling)
run noise. `eval_dflash2.py` scores a drafter offline instead, deterministically: one target
capture, then any number of drafter checkpoints (bf16, the W4A16 requant, a re-requant, a
repaired one) replayed through a pure-torch DFlash2 forward (`dflash2_ref.py`, a copy of
vLLM 0.31's `qwen3_dflash{,2}.py` and the selector walk; the W4A16 tensors are dequantized).

```bash
# 1. capture (GPU, target only, ~10 GB of tap rows): reference responses, greedy and default
#    sampling, then one prefill per sequence dumping the layer 5/19/33/47/61 hidden states
#    and the target's top-64 logits for the response and the last 2,047 prompt rows
MODEL=models/Qwen3.8-27B-W4A16-AutoRound-fast $V drafter/capture_dflash2.py --dump-target drafter/runs/d2cap
# 2. score (GPU, after the target is gone; ~9 GB for the drafter plus the dequantized heads)
$V drafter/eval_dflash2.py --capture drafter/runs/d2cap --drafter bf16=models/Qwen3.8-27B-DFlash2 \
    --drafter w4a16=models/Qwen3.8-27B-DFlash2-W4A16 --out drafter/runs/d2cap-eval.json [--check-known]
$V drafter/eval_dflash2.py --selftest      # CPU: tiny synthetic drafter, target and capture
```

Per anchor it computes the acceptance probability of each draft slot against the target
(Σ min(p, q) at the request's sampling, argmax match at greedy; teacher-forced on the
reference) and reports per domain × mode `tau_chain`, the expected tokens per verify step of
the speculative-decoding renewal process (comparable to vLLM's tokens/step), DeepSpec's
`tau_anchor`, accept@position and a bootstrap CI, plus the paired ratio of every drafter
against the first. `--check-known` holds bf16 and W4A16 against the numbers above (3.4 /
3.2 sampling, 3.54 / 3.34–3.65 greedy, ±2%) and runs a negative control. Exit codes follow
`bench/README.md`.

## Notes that cost time

- `capture.py` aligns hidden states by vLLM's request ids, which are `"<counter>-<uuid>"`
  in 0.27, and by `input_batch.req_ids` order within a step.
- Decode-time hidden states differ from prefill ones by ~0.9% (fp16 recurrent state);
  training on either gives the same drafter.
- Positions right after a rejection are systematically harder: vLLM's per-position
  acceptance is measured there, so it sits ~5 points below a whole-sequence top-1 rate.
  The chain simulation in `train_mtp.py --eval-only` accounts for that.
- Greedy decoding with speculation is not bit-deterministic across drafter configs
  (verify batches of 5 vs 1 token round differently), so 8 prompts × 1k tokens has a
  ±3% spread on tokens/step. Repeat before believing a 2% difference. With DFlash2 the
  greedy spread is wider (3.1-3.6 tokens per step across launch configs), default sampling
  ±5% per run.
- `capture_dflash2.py`: an in-process vLLM engine does not give its GPU memory back on
  `del llm`; the Hessian reduction re-execs the process. The fused `qkv_proj.weight_shape`
  parameter only holds the last-loaded shard's shape — derive the dense shape from
  `weight_packed`/`input_size` (the backport's `_dense_kv_rows` does).
