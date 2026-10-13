# E7: draft-head quantisation

Not run. No image was built for it. This file records why, from evidence gathered without a GPU boot.

## What the arm assumes

`docs/decode-speed-landscape.md` G12 and `docs/decode-speed-experiments.md` E7 describe "the draft LM head is
read at full width every verify step". The proposed B arm quantises only the draft head and leaves the target
head unchanged.

## What is on disk and in the image

`raw-head-inspect.log` reads the safetensors headers in `G:/models` (read-only, host Python):

```
Qwen3.8-27B-DFlash2-W4A16/model.safetensors tensors: 153
tensors whose name contains head/embed/vocab: []
drafter config: {'architectures': ['DFlash2DraftModel'], 'tie_word_embeddings': False, 'vocab_size': 248320, 'draft_vocab_size': None}
Qwen3.8-27B-W4A16-AutoRound-fast head tensors: ['lm_head.weight_packed', 'lm_head.weight_scale', 'lm_head.weight_shape', 'mtp.draft_lm_head.weight_packed', ...]
```

`raw-dflash2-head-source.log` covers the image's vLLM 0.31.0 source.
`vllm/model_executor/models/qwen3_dflash2.py:419` computes the drafter's candidates through `self.lm_head`.
`vllm/v1/spec_decode/llm_base_proposer.py` `_maybe_share_lm_head` replaces a draft model's head with
`target_language_model.lm_head` when the checkpoint has none of its own. The shipped boot's verify gate prints
`PASS lm_head requantized to int4, packed geometry matches the declared width`.

## Finding

The DFlash2 drafter checkpoint carries no LM head and no embedding of its own. The head it reads is the
target's, and the target's is already int4-packed (`lm_head.weight_packed`). This checkpoint has no separate
full-width draft head to quantise. "Quantise the draft head, keep the target head" has no B side here without
adding a new draft-head artifact under `prepare/`. Hard rule 6 rules that out ("the only file you may change
outside bench/results/ and docs/ is the launcher knob permitted by E1").

`mtp.draft_lm_head` in the target checkpoint is the MTP drafter's head (`SPEC=mtp`), not DFlash2's. It is
already packed too.

**Verdict: not applicable on this checkpoint. A B arm would need a prepare change, which this run may not
make.**

## Verified vs assumed

Verified: the tensor lists, the drafter config, the source lines quoted, and the verify-gate line.

Not verified: that the shipped boot actually took the sharing branch. The boot log has no `Sharing target
model lm_head` line (`grep` returned nothing), so the sharing is inferred from the absent head tensors and the
source. The 5.80 tokens per step measured in E3 would be implausible with an unloaded head, but that is an
inference, not a check.
