# evals-2: hot-path GPU sync check per mode

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Commit | in this result's commit: `bench/sync_check.sh`, `bench/sync_allowlist.txt`, `bench/test_sync_check.sh`, `experiments/patches/evals2-sync-negctl.patch` |
| Image | stock `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`) for the six modes; `hq:exp-negctl` (stock + `evals2-sync-negctl.patch`) for the negative control |
| GPU | RTX 3090, 250 W, driver 610.88 |
| `.env` knobs | per mode as baseline.md, plus `VLLM_GPU_SYNC_CHECK=warn`; the control adds `DI_SYNC_NEGCTL=1` |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\probes\<mode>\sync_check.txt`, `…\evals-2\negctl_B\` |

## Method

vLLM 0.31 arms torch's sync-debug mode around `execute_model` and `sample_tokens` once warmup is done, and logs every synchronizing call and every implicit pageable `non_blocking` copy with its file:line. `bench/sync_check.sh` drives `api_smoke.py` (13 features, including a 20k prompt and a streamed tool call) plus the 8 real prompts × 256 tokens through the server. It then collects the unique sites from the server log and fails on any that is not in `bench/sync_allowlist.txt`.

```bash
bash bench/sync_check.sh /out/container.log --mode B
```

**Negative control.** `evals2-sync-negctl.patch` adds one `.item()` to `sample_tokens` under `DI_SYNC_NEGCTL=1`. The check names it: `vllm/v1/worker/gpu/model_runner.py:2093 sync x1032`, FAIL. So a zero below means zero, not "not armed".

## Numbers

| Mode | Sites | Warnings | Verdict |
|---|---|---|---|
| A (batch) | 0 | 0 | PASS |
| B (DFlash2, `CTX=fast`) | 0 | 0 | PASS |
| C (B, k=15, lookup long block) | 0 | 0 | PASS |
| D (MTP, pthfa) | 0 | 0 | PASS |
| D′ (DFlash2, pthfa) | 0 | 0 | PASS |
| **E (KVarN)** | **13** | **10,074** | **FAIL** |
| negative control (B + injected `.item()`) | 1 | 1,032 | FAIL, as designed |

The 13 E sites, all in `vllm/v1/attention/backends/kvarn_attn.py` (the KVarN backend; `kvarn/` in the repo):

| Line | Calls | What |
|---|---|---|
| 665 | 2,292 | `cam.seq_lens.tolist()`: the builder's per-step CPU copy of the sequence lengths (the runner's cached CPU copy is absent on this path) |
| 683 | 2,292 | `cam.block_table_tensor.cpu().numpy()` |
| 684 | 2,292 | `cam.slot_mapping.tolist()` |
| 2129, 2130 | 816 each | `torch.as_tensor(slots/bids, device=dev)`: H2D per tail flush (already batched once per flush, "WSL fix (PR #16)") |
| 981, 939, 947, 607, 611, 822, 913 | 4–524 | scalar writes into device tensors (`b2s_t[bid] = …`, `is_sink_t[bid] = …`) in the sink, recycle and flush bookkeeping |
| 2721 | 432 | `int(cu_k[-1].item())` in `_decode_path` |

## Reading

- **Five of six modes have zero hot-path syncs on the 0.31 pin.** The allowlist stays empty for them; `bench/sync_allowlist.txt` is committed empty, so any new site in A–D′ fails.
- **E's KVarN backend syncs three times per step in metadata build**, plus once per tail flush and per bookkeeping write. The cost shows up as host time between steps: 5.7 ms of a ~150 ms step at ~170k (e-profile). At shallow depth the same fixed cost is a larger share; evals-8 measures it at 20k.
- These are known design choices in KVarN (its comments note the trade-offs), not regressions. They are recorded here rather than allowlisted, because removing them is a KVarN change (device-side metadata) that belongs with attn-2.

## Verdict

INFO. The check works: the negative control is caught, and five modes are clean. Mode E has 13 per-step or per-flush sync sites, all in KVarN's metadata builder and decode path.
