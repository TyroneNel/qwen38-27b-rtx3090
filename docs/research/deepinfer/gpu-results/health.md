# health: `make verify` and the repo's bench tests

| Field | Value |
|---|---|
| Date | 2026-10-20 |
| Commit | `88549e7` (research/deepinfer after merging main `63f7581`) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` = `sha256:c696d22e80b4…` (main `2cf50a0`), stock |
| GPU | RTX 3090, 250 W enforced, driver 610.88, idle 46 °C (tests are short; clocks not tracked per test) |
| `.env` knobs | box `.env` (`PREPARE=0`, `VERIFY=0`, GPU pinned by UUID, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`) |
| Context | n/a |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\health\` |

## Commands

```bash
# WSL, repo root. MODELS_DIR must be the WSL spelling from WSL (the .env value G:/models is rejected there).
MODELS_DIR=/mnt/g/models make verify
# every bench/ test the bench README lists as CPU / image / torch / GPU, plus kvarn's torch test,
# inside the stock image on GPU 0 with the repo's bench/ bind-mounted:
docker run --rm --gpus device=<3090 UUID> --ipc host -v $REPO/bench:/app/bench -v /mnt/g/models:/app/models:ro \
  -v $OUT:/out -v $TOOLS:/tools:ro --entrypoint bash ghcr.io/syv-ai/hyperqwen:latest /tools/health_tests.sh
```

`health_tests.sh` is in `G:\dev\hq-results\deepinfer-20261020\tools\`.

## Results

`make verify` (no server, `docker compose run --rm single verify`): **OK, 0 failures**. Warnings: no API key in `.env`, and live-server checks skipped (no server). The live section is re-run with a server up at the end of each phase.

| Test | Result |
|---|---|
| `test_lookup_kernels.py`, `test_prefill_attn_bigpool.py`, `test_spec_decode_attn.py`, `test_pthfa_attn.py`, `mq3d_layer2_oracle.py` (GPU) | pass |
| `test_spec_decode_bigpool.py`, `test_spec_decode_fp8.py` (GPU) | designed SKIP (fp8 path needs sm89+) |
| `test_kv_cache_seat_budget.py`, `test_kvarn_recycled_pages.py`, `mq3d_scratch_pool_test.py`, `mq3d_capacity_property.py`, `test_bench_sse_keepalive.py`, `test_conc_ladder_salt.py`, `test_harness.py`, `test_labd_bench_clock.py`, `test_model_verification.py`, `test_prepare_state.py`, `test_prepare_crash.py`, `test_quant_schema.py`, `test_t4_battery.py`, `verbatim.py`, `kvarn/tests/test_kvarn_fp16_dequant_torch.py` | pass |
| `test_bool_knobs.sh`, `test_dynamic_k_knob.sh`, `test_fa2_fp8kv_knob.sh`, `test_kv_offload_knob.sh`, `test_pthfa_knob.sh`, `test_resident_print.sh`, `test_tp_size.sh`, `test_pthfa_install_toolchain.sh` | pass |
| `test_no_key_bind.sh` | designed skip under `/.dockerenv` |
| **`test_marlin_int8_asym.py`** | **FAIL (exit 1), test bug:** `MPLinearLayerConfig.__init__() got an unexpected keyword argument 'has_g_idx'`. vLLM 0.31.0 dropped that field; the test was not updated for the 0.31 pin. The kernel is not exercised, so this says nothing about Marlin W4A8. |
| **`test_prefix_cache_knob.sh`** | **FAIL (1 of 47 checks), environment:** `PRINT_ARGV=1: first stdout line` expects `venv/bin/vllm` but gets the launcher's WSL banner (`WSL detected: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False …`). Docker Desktop containers share the WSL kernel, so the launcher's WSL detection fires inside the container. Every other check passes; the test should skip the banner or the banner should go to stderr. |

## Verdict

INFO. The stack is healthy for measurement: every kernel test that runs on sm86 passes. The two failures are a stale test (`test_marlin_int8_asym.py` on the 0.31 API) and a WSL-only output-order assertion; neither touches the serving path. Both are left for the user (they are repo bugs outside this expedition's scope).
