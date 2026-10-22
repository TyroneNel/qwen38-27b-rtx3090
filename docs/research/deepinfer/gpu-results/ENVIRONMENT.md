# GPU box environment

Recorded 2026-10-20 at the start of the run (step 0 of [../gpu-agent-prompt.md](../gpu-agent-prompt.md)).

## Hardware and drivers

| What | Value |
|---|---|
| GPU 0 (all runs) | NVIDIA GeForce RTX 3090 24 GB, sm86, PCI 01:00.0, UUID `GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84`, VBIOS 94.02.42.00.a7, WDDM LUID `0x1144f` until the 2026-10-22 restart, `0x11914` after it |
| GPU 1 (not used) | NVIDIA GeForce RTX 3080 Ti 12 GB, PCI 03:00.0, drives the display (~435 MiB used at idle). Every container is pinned to GPU 0 by UUID (`CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` in `.env`; compose `count: 1` exposes GPU 0). |
| Power limit, GPU 0 | 250 W enforced (default 350 W, max 365 W), already set when the run started |
| Clocks, GPU 0 | max SM 2100 MHz; idle 210 MHz, 46 °C |
| Driver | 610.88 (Windows host, WSL2 paravirt) |
| OS | Windows 10 19045, WSL2 kernel 6.18.54.1-microsoft-standard-WSL2, Ubuntu distro |
| WSL VM | 16 vCPU, 23 GB RAM, 24 GB swap (`%USERPROFILE%\.wslconfig`) |
| Docker | Docker Desktop, Engine 29.8.2 (WSL2 backend, NVIDIA runtime) |
| Disk | `G:` 466 GB, 130 GB free at start |

## Stack

| What | Value |
|---|---|
| Repo | `G:\dev\HyperQwen`, branch `research/deepinfer`; git: Windows git 2.55 (`C:\Program Files\Git`), identity of the box |
| Stock image | `ghcr.io/syv-ai/hyperqwen:latest` = `sha256:c696d22e80b4…`, built from `main` `2cf50a0`; also tagged `ghcr.io/syv-ai/hyperqwen:sha-2cf50a0`, which `docker-compose.override.yml` (box-local, untracked) pins with `pull_policy: never` |
| Branch vs image | `research/deepinfer` was based on `95ac251`. `main` (`63f7581`, docs-only past `2cf50a0`) was merged into it (`88549e7`) so a stock build of the branch serves what the image serves. |
| Launch convention | `docker compose --profile single up -d` (or `batch`) from WSL, knobs in `.env`, `docker-compose.override.local.yml` bind-mounts `bench/` over `/app/bench`. From WSL `MODELS_DIR` must be `/mnt/g/models` (the `.env` value `G:/models` is the Windows-side spelling and Docker rejects it from WSL). |
| `.env` (box defaults) | `MODELS_DIR`, `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, GPU pin by UUID, `PREPARE=0`, `VERIFY=0`, `PORT=18020`, `MODEL=/app/models/Qwen3.8-27B-W4A16-AutoRound-fast`, `CTX=fast`, `VISION=1`, `VISION_OFFLOAD=0` (plus the API key, not recorded). Every arm writes its own `.env` from this base and restores the original afterwards. |
| VRAM spill check | WDDM counter `\GPU Process Memory(pid_0_luid_…_<LUID>…)\Non Local Usage` (the WSL VM's GPU memory demoted to system RAM), sampled every 2 s per boot with `G:\dev\hq-bench\gpumem_sample.ps1 -Match <LUID>`. A boot whose non-local usage grows during the measurement window is re-run with a smaller `KV_MEM`. The LUID changed with the 2026-10-22 restart and the sampler kept matching the old one until `phase3x.sh`, so the baseline re-check (`B_new1`, `E_new1`), drafter-1a, quant-7 and quant-5 boots have no WDDM record. For them, spill is judged from the `nvidia-smi` VRAM peak and headroom recorded per boot in `summary.txt`. |
| Spill policy from 2026-10-22 15:30 | Spill is judged by PCIe traffic, not memory arithmetic. Each boot logs `nvidia-smi dmon -s put -d 1` (`pcie.txt`), and a continuous Windows-side `dmon` log covers the whole run (`pcie_all.txt`). Clean decode moves ~25–90 MB/s, and prefill/host copies cause a few one-second bursts. A speed boot whose job window has median PCIe > 150 MB/s, or ≥ 10% (at least 30) of its seconds above 500 MB/s, is spilled: it is marked for re-run, Docker Desktop and WSL are restarted (`tools/supervise.sh` → `restart_stack.ps1`), and the queue resumes. Before every boot the idle GPU is checked (PCIe > 50 MB/s or the WSL VM holding > 256 MiB in system RAM triggers the same restart). Speed boots never share a boot with the KL probe or the perplexity pass (their `prompt_logprobs` spilled 300–650 MiB in modes A and E). The WDDM non-local growth is still recorded, informational only. The ≤ 55 °C cooling gate was dropped at the same time. |
| Raw output | `G:\dev\hq-results\deepinfer-20261020\<item-id>\` |

**Image change on 2026-10-22.** On resuming after the pause, `ghcr.io/syv-ai/hyperqwen:latest` had moved to `sha256:0cf71b42d301…` (main `ce9615c`). Between `2cf50a0` and `ce9615c`:
- no vLLM patches changed;
- pthfa gained upstream refactors;
- the launchers moved Vision handling into `launcher_common.sh`;
- the bench corpus loading changed (`bench/labd_corpus.py`, `bench/code_corpus.py`).

Following the user's instruction to use the latest image, `main` was merged into the branch (`27fbf65`), every variant image was rebuilt on the new base, and every result from here on pairs its arms on the new base. Results recorded before the pause name `c696d22e80b4…` in their headers and stay valid as recorded. `baseline.md` adds a re-check of modes B and E on the new image.

## Models (`G:\models`, mounted at `/app/models`)

| Directory | Size | Role in the plan | Status |
|---|---|---|---|
| `Qwen3.8-27B-W4A16-AutoRound` | 22 GB | W4A16 AutoRound target (base; quant-2-check reads it) | present |
| `Qwen3.8-27B-W4A16-AutoRound-fast` | 15 GB | W4A16 AutoRound target, fast variant (int4-GPTQ lm_head, MTP); the launcher's default when present | present |
| `Qwen3.8-27B-DFlash2-W4A16` | 1.2 GB | W4A16 DFlash2 drafter (shipped) | present |
| `Qwen3.8-27B-DFlash2` | 3.6 GB | bf16 DFlash2 drafter (`incoai/Qwen3.8-27B-DFlash2`), teacher for drafter-2 and drafter-3 | fetched 2026-10-20 with `prepare/fetch_dflash2.py --bf16` |
| `Orca-*`, `Swift-*`, `OrcaSAQ-2-27B` | 12–16 GB each | third-party checkpoints, not in the plan | present, untouched |
| `quality-data/` | 45 MB | wikitext / gsm8k / fineweb2 for `bench/quality_battery.py` | present |

`Qwen3.8-27B-W4A16-AutoRound/` holds `.bak*` files from earlier work (`*.bak-mtp`, `*.bak-draft`, `*.bak_embed`); they are left in place and no glob in the plan matches them.
