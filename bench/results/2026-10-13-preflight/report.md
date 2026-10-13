# Preflight, 2026-10-13

Power limit on the card under test: 250 W (verified, see below). The box has a second GPU, which the brief
did not expect.

Raw output: `preflight.txt` (commands from `preflight.ps1`, run in Windows PowerShell 5.1, plus the git lines
after the branch checkout appended at the end).

## Answers

**Branch clean, expected commit.** When preflight started, the worktree was on `main` at `95ee3b6`, not on the
experiment branch. `git status --short` showed untracked files only: `.env.bench`, `.env.hf`, `.env.long`,
`docker-compose.override.local.yml`, 12 untracked `docs/*` entries, `evals/` and `prepare/logs/`. It showed no
modified or staged tracked files. These are the owner's local files. I did not commit them, and every `git add`
in this run names its paths explicitly. The brief's `git add bench/results docs` would also have picked up the
untracked docs. `git checkout -b cline/gf59cdmh --track origin/cline/gf59cdmh` then gave:

```
PS> git log --oneline -1
b9c2220 docs: experiment list and GPU-agent brief for the 3090 box
PS> git status -sb --untracked-files=no
## cline/gf59cdmh...origin/cline/gf59cdmh
```

Tracked tree: clean.

**core.autocrlf.** `git config core.autocrlf` printed `false`. That value is repo-local and overrides the
system `true`:

```
file:C:/Program Files/Git/etc/gitconfig	true
file:.git/config	false
```

`git ls-files --eol` on `docker/entrypoint.sh` and `single-user/start_qwen.sh` printed `i/lf w/lf`.

**GPU in a container, power limit.** The container sees both cards:

```
GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84)
GPU 1: NVIDIA GeForce RTX 3080 Ti (UUID: GPU-c190f571-609f-24a9-d678-207c2d0ac6fa)
```

Host query for the card under test:

```
0, NVIDIA GeForce RTX 3090, 24576 MiB, 610.88, 250.00 W, 350.00 W, 365.00 W, 20.09 W, Disabled, ..., 00000000:01:00.0, GPU-2e8e4d29-...
```

`power.limit` is 250.00 W and `power.default_limit` is 350.00 W, so the 250 W cap is set on this card. I did
not change it.

**Second GPU.** The box also carries an RTX 3080 Ti (12 GB, 350 W limit, `display_active Enabled`). Gotcha 53
says Docker Desktop does not enforce a `--gpus device=N` request. For that reason every arm's `.env` pins the
3090 by UUID: `CUDA_VISIBLE_DEVICES=GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84` and the same value for
`NVIDIA_VISIBLE_DEVICES`. The first boot's log confirms which card the engine used.

**Display.** The 3090 shows `display_active Disabled` and `Disp.A Off`, with `0MiB / 24576MiB` used and no
process listed on GPU 0 in `nvidia-smi`. All desktop processes (explorer, Edge, Docker Desktop, Terminal) sit
on GPU 1, the 3080 Ti, which has `Disp.A On` and 523 MiB used. The 3090 is headless, so `KV_MEM` needs no
display deduction (gotcha 31).

**Models present** (`Get-ChildItem G:\models`): `Qwen3.8-27B-W4A16-AutoRound`,
`Qwen3.8-27B-W4A16-AutoRound-fast`, `Qwen3.8-27B-DFlash2-W4A16`, three fine-tune families (`Orca-*`,
`Swift-*`, `OrcaSAQ-2-27B`), `quality-data`, `hf-download`, `hf-upload`, `mpf`.

- `single-user/select_model.sh` looks for `$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast` and falls back to
  `$REPO/models/Qwen3.8-27B-W4A16-AutoRound`. Both are present. The fast variant wins.
- The `SPEC=dflash2` branch of `single-user/start_qwen.sh` looks for `$REPO/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors`,
  then `$REPO/models/Qwen3.8-27B-DFlash2/model.safetensors`. The first is present, at 1,280,633,960 bytes.
- I checked the `docker/prepare.sh` state keys with a host Python. Both AutoRound dirs report
  `lm_head.weight_packed`, packed `embed_tokens`, the MTP layer, `mtp.draft_lm_head.weight_packed`,
  `mtp_draft_vocab_ids.pt` and `tokenizer.json` all present (`True True True True True True`). Prepare has
  nothing to do, so I did not run the `prepare` profile.

**Docker drive.** `docker_data.vhdx` lives on `G:` (`G:\DockerDesktop\DockerDesktopWSL\disk\docker_data.vhdx`,
87.9 GB). G: has 180.7 GB free. A named volume `hyperqwen_qwen-cache` (compile and JIT cache) already exists
from earlier sessions. The arms reuse it, so the first boot may be a warm-cache boot. Its contents were not
verified.

**Idle Shared Usage baseline.** There are three adapter instances. My preflight guess at the mapping was wrong.
The first boot (E1-A, `../2026-10-13-E1/mem-trace-A.csv`) corrected it: the 3090 is `..114dd`. Its Dedicated
Usage rose from 4.0 to 22,859.8 MiB while `nvidia-smi` showed 22,856 MiB used on GPU 0. `..1358d` stayed at
0 throughout.

| LUID | Idle dedicated | Idle shared | Card |
|---|---|---|---|
| `luid_0x00000000_0x000114dd_phys_0` | 4,227,072 B | 262,144 B (0.25 MiB) | 3090, confirmed by the E1-A boot |
| `luid_0x00000000_0x0001252b_phys_0` | 554,065,920 B | 95.1-95.4 MB (90.7-91.0 MiB) | 3080 Ti, display card |
| `luid_0x00000000_0x0001358d_phys_0` | 0 | 8,192 B | Not identified. It stays at 0 when the 3090 loads. |

The 3090's idle Shared Usage is 0.25 MiB and stays flat across all six samples. It is not ~86 MiB. In E1-A it
jumped to 80.2 MiB at the sample where the engine created its CUDA context (dedicated 391.6 MiB) and settled at
92.2 MiB once the weights loaded. That happened with 1,471 MiB still free on the card, so it is the engine's
fixed host-side mapping, not a spill. The spill reference for every arm is therefore the loaded engine's own
floor: the minimum Shared Usage over samples with Dedicated above 20,000 MiB. Any sample more than 16 MiB above
that floor is flagged (`../2026-10-13-tools/memsum.sh`).

**Free VRAM before load:** 3090 `24576 MiB, 0 MiB, 24326 MiB` (total, used, free).
