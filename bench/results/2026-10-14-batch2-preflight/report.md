# Batch 2 preflight, 2026-10-14

Raw output: `preflight.txt` (produced by `../2026-10-14-batch2-tools/preflight.ps1`).

- **Branch:** `cline/gf59cdmh` at `f1a033e`, level with `origin`. No tracked modifications. The untracked files
  are the owner's (`.env.bench`, `.env.hf`, `.env.long`, `docker-compose.override.local.yml`, `evals/`,
  `prepare/logs/`, twelve `docs/*` files). They were left alone, and every `git add` in this batch names its paths.
  The owner's `.env` was copied to `.git/env-owner-backup-2026-10-14` before the first arm overwrote it, and is
  restored at the end.
- **`core.autocrlf`:** `false`.
- **GPUs:** index 0 is the RTX 3090, 24,576 MiB, `power.limit 250.00 W`, `power.default_limit 350.00 W`,
  `display_active Disabled`, UUID `GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84`. Index 1 is the RTX 3080 Ti,
  `display_active Enabled`. Driver 610.88.
- **Counter instance:** batch 1's `luid_0x00000000_0x000114dd_phys_0` no longer exists (the first preflight
  pass, filtered on it, printed nothing). The 3090 is now `luid_0x00000000_0x00011456_phys_0`: Dedicated 4.03 MiB
  and Shared 0.25 MiB idle. The 3080 Ti is `..12457` (529.67 MiB dedicated, matching `nvidia-smi`'s 528-533 MiB).
  Confirmed at the first boot, when `..11456` rose to 24,085.9 MiB dedicated.
- **Idle Shared Usage on the 3090:** 0.25 MiB in all six samples.
- **Models:** `Qwen3.8-27B-W4A16-AutoRound-fast`, `Qwen3.8-27B-DFlash2-W4A16` and `quality-data` are present.
  `prepare` was not run.
- **Image:** `docker images` lists no `hyperqwen` image, so batch 1's `hyperqwen:exp-e7fb509` is gone. See
  `../2026-10-14-batch2-image/image.txt` for the rebuild.
- **`upstream` push URL is not disabled.** `git config --get-all remote.upstream.pushurl` exits 1 (unset) and
  `git remote -v` lists `https://github.com/syv-ai/qwen38-27b-rtx3090 (push)`. The brief says it is disabled on
  purpose. It is not, on this checkout. Left unchanged; nothing in this batch pushes anywhere but `origin`.
- **Free space on G:** 181,028,376,576 bytes.

Found while running, not part of the preflight:

- **The local image contains the owner's untracked `.env.bench`, `.env.hf` and `.env.long`**
  (`ls -a /app | grep ^.env` inside the container). `.dockerignore` excludes `.env` but not `.env.*`, and
  `docker compose build` sends the whole working tree. `docker-compose.override.local.yml` and `evals/` are in
  `/app` as well. The image was never pushed. Batch 1's image was built the same way and presumably had the same
  content (not verified: it no longer exists).
- Batch 1's quality smokes wrote `result_e2-B.json` and `result_e3-A.json` into `G:/models/quality-data`, because
  `quality_battery.py` writes its result into `QUALITY_DATA`. Batch 2 points `QUALITY_DATA` at a `/tmp` directory
  of symlinks inside the container, so nothing new lands there. The two batch 1 files were left in place.
