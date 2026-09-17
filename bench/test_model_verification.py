#!/usr/bin/env python3
"""CPU-only regression checks for supported model verification and selection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class ModelVerificationTests(unittest.TestCase):
    def test_supported_and_invalid_heads(self):
        # Execute the actual verifier block; no vLLM/GPU imports are needed.
        source = (REPO / "verify.sh").read_text(encoding="utf-8")
        block = source.split('$PY - "$MODEL" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
        for bits, packed, expected in [(8, True, 0), (4, True, 0),
                                       (16, True, 1), (None, True, 1),
                                       (4, False, 1)]:
            with self.subTest(bits=bits, packed=packed), tempfile.TemporaryDirectory() as tmp:
                model = Path(tmp)
                groups = {"embed": {"targets": ["re:.*embed_tokens$"], "weights": {"num_bits": 8}}}
                if bits is not None:
                    groups["head"] = {"targets": ["re:.*lm_head$"], "weights": {"num_bits": bits}}
                (model / "config.json").write_text(json.dumps({"quantization_config": {"config_groups": groups}}))
                weights = {"model.embed_tokens.weight_packed": "weights.safetensors"}
                weights["lm_head.weight_packed" if packed else "lm_head.weight"] = "weights.safetensors"
                (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
                (model / "weights.safetensors").touch()
                result = subprocess.run([sys.executable, "-", tmp], input=block, text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_single_verifies_selected_model_without_changing_batch(self):
        base = "models/Qwen3.8-27B-W4A16-AutoRound"
        cases = [("single", False, "", base),
                 ("single", True, "", base + "-fast"),
                 ("single", True, "custom model", "custom model"),
                 ("batch", True, "", base)]
        for mode, fast, override, expected in cases:
            with self.subTest(mode=mode, fast=fast, override=override), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for directory in ("docker", "single-user", "batch", base):
                    (root / directory).mkdir(parents=True, exist_ok=True)
                if fast:
                    (root / (base + "-fast")).mkdir()
                # Relocate only the container root; execute the real entrypoint.
                entry = (REPO / "docker/entrypoint.sh").read_text(encoding="utf-8")
                entry = entry.replace("cd /app", 'cd "$PWD"').replace("REPO=/app", 'REPO="$PWD"')
                (root / "entrypoint.sh").write_text(entry, encoding="utf-8")
                helper = (REPO / "single-user/select_model.sh").read_text(encoding="utf-8")
                (root / "single-user/select_model.sh").write_text(helper, encoding="utf-8")
                (root / "verify.sh").write_text('printf "%s\\n" "${MODEL:-$PWD/' + base + '}" > verified\n')
                single = 'REPO="$PWD"\nsource "$REPO/single-user/select_model.sh"\nprintf "%s\\n" "$MODEL" > served\n'
                (root / "single-user/start_qwen.sh").write_text(single)
                (root / "batch/start_qwen.sh").write_text('printf "%s\\n" "${MODEL:-$PWD/' + base + '}" > served\n')
                env = dict(os.environ, MODEL=override, PREPARE="0", VERIFY="1")
                run = subprocess.run(["bash", "entrypoint.sh", mode], cwd=root, env=env, text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                verified = (root / "verified").read_text().strip()
                served = (root / "served").read_text().strip()
                self.assertEqual(verified, served)
                if override:
                    self.assertEqual(served, expected)
                else:
                    self.assertTrue(served.endswith("/" + expected), served)


if __name__ == "__main__":
    unittest.main()
