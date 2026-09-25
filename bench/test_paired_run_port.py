"""bench/paired_run.sh measures the server on its own --port (audit P-4).

Symptom: paired_run.sh set PORT=18021 and did not export it. The measure
script (measure_c1.sh: PORT=${PORT:-18020}) then measured whatever server
answered on the default port, and the run recorded those numbers as arm
evidence.

Run: python3 bench/test_paired_run_port.py (Linux/WSL2; needs bash, curl,
python3; no GPU, no vLLM). The driver runs with --no-boot and a stub measure
script that keeps the measure_c1.sh default-port contract. A "booted" server
listens on --port and a decoy listens on the stub's default port.

Fail-closed: the test fails if any repeat reaches the decoy, or if the booted
server gets no request. It skips (exit 0) only when another run holds the GPU
lock, so that it never removes a real lock.
"""
import http.server
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import threading
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
LOCK = REPO / "bench" / "results" / ".gpu-lock"


def serve():
    hits = []

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, hits


STUB = """#!/bin/bash
PORT=${PORT:-%d}
code=$(curl -s -o /dev/null -w '%%{http_code}' "http://127.0.0.1:$PORT/health")
if [ "$code" != 200 ]; then
  echo '{"status":"INVALID","reason":"no-server","measurements":{},"artifacts":[]}' > "$1"; exit 1
fi
echo '{"status":"PASS","reason":"ok","measurements":{"e2e":100.0},"artifacts":[]}' > "$1"
"""


class PairedRunPort(unittest.TestCase):
    def test_measure_reaches_the_driver_port_not_the_default(self):
        if LOCK.exists():
            self.skipTest(f"GPU lock held at {LOCK}; not removing a real run's lock")
        booted, booted_hits = serve()
        decoy, decoy_hits = serve()
        exp = f"test-paired-port-{os.getpid()}"
        out = REPO / "bench" / "results" / exp
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = pathlib.Path(tmp)
                measure = tmp / "measure.sh"
                measure.write_text(STUB % decoy.server_port)
                for arm in ("control", "treatment"):
                    (tmp / f"{arm}.env").write_text("PAIRED_TEST_ARM=same\n")
                env = {k: v for k, v in os.environ.items() if k != "PORT"}
                run = subprocess.run(
                    ["bash", "bench/paired_run.sh", "--exp", exp,
                     "--control-env", str(tmp / "control.env"),
                     "--treatment-env", str(tmp / "treatment.env"),
                     "--measure", str(measure), "--pairs", "2", "--aa",
                     "--port", str(booted.server_port), "--no-boot", "--no-teardown"],
                    cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
            results = sorted(out.glob("*/result.json"))
            statuses = [json.loads(p.read_text())["status"] for p in results]
            msg = f"stdout={run.stdout[-500:]} stderr={run.stderr[-500:]}"
            self.assertEqual(len(results), 4, msg)
            self.assertEqual(decoy_hits, [], "a repeat measured the decoy on the default port")
            self.assertEqual(len(booted_hits), 4, msg)
            self.assertEqual(statuses, ["PASS"] * 4, msg)
        finally:
            booted.shutdown()
            decoy.shutdown()
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
