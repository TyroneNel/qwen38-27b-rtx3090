"""Peak GPU memory by polling nvidia-smi. Library: `with Peak() as p: ...; p.peak_mib`. CLI wrapper:

  python3 vram.py OUT.json -- cmd args...     # runs cmd, writes {"idle_mib","peak_mib","samples","gpu"} to OUT.json

GPU is HARNESS_GPU (index or UUID), else the first CUDA_VISIBLE_DEVICES entry (the box pins the 3090 by UUID,
ENVIRONMENT.md), else 0. nvidia-smi memory.used is the SECONDARY reading: under WSL2/WDDM it is near-flat per cell
(preallocated KV pool) and misses spill. The primary spill/VRAM source on the GPU box is the WDDM "Non Local Usage"
sampler, see gpumem_summary.py and README "VRAM and spill".
No nvidia-smi (the CPU self-check box) gives peak_mib = None, never an error. memory.used is the whole
card, not this process: nothing else may run on it. On WSL2/WDDM per-process memory is not reported, and
memory the WSL VM spilled to system RAM is NOT in this number (see ENVIRONMENT.md "VRAM spill check").
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time


def gpu():
    return os.environ.get("HARNESS_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() or "0"


def used_mib():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-i", gpu(),
                              "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class Peak:
    def __init__(self, period=0.5):
        self.period, self.samples, self.peak_mib, self.idle_mib = period, 0, None, None
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            v = used_mib()
            if v is not None:
                self.samples += 1
                self.peak_mib = v if self.peak_mib is None else max(self.peak_mib, v)
            self._stop.wait(self.period)

    def __enter__(self):
        self.idle_mib = used_mib()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        v = used_mib()  # one last read so a very short run still has a sample
        if v is not None:
            self.peak_mib = v if self.peak_mib is None else max(self.peak_mib, v)

    def as_dict(self):
        return {"gpu": gpu(), "source": "nvidia-smi memory.used, secondary: misses WDDM spill (see spill.json)", "idle_mib": self.idle_mib,
                "peak_mib": self.peak_mib, "samples": self.samples}


if __name__ == "__main__":
    if "--" not in sys.argv or len(sys.argv) < 4:
        sys.exit(__doc__)
    i = sys.argv.index("--")
    with Peak() as p:
        rc = subprocess.call(sys.argv[i + 1:])
    json.dump(p.as_dict(), open(sys.argv[1], "w"))
    sys.exit(rc)
