#!/usr/bin/env python3
"""CPU check for docs/research/deepinfer/experiments/patches/spec5-accept-telemetry.patch, on the image's
vLLM with Triton's interpreter (no GPU): the patch applies with exact context to the installed vllm package,
the telemetry kernel's four rows match a torch reference (overlap sum_x min(p, q), survival, drafts, realised
accepts) for sampled requests with and without draft logits, greedy requests and a -1 placeholder draft, the
sampled tokens are the same with the telemetry on and off, and the publisher writes the JSON the client reads.

  docker run --rm -v "$PWD":/repo --entrypoint /app/venv/bin/python ghcr.io/syv-ai/hyperqwen:latest \\
      /repo/bench/test_spec5_telemetry.py

It patches copies of the two files in a temp directory and never touches the installed tree. Exit 0 pass,
1 a check failed, 3 the negative control failed (a reference fed the wrong drafts still matched, so the
comparison cannot tell).
"""
import os

# Before torch, triton or vllm is imported: the interpreter runs @triton.jit kernels on CPU tensors, and an
# empty CUDA_VISIBLE_DEVICES is the one condition under which vllm.triton_utils keeps Triton with no driver.
os.environ["TRITON_INTERPRET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "docs/research/deepinfer/experiments/patches/spec5-accept-telemetry.patch"
VLLM = Path(importlib.util.find_spec("vllm").origin).parent
REL = "v1/worker/gpu/spec_decode"
FILES = ("rejection_sampler.py", "rejection_sampler_utils.py")


def fail(msg):
    print(f"SPEC5TEST FAIL {msg}")
    sys.exit(1)


dry = subprocess.run(["patch", "-p1", "--dry-run", "--forward", "--fuzz", "0", "-d", str(VLLM), "-i", str(PATCH)],
                     capture_output=True, text=True)
if dry.returncode:
    fail(f"patch does not apply to {VLLM}:\n{dry.stdout}{dry.stderr}")

tmp = Path(tempfile.mkdtemp())
(tmp / REL).mkdir(parents=True)
for f in FILES:
    shutil.copy(VLLM / REL / f, tmp / REL / f)
p = subprocess.run(["patch", "-p1", "--forward", "--fuzz", "0", "-d", str(tmp), "-i", str(PATCH)],
                   capture_output=True, text=True)
if p.returncode:
    fail(f"patch does not apply to the copy:\n{p.stdout}{p.stderr}")


def load(name):
    mod = f"vllm.{REL.replace('/', '.')}.{name}"
    spec = importlib.util.spec_from_file_location(mod, tmp / REL / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m  # rejection_sampler imports the patched utils by this name
    spec.loader.exec_module(m)
    return m


import torch  # noqa: E402
import types  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402

utils = load("rejection_sampler_utils")


@triton.jit
def _log1p(x):
    return tl.log(1.0 + x)


# Triton's interpreter has no libdevice, and upstream's _resample_kernel (not under test) calls
# tldevice.log1p on the rejected-token path. The shim exists only in this process.
utils.tldevice = types.SimpleNamespace(log1p=_log1p)
sampler_mod = load("rejection_sampler")
rejection_sample = utils.rejection_sample


def case(V, K, temps, has_draft_logits, placeholder=None, seed=0):
    """One batch: request r is request state r, with K drafts and temperature temps[r]."""
    g = torch.Generator().manual_seed(seed)
    R = len(temps)
    nl = R * (K + 1)
    target = torch.randn(nl, V, generator=g) * 2
    draft_logits = torch.randn(R, K, V, generator=g) * 2 if has_draft_logits else None
    drafts = torch.randint(0, V, (nl,), generator=g)
    # Make some drafts the target argmax, so greedy rows accept a few.
    for r in range(R):
        for i in range(K):
            if (r + i) % 3:
                drafts[r * (K + 1) + i + 1] = target[r * (K + 1) + i].argmax()
    if placeholder is not None:
        drafts[placeholder] = -1
    cu = torch.arange(0, nl + 1, K + 1)
    args = (target, draft_logits, drafts, cu, torch.arange(nl), torch.arange(R),
            torch.arange(R).repeat_interleave(K + 1), torch.arange(K + 1).repeat(R),
            torch.tensor(temps, dtype=torch.float32), torch.arange(R) + 7, K)
    return args


def reference(args, num_sampled):
    target, draft_logits, drafts, cu, _, _, _, _, temps, _, K = args
    R = len(temps)
    out = torch.zeros(4, K, dtype=torch.float64)
    for r in range(R):
        surv = 1.0
        accepted = int(num_sampled[r]) - 1  # the resample always adds one token
        for i in range(K):
            row = r * (K + 1) + i
            d = int(drafts[row + 1])
            T = float(temps[r])
            if d < 0:
                a = ov = 0.0
            elif T == 0.0:
                a = ov = float(target[row].argmax() == d)
            else:
                p = torch.softmax(target[row].double(), 0)
                q = (torch.softmax(draft_logits[r, i].double() / T, 0) if draft_logits is not None
                     else torch.nn.functional.one_hot(torch.tensor(d), p.numel()).double())
                ov = float(torch.minimum(p, q).sum())
                a = min(1.0, float(p[d] / q[d]))
            surv *= a
            out[:, i] += torch.tensor([ov, surv, 1.0, float(i < accepted)], dtype=torch.float64)
    return out


def run(args):
    K = args[-1]
    acc = torch.zeros(4, K, dtype=torch.float64)
    s_on, n_on = rejection_sample(*args, accept_stats=acc)
    s_off, n_off = rejection_sample(*args)
    # Slots past num_sampled are uninitialised (new_empty) in upstream, so only the emitted ones compare.
    emitted = torch.arange(K + 1) < n_on[:, None]
    if not (torch.equal(n_on, n_off) and torch.equal(s_on[emitted], s_off[emitted])):
        fail("telemetry changed the sampled tokens")
    return acc, n_on


checks = 0
control_matched = False
for name, kw in [("sampled, draft logits", dict(V=300, K=3, temps=[1.0, 0.7, 1.3], has_draft_logits=True)),
                 ("sampled, multi-block vocab", dict(V=20000, K=2, temps=[0.9, 1.1], has_draft_logits=True)),
                 ("sampled, one-hot drafts", dict(V=300, K=3, temps=[1.0, 0.6], has_draft_logits=False)),
                 ("greedy", dict(V=300, K=4, temps=[0.0, 0.0, 0.0], has_draft_logits=True)),
                 ("mixed greedy and sampled, placeholder", dict(V=300, K=3, temps=[0.0, 1.0], has_draft_logits=True,
                                                                placeholder=6))]:
    args = case(**kw)
    got, num_sampled = run(args)
    want = reference(args, num_sampled)
    if not torch.allclose(got, want, rtol=1e-4, atol=1e-5):
        fail(f"{name}: kernel rows\n{got}\n!= reference\n{want}")
    if all(t == 0.0 for t in kw["temps"]) and not torch.equal(got[1], got[3]):
        fail(f"{name}: greedy survival {got[1].tolist()} != realised accepts {got[3].tolist()}")
    # Negative control: the reference fed the drafts shifted by one position must disagree.
    shifted = list(args)
    shifted[2] = torch.roll(args[2], 1)
    if torch.allclose(got, reference(tuple(shifted), num_sampled), rtol=1e-4, atol=1e-5):
        control_matched = True
    checks += 1
    print(f"ok  {name}: drafts={got[2].tolist()} accepted={got[3].tolist()} "
          f"survival={[round(x, 3) for x in got[1].tolist()]} overlap={[round(x, 3) for x in got[0].tolist()]}")

out_dir = tmp / "telemetry"
tel = sampler_mod._Spec5Telemetry(str(out_dir), 3, torch.device("cpu"))
tel.acc += torch.arange(12, dtype=torch.float64).reshape(4, 3)
tel.step()
tel.step()
body = json.loads((out_dir / f"spec5-{os.getpid()}.json").read_text())
if body["calls"] != 2 or body["k"] != 3 or body["overlap"] != [0, 1, 2] or body["accepted"] != [9, 10, 11]:
    fail(f"published file {body}")
print("ok  publisher: spec5-<pid>.json with calls, k and the four rows")
shutil.rmtree(tmp)

if control_matched:
    print("SPEC5TEST NEGATIVE-CONTROL-FAILED a reference with shifted drafts matched the kernel")
    sys.exit(3)
print(f"SPEC5TEST PASS cases={checks} patch=applies")
