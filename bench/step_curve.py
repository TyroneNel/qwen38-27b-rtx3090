"""Step-cost curve (DeepInfer runtime-2): GPU ms per decode step against total query tokens, per context depth.

What it times. The serving forward itself, as the launcher builds it: the tool sources
single-user/start_qwen.sh with PRINT_ARGV=1 for each backend, keeps the argv and the exported
environment it would exec vllm serve with, and builds the same engine in its own process
(vLLM's in-process LLMEngine: same model runner, same compile, same captured CUDA graphs, no
HTTP server). It then replays the V2 model runner's own dummy steps, the ones vLLM 0.31 uses to
seed its adaptive-verification cost curve (_dummy_run with context_len, set_dummy_context), so
the step goes through the runner's real CUDA-graph dispatch: FULL decode graph where a real
batch of that shape gets one, piecewise or eager where it does not. Per step:

  forward   target model forward (graph replay or piecewise), the runner's StepTimingCollector
  drafter   the speculator's propose on the same batch (DFlash2 drafter), same collector
  lm_head   target compute_logits over all total_q rows (a verify step scores every row)
  step      forward + drafter + lm_head of the same step; excludes rejection sampling and host
            scheduling time

Layouts. seq1: one request of q = 1..W tokens, W = num_speculative_tokens + 1 (8 on every
backend here), in the mode the runner dispatches: FULL at q = W, PIECEWISE below. seq1pw: the
q = W point forced PIECEWISE, so the width slope q=1..W is read inside one graph mode. seats:
total_q/W requests of W tokens each (multi-seat verify, FULL; total_q a multiple of W, at most
max_num_seqs requests). One request wider than W has no verify path in a W boot (GDN keeps W
speculative state slots), so it needs a boot at DFLASH_TOKENS = q - 1 and is not measured.

Two corrections make the dummy a verify step (rev 2; rows without "rev": 2 are never reused):
the context survives the FULL dispatch (the Triton and pthfa builders' build_for_cudagraph_capture
reset seq_lens to 1, so FULL points on those backends attended over 1 token), and every dummy row
carries q - 1 drafts, so GDN runs its spec-decode path instead of a chunked prefill.

Context. Each request continues after `depth` tokens of context; its block table spans
depth + q tokens of the KV pool. The pool stays the launcher's pinned KV_MEM (the memory layout
serving runs with), and max_model_len is raised to the deepest point if the launcher's is
shorter (vLLM's one-request-fits check is skipped for that; nothing is ever prefilled). Where
the spans need more blocks than the pool holds they wrap and re-read blocks (kv_wrap > 1 in
the output): the reads are still depth tokens per request and far larger than L2, so the
timing holds, but say so when quoting those rows.

Backends (one engine boot each; --backend takes a comma list and runs them in turn):
  fa2    SPEC=dflash2 CTX=long FA2_FP8KV=1 PTH_FA=0   fp8 KV, FA2 plugin       depths to 150k
  pthfa  SPEC=dflash2 CTX=long PTH_FA=1               int8 per-token-head, pthfa depths to 150k
  kvarn  SPEC=dflash2 CTX=huge                        KVarN 4/2-bit, Triton     depths to 240k
  int8   SPEC=dflash2 CTX=long FA2_FP8KV=0 PTH_FA=0   int8 + Triton split-KV    depths to 150k
Any other launcher variable in the environment (DFLASH_TOKENS, KV_MEM, INT8_ACT, ...) passes
through to the launcher, as it would for a server.

Run it in a container of the serving image that has the GPU to itself (no server running):

  docker run --rm --gpus all --ipc host -e HOME=/cache -v <compose cache volume>:/cache \\
    -v <models>:/app/models -v <repo>/bench:/app/bench --entrypoint /app/venv/bin/python \\
    ghcr.io/syv-ai/hyperqwen:latest /app/bench/step_curve.py --backend fa2,pthfa,kvarn

--dry-run resolves each backend's launcher argv and environment, parses the argv with vLLM's own
serve parser and prints the plan; it needs no GPU (the image without --gpus, with the models
mounted: the launcher refuses SPEC=dflash2 without the drafter). Output: JSONL (one row per point,
raw samples, appended per point so a crash keeps what ran; --resume skips those) at --out, a CSV
beside it (.csv: backend, graphs, layout, depth, total_q, part, ms_median, ms_p10, ms_p90, n,
cg_mode, ran_tokens, num_reqs, kv_wrap) and an ASCII plot (.plot.txt; the image has no
matplotlib). stdout ends with a q x depth table per backend and one `STEP_CURVE backend=...`
summary line each. --graphs dense boots with a graph at every q (no token padding: the attention
sees exactly q rows), for the kernel stairs; serve (the default) is what the server runs, where a
q between capture sizes is padded up (ran_tokens).

Self-check: from 8k to 150k context a step must rise by at least the time to read the extra KV
of the 16 full-attention layers once at 936 GB/s (x requests; p10 against p10). A (layout, q)
column below it is valid=INVALID in the CSV; a failing seq1 q=W column (the serving verify step)
marks the whole backend INVALID. --report-only re-checks an existing JSONL without a GPU.

Exit: 0 when every backend measured and passed the self-check (a measurement), 2 when a backend
could not be resolved or booted, measured nothing, or failed the self-check.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

BENCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH)
ENV_MARK = b"\n__STEP_CURVE_ENV__\n"
PLAN_ENV = "STEP_CURVE_PLAN"

BACKENDS = {
    "fa2": ({"SPEC": "dflash2", "CTX": "long", "FA2_FP8KV": "1", "PTH_FA": "0"}, (), 150_000,
            "fp8 KV on the FA2 plugin"),
    "pthfa": ({"SPEC": "dflash2", "CTX": "long", "PTH_FA": "1"}, ("FA2_FP8KV",), 150_000,
              "int8 per-token-head KV on pthfa (CUSTOM)"),
    "kvarn": ({"SPEC": "dflash2", "CTX": "huge"}, ("FA2_FP8KV", "PTH_FA"), 240_000,
              "KVarN 4/2-bit KV, Triton"),
    "int8": ({"SPEC": "dflash2", "CTX": "long", "FA2_FP8KV": "0", "PTH_FA": "0"}, (), 150_000,
             "int8 per-token-head KV, Triton split-KV verify"),
}
DEFAULT_DEPTHS = [8_000, 25_000, 64_000, 112_000, 150_000, 240_000]
PARTS = ("forward", "drafter", "lm_head", "step")
# 2: FULL dummy steps keep their context (build_for_cudagraph_capture reset seq_lens to 1 on Triton and
# pthfa) and every dummy row is a GDN spec-decode verify (it ran as a prefill chunk). Rows without it
# measured the wrong thing; a run never mixes them in, --report-only re-checks them.
REV = 2


def parse_ints(text):
    """'1-32' or '1,2,4,8' or '8000,25k' -> sorted unique ints."""
    out = set()
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(_int(a), _int(b) + 1))
        else:
            out.add(_int(part))
    return sorted(out)


def _int(s):
    s = s.strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(s)


def percentile(xs, p):
    """Linear-interpolated percentile of a non-empty list, p in [0, 100]."""
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats(xs):
    return {"median": percentile(xs, 50), "p10": percentile(xs, 10), "p90": percentile(xs, 90), "n": len(xs)}


# ---------------------------------------------------------------- launcher resolution (no GPU, no vLLM)

def run_launcher(launcher, overrides, unset):
    """Source the launcher with PRINT_ARGV=1 and return (argv, env, log). The EXIT trap dumps the exported
    environment after qwen_exec prints the argv and exits, so every default and export the launcher makes
    is in it. The API key is dropped; nothing here needs it."""
    env = dict(os.environ)
    for k in unset:
        env.pop(k, None)
    env.update(overrides)
    script = ('trap \'rc=$?; printf "\\n__STEP_CURVE_ENV__\\n"; env -0; exit $rc\' EXIT\n'
              'export PRINT_ARGV=1\n. "$1"\n')
    p = subprocess.run(["bash", "-c", script, "step_curve", launcher], cwd=REPO, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    out = p.stdout
    cut = out.rfind(ENV_MARK)
    head, tail = (out[:cut], out[cut + len(ENV_MARK):]) if cut >= 0 else (out, b"")
    lines = head.decode(errors="replace").split("\n")
    log = "\n".join(lines) + p.stderr.decode(errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"launcher exited {p.returncode}:\n{log.strip()[-3000:]}")
    start = None
    for i in range(len(lines) - 1):
        if lines[i].endswith("venv/bin/vllm") and lines[i + 1] == "serve":
            start = i
    if start is None:
        raise RuntimeError(f"no 'venv/bin/vllm serve' argv in the launcher output:\n{log.strip()[-3000:]}")
    argv = [ln for ln in lines[start + 2:]]
    while argv and argv[-1] == "":
        argv.pop()
    child_env = {}
    for item in tail.split(b"\0"):
        if b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        k = k.decode(errors="replace").strip("\n")
        if k.startswith("BASH_FUNC_") or k in ("PRINT_ARGV", "VLLM_API_KEY", "OPENAI_API_KEY"):
            continue
        child_env[k] = v.decode(errors="replace")
    return argv, child_env, log


def arg_value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def set_arg(argv, flag, value):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            argv[i + 1] = value
            return argv
        if a.startswith(flag + "="):
            argv[i] = f"{flag}={value}"
            return argv
    return argv + [flag, value]


def resolve_backend(name, args):
    overrides, unset, cap, desc = BACKENDS[name]
    depths = args.depths if args.depths else [d for d in DEFAULT_DEPTHS if d <= cap]
    qmax = max(args.q)
    need_len = -(-(max(depths) + qmax) // 128) * 128
    argv, env, log = run_launcher(args.launcher, overrides, unset)
    launcher_len = int(arg_value(argv, "--max-model-len") or 0)
    raised = False
    if launcher_len < need_len and "DFLASH_MAX_LEN" not in os.environ:
        argv, env, log = run_launcher(args.launcher, {**overrides, "DFLASH_MAX_LEN": str(need_len)}, unset)
        raised = True
    max_len = int(arg_value(argv, "--max-model-len") or 0)
    if max_len < need_len:
        raise RuntimeError(f"max-model-len {max_len} < {need_len} needed for depth {max(depths)} + q {qmax}"
                           " (DFLASH_MAX_LEN in the environment overrides the tool's raise)")
    cc = json.loads(arg_value(argv, "--compilation-config") or "{}")
    if args.graphs == "dense":
        # Every token count 1..max(qmax, launcher cap) gets a graph: no token padding, so the
        # attention rows are exactly total_q. Not the serving capture set; the rows say graphs=dense.
        top = max(qmax, int(cc.get("max_cudagraph_capture_size", 0) or 0))
        cc["cudagraph_capture_sizes"] = list(range(1, top + 1))
        cc["max_cudagraph_capture_size"] = top
        argv = set_arg(argv, "--compilation-config", json.dumps(cc, separators=(",", ":")))
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    return {
        "backend": name, "desc": desc, "env_overrides": overrides, "unset": list(unset),
        "depths": depths, "q": args.q, "layouts": args.layouts, "iters": args.iters,
        "warmup": args.warmup, "graphs": args.graphs, "argv": argv, "env": env,
        "launcher_max_len": launcher_len, "max_model_len": max_len, "raised_max_len": raised,
        "compilation_config": cc, "launcher_log": log[-4000:],
    }


def describe(plan):
    a = plan["argv"]
    spec = json.loads(arg_value(a, "--speculative-config") or "{}")
    width = int(spec.get("num_speculative_tokens", 0)) + 1 if spec else 1
    seqs = int(arg_value(a, "--max-num-seqs") or 1)
    pts = point_list(plan, width, seqs)
    keys = sorted(k for k in plan["env"] if k.startswith(("VLLM_", "FA2_", "PTH", "KVARN", "PYTORCH_CUDA")))
    print(f"== backend {plan['backend']}: {plan['desc']}")
    over = " ".join(k + "=" + v for k, v in plan["env_overrides"].items())
    print(f"   launcher env: {over}"
          + (f" (unset {','.join(plan['unset'])})" if plan["unset"] else ""))
    print(f"   model {a[0] if a and not a[0].startswith('-') else arg_value(a, '--model')}")
    print(f"   kv-cache-dtype {arg_value(a, '--kv-cache-dtype')}  attention-backend "
          f"{arg_value(a, '--attention-backend') or '(default)'}  block-size {arg_value(a, '--block-size') or 'auto'}"
          f"  kv-cache-memory {arg_value(a, '--kv-cache-memory') or '(GPU_UTIL)'}"
          f"  gpu-memory-utilization {arg_value(a, '--gpu-memory-utilization')}")
    print(f"   max-model-len {plan['max_model_len']}"
          + (f" (launcher {plan['launcher_max_len']}, raised with DFLASH_MAX_LEN)"
             if plan["raised_max_len"] else " (launcher's)") + "; one-request KV check skipped"
          + f"  max-num-seqs {seqs}  decode width {width}")
    print(f"   speculative-config {arg_value(a, '--speculative-config')}")
    print(f"   compilation-config {json.dumps(plan['compilation_config'], separators=(',', ':'))}"
          f"  graphs={plan['graphs']}")
    env_text = " ".join(k + "=" + plan["env"][k] for k in keys)
    print(f"   env: {env_text}")
    print(f"   depths {plan['depths']}  layouts {plan['layouts']}  q {compact(plan['q'])}")
    steps = len(pts) * (plan["iters"] + plan["warmup"])
    est = sum((plan["iters"] + plan["warmup"]) * (30 + 0.25 * d / 1000) for _, d, _ in pts) / 1000
    print(f"   {len(pts)} points x ({plan['warmup']} warmup + {plan['iters']}) = {steps} steps,"
          f" roughly {est / 60:.0f} min of steps after a 3-8 min boot (compile + capture)")
    return pts


def compact(qs):
    out, i = [], 0
    while i < len(qs):
        j = i
        while j + 1 < len(qs) and qs[j + 1] == qs[j] + 1:
            j += 1
        out.append(str(qs[i]) if i == j else f"{qs[i]}-{qs[j]}")
        i = j + 1
    return ",".join(out)


def point_list(plan, width, max_seqs):
    """seq1: one request of q <= W tokens (W = decode width = num_speculative_tokens + 1), run in whatever
    graph mode the runner dispatches (FULL only at q = W). seq1pw: the q = W point forced PIECEWISE, the
    same mode as q < W, so the width slope is read inside one mode. seats: q/W requests of W tokens (FULL).
    A single request wider than W has no verify path in this boot (the GDN state holds W speculative slots,
    the attention decode slice stops at W), so it needs a boot with DFLASH_TOKENS = q - 1 and is skipped."""
    pts = []
    for depth in plan["depths"]:
        if "seats" in plan["layouts"]:
            for q in plan["q"]:
                if q % width == 0 and 2 <= q // width <= max_seqs:
                    pts.append(("seats", depth, q))
        if "seq1" in plan["layouts"]:
            for q in plan["q"]:
                if q <= width:
                    pts.append(("seq1", depth, q))
            if width in plan["q"]:
                pts.append(("seq1pw", depth, width))
    return pts


# ---------------------------------------------------------------- engine side (child process, GPU)

def _runner(obj):
    worker = getattr(obj, "worker", None) or obj
    return worker.model_runner


def _builder_classes(r):
    """Every attention-metadata builder class the target and the drafter use (found through any
    attn_groups-shaped attribute of the runner and the speculator)."""
    classes = set()

    def walk(x, depth=0):
        if depth > 3:
            return
        if hasattr(x, "get_metadata_builder"):
            try:
                classes.add(type(x.get_metadata_builder(0)))
            except Exception:   # noqa: BLE001 - a group without a builder
                pass
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y, depth + 1)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y, depth + 1)

    for owner in (r, getattr(r, "speculator", None)):
        if owner is None:
            continue
        for v in vars(owner).values():
            if isinstance(v, (list, tuple, dict)):
                walk(v)
        for v in vars(owner).values():   # one level down (speculator.model_state, ...)
            if hasattr(v, "__dict__") and not isinstance(v, type):
                for w in vars(v).values():
                    if isinstance(w, (list, tuple)) and w and isinstance(w[0], (list, tuple)):
                        walk(w)
    return classes


def _measure_point(obj, layout, depth, q, warmup, iters):
    """Runs in the worker (in-process engine). One point: warmup, then iters timed dummy steps.

    Two corrections turn the runner's profiling dummy into a verify step at `depth`:
    - A dummy step that dispatches to a FULL graph builds its metadata with build_for_cudagraph_capture,
      and the Triton and pthfa builders reset the shared seq_lens buffer to 1 there (so a capture walks no
      cache). That wiped set_dummy_context's depth: the FULL point attended over 1 token. Real steps call
      build(), which leaves seq_lens alone, so the wrapper below puts the context back after the capture
      build, for every builder class of the target and the drafter.
    - make_dummy leaves num_draft_tokens_per_req unset, so GDN classed every non-FULL row wider than 1
      token as a prefill chunk (chunked GDN, ~3x the step). The wrapper marks each dummy request's q - 1
      trailing tokens as drafts, which puts it on GDN's spec-decode path, as a real verify is."""
    import numpy as np
    import torch
    from vllm.v1.worker.gpu import model_runner as mr_mod

    r = _runner(obj)
    if not hasattr(r, "step_timing") or r.speculator is None:
        raise RuntimeError(f"{type(r).__module__}.{type(r).__name__}: needs the V2 model runner with a speculator")
    width = int(r.decode_query_len)
    seen = {"restored": 0}
    orig_dispatch = mr_mod.dispatch_cg_and_sync_dp

    def spy(*a, **k):
        if layout == "seq1pw":
            a = list(a)
            a[3] = None   # uniform_token_count: no FULL decode graph matches, PIECEWISE does
            k["uniform_decode"] = False
        out = orig_dispatch(*a, **k)
        seen["desc"] = out[0]
        return out

    InputBatch = mr_mod.InputBatch
    orig_make = InputBatch.__dict__["make_dummy"]

    def make_dummy(cls, *a, **k):
        b = orig_make.__func__(cls, *a, **k)
        drafts = (np.asarray(b.num_scheduled_tokens, dtype=np.int32) - 1).clip(0, width - 1)
        b.num_draft_tokens_per_req = drafts
        b.num_draft_tokens = int(drafts.sum())
        return b

    patched = []

    def wrap_capture(cls):
        own = "build_for_cudagraph_capture" in cls.__dict__
        orig = cls.__dict__.get("build_for_cudagraph_capture")
        inherited = getattr(cls, "build_for_cudagraph_capture")

        def build_for_cudagraph_capture(self, cam, *a, **k):
            saved = cam.seq_lens.clone()
            md = inherited(self, cam, *a, **k)
            cam.seq_lens.copy_(saved)
            sl = getattr(md, "seq_lens", None)
            if isinstance(sl, torch.Tensor) and sl.shape == saved.shape and sl.data_ptr() != saved.data_ptr():
                sl.copy_(saved)
            seen["restored"] += 1
            return md

        cls.build_for_cudagraph_capture = build_for_cudagraph_capture
        patched.append((cls, own, orig))

    def once():
        if layout == "seats":
            return r._dummy_run(q, uniform_decode=True, context_len=depth)
        saved = r.max_num_reqs
        r.max_num_reqs = 1   # _dummy_run splits num_tokens over min(num_tokens, max_num_reqs) requests
        try:
            return r._dummy_run(q, context_len=depth)
        finally:
            r.max_num_reqs = saved

    mr_mod.dispatch_cg_and_sync_dp = spy
    InputBatch.make_dummy = classmethod(make_dummy)
    for cls in _builder_classes(r):
        if hasattr(cls, "build_for_cudagraph_capture"):
            wrap_capture(cls)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                h, _ = once()
                r.model.compute_logits(h[:q])
            torch.cuda.synchronize()
            nreq_probe = q // width if layout == "seats" else 1
            seq_len_seen = int(r.input_buffers.seq_lens[:nreq_probe].max().item())
            events = []
            with r.step_timing.collect() as samples:
                for _ in range(iters):
                    h, _ = once()
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    r.model.compute_logits(h[:q])
                    b.record()
                    events.append((a, b))
            torch.cuda.synchronize()
    finally:
        mr_mod.dispatch_cg_and_sync_dp = orig_dispatch
        InputBatch.make_dummy = orig_make
        for cls, own, orig in reversed(patched):
            if own:
                cls.build_for_cudagraph_capture = orig
            else:
                del cls.build_for_cudagraph_capture
    if len(samples) != iters:
        raise RuntimeError(f"{len(samples)} step timings for {iters} steps")
    desc = seen["desc"]
    nreq = q // width if layout == "seats" else 1
    qpr = q // nreq
    # What the step left in the shared seq_lens buffer; depth + q when the context reached the metadata. A
    # record, not a gate (a drafter may reuse the buffer); the DRAM-floor check in the report is the gate.
    context_ok = seq_len_seen == depth + qpr
    bt = r.block_tables
    wrap = 0.0
    for bs, bpk in zip(bt.kernel_block_sizes, bt.blocks_per_kv_block):
        need = nreq * -(-(depth + qpr) // bs)
        wrap = max(wrap, need / (r.kv_cache_config.num_blocks * bpk))
    fwd = [s.forward_ms for s in samples]
    dft = [s.drafter_ms for s in samples]
    lmh = [a.elapsed_time(b) for a, b in events]
    return {
        "forward": fwd, "drafter": dft, "lm_head": lmh,
        "step": [x + y + z for x, y, z in zip(fwd, dft, lmh)],
        "cg_mode": getattr(desc.cg_mode, "name", str(desc.cg_mode)),
        "ran_tokens": int(samples[0].num_target_tokens), "ran_reqs": int(samples[0].num_reqs),
        "full_cudagraph": bool(samples[0].full_cudagraph), "num_reqs": nreq,
        "kv_wrap": round(wrap, 3), "decode_width": width, "seq_len_seen": seq_len_seen, "context_ok": context_ok,
        "capture_builds_restored": seen["restored"],
        "max_num_seqs": int(r.max_num_reqs), "kv_blocks": int(r.kv_cache_config.num_blocks),
    }


def _engine_info(obj):
    r = _runner(obj)
    return {"runner": f"{type(r).__module__}.{type(r).__name__}", "decode_width": int(r.decode_query_len),
            "max_num_seqs": int(r.max_num_reqs), "kv_blocks": int(r.kv_cache_config.num_blocks),
            "kernel_block_sizes": list(r.block_tables.kernel_block_sizes),
            "captured_token_counts": (r.cudagraph_manager.captured_token_counts()
                                      if r.cudagraph_manager is not None else [])}


def build_engine_args(plan):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    argv = list(plan["argv"])
    if argv and not argv[0].startswith("-"):
        argv = ["--model", argv[0]] + argv[1:]
    ns = make_arg_parser(FlexibleArgumentParser(prog="vllm serve")).parse_args(argv)
    if getattr(ns, "model_tag", None):
        ns.model = ns.model_tag
    return EngineArgs.from_cli_args(ns)


def child_main(args):
    plan = json.loads(os.environ[PLAN_ENV])
    name = plan["backend"]
    if args.dry_run:
        import torch
        note = ""
        if not torch.cuda.is_available():
            # The parser builds config defaults that ask the platform for a device type, and a box with
            # no GPU has none. A stub CUDA platform lets the argv parse; nothing here touches a device.
            import vllm.platforms as platforms
            from vllm.platforms.interface import Platform, PlatformEnum

            class _ParseOnly(Platform):
                _enum = PlatformEnum.CUDA
                device_name = device_type = "cuda"
                dispatch_key = "CUDA"

            platforms._current_platform = _ParseOnly()
            note = " (no GPU: parsed against a stub CUDA platform)"
        ea = build_engine_args(plan)
        cc = ea.compilation_config
        print(f"   engine args parse ok{note}: model={ea.model} max_model_len={ea.max_model_len}"
              f" kv_cache_dtype={ea.kv_cache_dtype} block_size={ea.block_size} max_num_seqs={ea.max_num_seqs}"
              f" kv_cache_memory_bytes={getattr(ea, 'kv_cache_memory_bytes', None)}"
              f" max_cudagraph_capture_size={getattr(cc, 'max_cudagraph_capture_size', None)}"
              f" cudagraph_capture_sizes={getattr(cc, 'cudagraph_capture_sizes', None)}")
        return 0
    done = set()
    if args.resume and os.path.exists(args.out):
        for row in read_rows(args.out):
            if (row.get("backend") == name and row.get("graphs") == plan["graphs"] and row.get("rev") == REV
                    and not row.get("error")):
                done.add((row["layout"], row["depth"], row["total_q"]))
    # Nothing is ever prefilled here, so a pool smaller than max_model_len only means wrapped block spans
    # (kv_wrap); skip vLLM's refusal to boot a max_model_len one request could not fill.
    from vllm.v1.core import kv_cache_utils
    kv_cache_utils._check_enough_kv_cache_memory = lambda *a, **k: None
    from vllm.v1.engine.llm_engine import LLMEngine

    t0 = time.time()
    try:
        engine = LLMEngine.from_engine_args(build_engine_args(plan))
        info = engine.collective_rpc(_engine_info)[0]
    except Exception as e:   # noqa: BLE001 - reported as an invalid run
        print(f"step_curve: {name}: engine did not build: {type(e).__name__}: {e}", flush=True)
        return 2
    print(f"step_curve: {name}: engine up in {time.time() - t0:.0f} s: {json.dumps(info)}", flush=True)
    pts = point_list(plan, info["decode_width"], info["max_num_seqs"])
    ok = 0
    with open(args.out, "a") as f:
        for layout, depth, q in pts:
            if (layout, depth, q) in done:
                continue
            row = {"backend": name, "graphs": plan["graphs"], "layout": layout, "depth": depth, "total_q": q,
                   "rev": REV, "max_model_len": plan["max_model_len"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                res = engine.collective_rpc(_measure_point, args=(layout, depth, q, plan["warmup"], plan["iters"]))[0]
                row.update(res)
                ok += 1
                s = stats(res["step"])
                fw = stats(res["forward"])
                print(f"  {name} {layout:6s} depth={depth:>6d} q={q:>2d}  step {s['median']:7.2f} ms "
                      f"[{s['p10']:.2f}-{s['p90']:.2f}]  forward {fw['median']:7.2f}  {res['cg_mode']}"
                      f" ran={res['ran_tokens']}x{res['ran_reqs']}" + (f" kv_wrap={res['kv_wrap']}"
                                                                       if res["kv_wrap"] > 1 else ""), flush=True)
            except Exception as e:   # noqa: BLE001 - recorded per point
                row["error"] = f"{type(e).__name__}: {e}"[:2000]
                print(f"  {name} {layout} depth={depth} q={q}: ERROR {row['error'][:300]}", flush=True)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if "error" in row and any(m in row["error"] for m in ("CUDA error", "illegal memory", "CUBLAS", "cudaError")):
                print(f"step_curve: {name}: CUDA context is poisoned; stopping this backend (--resume skips the done points)",
                      flush=True)
                break
    return 0 if ok or done else 2


# ---------------------------------------------------------------- reporting (CPU)

def read_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest(rows, any_rev=False):
    """The last good row per point, from this revision of the tool only: rows written before the dummy-context
    and GDN-path fixes (no 'rev') measured something else and are never mixed in."""
    by = {}
    for row in rows:
        if row.get("error") or (row.get("rev") != REV and not any_rev):
            continue
        by[(row["backend"], row.get("graphs", "serve"), row["layout"], row["depth"], row["total_q"])] = row
    return by


def write_csv(path, rows, state):
    cols = ["backend", "graphs", "layout", "depth", "total_q", "part", "ms_median", "ms_p10", "ms_p90", "n",
            "cg_mode", "ran_tokens", "num_reqs", "kv_wrap", "seq_len_seen", "valid"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for key in sorted(rows):
            row = rows[key]
            for part in PARTS:
                s = stats(row[part])
                vals = [row["backend"], row.get("graphs", "serve"), row["layout"], row["depth"], row["total_q"],
                        part, f"{s['median']:.4f}", f"{s['p10']:.4f}", f"{s['p90']:.4f}", s["n"], row["cg_mode"],
                        row["ran_tokens"], row["num_reqs"], row["kv_wrap"], row.get("seq_len_seen", ""),
                        state.get(key, "unchecked")]
                f.write(",".join(str(v) for v in vals) + "\n")


def ascii_plot(rows, width=56):
    """Step median per q, one panel per (backend, graphs, layout, depth). The axis starts just below the
    panel's smallest step, so a stair of a few percent is visible; the numbers are step and forward ms."""
    out = []
    panels = sorted({(k[0], k[1], k[2], k[3]) for k in rows})
    for backend, graphs, layout, depth in panels:
        pts = sorted((k[4], rows[k]) for k in rows if k[:4] == (backend, graphs, layout, depth))
        meds = [stats(r["step"])["median"] for _, r in pts]
        lo, top = 0.97 * min(meds), max(meds)
        span = (top - lo) or 1.0
        out.append(f"{backend} graphs={graphs} layout={layout} depth={depth}: step ms, axis {lo:.2f} to {top:.2f}"
                   "  | step  forward  graph mode")
        for (q, r), s in zip(pts, meds):
            f = stats(r["forward"])["median"]
            n = max(int(round(width * (s - lo) / span)), 1)
            pad = "" if r["ran_tokens"] == q else f" ran {r['ran_tokens']}"
            out.append(f"  q={q:>2d} {'#' * n}{' ' * (width - n)} {s:7.2f} {f:7.2f} {r['cg_mode']}{pad}"
                       + (f" wrap {r['kv_wrap']}" if r["kv_wrap"] > 1 else ""))
        out.append("")
    return "\n".join(out)


def table(rows):
    """Step median (ms) per q (rows) and depth (columns), one block per (backend, graphs, layout)."""
    out = []
    for backend, graphs, layout in sorted({k[:3] for k in rows}):
        keys = [k for k in rows if k[:3] == (backend, graphs, layout)]
        depths = sorted({k[3] for k in keys})
        out.append(f"{backend} graphs={graphs} layout={layout}: step ms median (forward ms), '*' = not a FULL graph")
        out.append("    q " + "".join(f"{d:>18d}" for d in depths))
        for q in sorted({k[4] for k in keys}):
            cells = []
            for d in depths:
                r = rows.get((backend, graphs, layout, d, q))
                cells.append(f"{'':>18s}" if r is None else
                             f"{stats(r['step'])['median']:9.2f} ({stats(r['forward'])['median']:5.2f})"
                             + ("*" if r["cg_mode"] != "FULL" else " "))
            out.append(f"  {q:>3d} " + "".join(f"{c:>18s}" for c in cells))
        out.append("")
    return "\n".join(out)


def summary(rows, backend, graphs, verdict):
    """One greppable line per backend: the self-check verdict, then per depth the q=1 step, the q=1->W slope
    as % of the q=1 step per extra token, both ends PIECEWISE (seq1 q=1 against seq1pw q=W: one graph mode;
    runtime-2: under 5% means runtime-1's ceiling holds), and the FULL step at q=W."""
    parts = []
    for depth in sorted({k[3] for k in rows if k[0] == backend and k[1] == graphs}):
        a = rows.get((backend, graphs, "seq1", depth, 1))
        if not a:
            continue
        w = a["decode_width"]
        b = rows.get((backend, graphs, "seq1pw", depth, w))
        full = rows.get((backend, graphs, "seq1", depth, w))
        s1 = stats(a["step"])["median"]
        txt = f"{depth}:{s1:.2f}ms"
        if b and w > 1:
            txt += f"/{(stats(b['step'])['median'] - s1) / (w - 1) / s1 * 100:+.2f}%"
        if full:
            txt += f"/full{w}={stats(full['step'])['median']:.2f}ms"
        parts.append(txt)
    n = sum(1 for k in rows if k[0] == backend and k[1] == graphs)
    return (f"STEP_CURVE backend={backend} graphs={graphs} check={verdict} points={n}"
            " q1ms/slope_pct_per_tok(pw q1->W)/fullW=" + (",".join(parts) or "n/a"))


# Self-check: going from 8k to 150k context, a step must take at least as long again as reading the extra
# KV of the 16 full-attention layers once at the 3090's 936 GB/s. Bytes per token per layer: K and V,
# 4 KV heads x 256 dims, plus per-token-head scales on int8; KVarN's 4-bit K + 2-bit V without its scales
# (a lower bound). A point below its floor attended over less context than it claims.
FLOOR_DEPTHS = (8_000, 150_000)
FULL_ATTN_LAYERS = 16
DRAM_BYTES_PER_S = 936e9
KV_BYTES_PER_TOKEN_LAYER = {"fa2": 2048, "pthfa": 2080, "int8": 2080, "kvarn": 768}


def validate(rows):
    """Returns ({row key: 'ok' | 'INVALID' | 'unchecked'}, {(backend, graphs): 'PASS' | 'FAIL' | 'unchecked'},
    report lines). Each (backend, graphs, layout, q) column is checked on its 8k and 150k points; a failed
    column is INVALID at every depth, and a failed seq1 q=W column (the serving verify step) invalidates the
    whole backend."""
    lo_d, hi_d = FLOOR_DEPTHS
    col_state, lines = {}, []
    for key in sorted({k[:3] + (k[4],) for k in rows}):
        backend, graphs, layout, q = key
        lo = rows.get((backend, graphs, layout, lo_d, q))
        hi = rows.get((backend, graphs, layout, hi_d, q))
        per_tok = KV_BYTES_PER_TOKEN_LAYER.get(backend)
        if not (lo and hi and per_tok):
            col_state[key] = "unchecked"
            continue
        need = hi["num_reqs"] * (hi_d - lo_d) * FULL_ATTN_LAYERS * per_tok / DRAM_BYTES_PER_S * 1000
        # p10, not the median: a slow outlier cluster at 8k (seen on kvarn q=1) would hide a real rise.
        got = stats(hi["step"])["p10"] - stats(lo["step"])["p10"]
        col_state[key] = "ok" if got >= need else "INVALID"
        if got < need:
            lines.append(f"step_curve: INVALID {backend} graphs={graphs} {layout} q={q}: step rises {got:.2f} ms"
                         f" from {lo_d} to {hi_d}, below the {need:.2f} ms it takes to read the extra KV once")
    verdict, whole = {}, set()
    for backend, graphs in sorted({k[:2] for k in rows}):
        cols = {k: v for k, v in col_state.items() if k[:2] == (backend, graphs)}
        widths = {r["decode_width"] for k, r in rows.items() if k[:2] == (backend, graphs)}
        if any(cols.get((backend, graphs, "seq1", w)) == "INVALID" for w in widths):
            whole.add((backend, graphs))
            lines.append(f"step_curve: INVALID {backend} graphs={graphs}: the serving verify step (seq1 q=W) fails"
                         " the floor, so every point of this backend is marked INVALID")
        if any(v == "INVALID" for v in cols.values()):
            verdict[(backend, graphs)] = "FAIL"
        elif any(v == "ok" for v in cols.values()):
            verdict[(backend, graphs)] = "PASS"
        else:
            verdict[(backend, graphs)] = "unchecked"
    state = {k: "INVALID" if k[:2] in whole else col_state.get(k[:3] + (k[4],), "unchecked") for k in rows}
    return state, verdict, lines


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring (head of the file) for method, run command and limits.")
    ap.add_argument("--backend", default="fa2,pthfa,kvarn",
                    help=f"comma list of {','.join(BACKENDS)} (default fa2,pthfa,kvarn); one engine boot each")
    ap.add_argument("--depths", default=None,
                    help="context depths, e.g. 8k,25k,64k (default 8k,25k,64k,112k,150k,240k capped per backend:"
                         " 150k on fa2/pthfa/int8, 240k on kvarn)")
    ap.add_argument("--q", default="1-32", help="total query tokens per step (default 1-32)")
    ap.add_argument("--layouts", default="seq1,seats",
                    help="seq1: one request of q <= W tokens (W = decode width), plus q=W forced PIECEWISE as"
                         " layout seq1pw; seats: q/W requests of W tokens (default both)")
    ap.add_argument("--graphs", choices=("serve", "dense"), default="serve",
                    help="serve: the launcher's capture sizes (default). dense: a graph for every q (no padding)")
    ap.add_argument("--iters", type=int, default=20, help="timed steps per point (default 20)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed steps per point (default 3)")
    ap.add_argument("--out", default=None, help="JSONL path (default bench/results/step_curve-<time>.jsonl);"
                                                " .csv and .plot.txt are written beside it")
    ap.add_argument("--resume", action="store_true", help="append to --out and skip points it already holds")
    ap.add_argument("--launcher", default=os.path.join(REPO, "single-user", "start_qwen.sh"))
    ap.add_argument("--dry-run", action="store_true", help="resolve each backend's launcher argv/env, print the plan")
    ap.add_argument("--report-only", action="store_true",
                    help="no GPU: re-read --out, re-run the self-check, rewrite the CSV/plot and exit as a run would")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.q = parse_ints(args.q)
    args.depths = parse_ints(args.depths) if args.depths else None
    args.layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    bad = [b for b in backends if b not in BACKENDS] + [x for x in args.layouts if x not in ("seq1", "seats")]
    if bad or not args.q or min(args.q) < 1 or args.iters < 1:
        ap.error(f"bad --backend/--layouts/--q/--iters: {bad or args.q}")
    if args.out is None:
        args.out = os.path.join(BENCH, "results", time.strftime("step_curve-%Y%m%d-%H%M%S.jsonl"))
    if args._child:
        rc = child_main(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)   # skip the in-process engine's teardown; the process exit frees the GPU

    stem = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    if args.report_only:
        return report(args, stem, backends, [], any_rev=True)
    py = os.path.join(REPO, "venv", "bin", "python")
    py = py if os.path.exists(py) else sys.executable
    print(f"step_curve: {'dry run, ' if args.dry_run else ''}backends {','.join(backends)}; out {args.out}")
    failed = []
    for name in backends:
        try:
            plan = resolve_backend(name, args)
        except Exception as e:   # noqa: BLE001
            print(f"== backend {name}: launcher did not resolve: {e}")
            failed.append(name)
            continue
        describe(plan)
        child_env = dict(plan["env"])
        child_env[PLAN_ENV] = json.dumps({k: v for k, v in plan.items() if k not in ("env", "launcher_log")})
        cmd = [py, os.path.abspath(__file__), "--_child", "--out", args.out, "--backend", name]
        if args.resume:
            cmd.append("--resume")
        if args.dry_run:
            cmd.append("--dry-run")
            try:
                if subprocess.run(cmd, cwd=REPO, env=child_env, timeout=600).returncode != 0:
                    print(f"   (engine args did not parse in {py}; the launcher plan above still stands)")
            except (OSError, subprocess.TimeoutExpired) as e:
                print(f"   (engine args not checked: {e})")
            continue
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        if subprocess.run(cmd, cwd=REPO, env=child_env).returncode != 0:
            failed.append(name)
    if args.dry_run:
        print(f"step_curve: dry run done; {len(failed)} backend(s) unresolved" + (f": {','.join(failed)}" if failed else ""))
        return 0
    return report(args, stem, backends, failed)


def report(args, stem, backends, failed, any_rev=False):
    rows = latest(read_rows(args.out), any_rev) if os.path.exists(args.out) else {}
    state, verdict, lines = validate(rows)
    if rows:
        write_csv(stem + ".csv", rows, state)
        plot = ascii_plot(rows)
        with open(stem + ".plot.txt", "w") as f:
            f.write(plot + "\n")
        print(table(rows))
        print(f"step_curve: wrote {args.out}, {stem}.csv, {stem}.plot.txt")
    for line in lines:
        print(line)
    bad = failed + [b for b in backends if verdict.get((b, args.graphs)) == "FAIL"]
    for name in backends:
        print(summary(rows, name, args.graphs, verdict.get((name, args.graphs), "unchecked")))
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
