#!/usr/bin/env python3
"""Measure, per linear layer, how much per-token int8 activation quantization
perturbs that layer's output (relative L2 error of y=Wx vs y=W·q(x)) on a
small calibration corpus. Runs vLLM in-process (no multiprocessing) with the
plain W4A16 kernels and forward pre-hooks. Output: JSON {layer_name: err} and
a sorted listing.
Use it to pick INT8_LAYERS for batch/start_qwen.sh: layers with small error are
safe to run with int8 activations, the rest cost perplexity. On Qwen3.8-27B the
GDN in_proj (early layers) and down_proj (last layers) are the worst.
Usage: PATH=venv/bin:$PATH VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
       python bench/act_calib.py models/Qwen3.8-27B-W4A16-AutoRound act_calib.json

--modes per-token,per-128,hadamard-128 (or `all`; quant-1 step 1) measures each named
activation quantizer side by side on the same rows (bench/act_quant.py: per-token,
per-128, per-128-pow2, hadamard-128, hadamard-128-g128, and `floor`, the measurement's
own bf16 noise). In this mode the quant math is fp32 and the dequantized input reaches
the layer at fp32 precision (hi + lo split), so its per-token column reads a little
lower than the default run's; compare modes within one run. It prints a per-layer table
with each mode's ratio to per-token, counts the layers whose error falls by at least
--threshold (default 3, the quant-1 GPU validation bar), and ends with one ACT_CALIB line.
The JSON is {"modes", "threshold", "layers": {layer: {mode: {...}}}, "summary"}.
Exit 0, or 2 if no layer was measured. Without --modes the run is the original one.
"""
import json, os, sys, glob, re, torch, argparse
import pyarrow.parquet as pq

_ap = argparse.ArgumentParser(description="Per-layer error that int8 activations add (see module docstring).")
_ap.add_argument("model_path")
_ap.add_argument("out_path")
_ap.add_argument("--modes", default=None,
                 help="comma list of act_quant.py modes, or 'all'; per-token is always added (default: the original per-token run)")
_ap.add_argument("--threshold", type=float, default=3.0,
                 help="with --modes: a layer passes when per-token err / mode err >= this (default 3)")
_args = _ap.parse_args()
model_path, out_path = _args.model_path, _args.out_path
modes = None
if _args.modes is not None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import act_quant
    try: modes = act_quant.parse_modes(_args.modes)
    except ValueError as e: _ap.error(str(e))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.pop("VLLM_MARLIN_INPUT_DTYPE", None)
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from vllm import LLM, SamplingParams

llm = LLM(model=model_path, served_model_name="q", gpu_memory_utilization=0.85, max_model_len=2048,
          max_num_seqs=4, enforce_eager=True, language_model_only=True, kv_cache_dtype="fp8",
          mamba_ssm_cache_dtype="float16", max_num_batched_tokens=2048)

# find the torch model
def find_model(obj, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if depth > 14 or id(obj) in seen: return None
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        if any(n.endswith("gate_up_proj") for n, _ in obj.named_modules()): return obj
        return None
    for attr in ("llm_engine", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "model", "engine", "_model", "runner"):
        if hasattr(obj, attr):
            try: child = getattr(obj, attr)
            except Exception: continue
            r = find_model(child, depth+1, seen)
            if r is not None: return r
    if hasattr(obj, "get_model"):
        try:
            r = obj.get_model()
            if isinstance(r, torch.nn.Module): return r
        except Exception: pass
    return None
model = find_model(llm)
if model is None:
    # last resort: collective_rpc into the worker
    try:
        model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.get_model()
    except Exception as e:
        print("fallback failed:", e)
assert model is not None, "could not find model"
print("model:", type(model).__name__, flush=True)

pat = re.compile(r"(gate_up_proj|down_proj|in_proj_qkvz|out_proj|qkv_proj|o_proj)$")
stats = {}
mode_stats = {}
def make_mode_hook(name):
    def hook(m, inputs):
        if getattr(m, "_calib_off", False): return
        x = inputs[0]
        x2 = x.reshape(-1, x.shape[-1])
        n = min(x2.shape[0], 64)
        idx = torch.linspace(0, x2.shape[0]-1, n).long().to(x2.device) if x2.shape[0] > n else torch.arange(x2.shape[0], device=x2.device)
        xs = x2[idx]
        def fwd(t):
            y = m(t)
            return y[0] if isinstance(y, tuple) else y
        with torch.no_grad():
            m._calib_off = True
            try:
                errs = act_quant.layer_errors(fwd, xs, modes)
            finally:
                m._calib_off = False
        d = mode_stats.setdefault(name, {})
        for mode, (err, xerr) in errs.items():
            s = d.setdefault(mode, [0.0, 0.0, 0]); s[0] += err; s[1] += xerr; s[2] += 1
    return hook
def make_hook(name, mod):
    def hook(m, inputs):
        x = inputs[0]
        x2 = x.reshape(-1, x.shape[-1])
        n = min(x2.shape[0], 64)
        idx = torch.linspace(0, x2.shape[0]-1, n).long().to(x2.device) if x2.shape[0] > n else torch.arange(x2.shape[0], device=x2.device)
        xs = x2[idx]
        scale = xs.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        q = (torch.round(xs / scale).clamp(-128, 127) * scale).to(xs.dtype)
        with torch.no_grad():
            m._calib_off = True
            y = m(xs); yq = m(q)
            m._calib_off = False
            if isinstance(y, tuple): y = y[0]; yq = yq[0]
            err = ((yq.float() - y.float()).norm() / y.float().norm().clamp(min=1e-8)).item()
            xerr = ((q.float() - xs.float()).norm() / xs.float().norm().clamp(min=1e-8)).item()
        s = stats.setdefault(name, [0.0, 0.0, 0]); s[0] += err; s[1] += xerr; s[2] += 1
    def guarded(m, inputs):
        if getattr(m, "_calib_off", False): return
        return hook(m, inputs)
    return guarded

count = 0
for name, mod in model.named_modules():
    if pat.search(name) and "mtp" not in name and "lm_head" not in name and hasattr(mod, "forward") and "Linear" in type(mod).__name__:
        mod.register_forward_pre_hook(make_hook(name, mod) if modes is None else make_mode_hook(name)); count += 1
print("hooked layers:", count, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.environ.get("QUALITY_DATA", os.path.join(HERE, "quality-data"))
texts = []
t = "".join(pq.read_table(f"{Q}/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet").column("text").to_pylist())
texts += [t[i:i+3000] for i in range(0, 8*3000, 3000)]
tb = pq.read_table(f"{Q}/fineweb2/data/dan_Latn/test/000_00000.parquet", columns=["text"]).column("text").to_pylist()
texts += [d[:3000] for d in tb if len(d) > 3000][:8]
code = sorted(glob.glob(os.path.join(HERE, "..", "venv/lib/python3.12/site-packages/vllm/v1/core/*.py")))
texts += [open(f).read()[:3000] for f in code[:6]]
# chat-formatted prompts too
for p in [json.loads(l)["prompt"] for l in open(os.path.join(HERE, "prompts_real.jsonl"))]:
    texts.append(p)
texts = [t for t in texts if t and t.strip()]
print("calibration texts:", len(texts), flush=True)
sp = SamplingParams(max_tokens=32, temperature=0)
llm.generate(texts, sp)
KINDS = ["gate_up_proj", "down_proj", "in_proj_qkvz", "out_proj", "qkv_proj", "o_proj"]
if modes is not None:
    import statistics
    thr = _args.threshold
    layers = {k: {mode: {"out_err": s[0]/s[2], "act_err": s[1]/s[2], "n": s[2]} for mode, s in d.items()}
              for k, d in mode_stats.items()}
    if not layers:
        print("ACT_CALIB INVALID: no layer was measured", flush=True)
        sys.exit(2)
    base = act_quant.BASELINE
    def ratio(v, mode):
        if mode not in v or base not in v: return None
        return v[base]["out_err"] / max(v[mode]["out_err"], 1e-12)
    cmp_modes = [md for md in modes if md != base and md != act_quant.FLOOR]
    print(f"per-layer relative output error (ratio = per-token / mode; >= {thr:g} passes):")
    print("  " + "  ".join(f"{md:>20s}" for md in modes) + "  layer")
    for k, v in layers.items():
        cells = []
        for md in modes:
            if md not in v: cells.append(f"{'-':>20s}"); continue
            r = ratio(v, md)
            cell = f"{v[md]['out_err']:.4f}" + (f" ({r:5.1f}x)" if r is not None and md in cmp_modes else "")
            cells.append(f"{cell:>20s}")
        print("  " + "  ".join(cells) + f"  {k}")
    summary = {}
    for md in modes:
        per_kind = {}
        for kind in KINDS:
            ks = [k for k in layers if k.endswith(kind) and md in layers[k]]
            if not ks: continue
            vals = [layers[k][md]["out_err"] for k in ks]
            rs = [ratio(layers[k], md) for k in ks]
            per_kind[kind] = {"n": len(ks), "median": statistics.median(vals), "max": max(vals),
                              "ge_threshold": sum(1 for r in rs if r is not None and r >= thr) if md in cmp_modes else None,
                              "median_ratio": statistics.median(rs) if md in cmp_modes else None}
        tot = [k for k in layers if md in layers[k]]
        summary[md] = {"n": len(tot), "per_kind": per_kind,
                       "ge_threshold": sum(1 for k in tot if (ratio(layers[k], md) or 0) >= thr) if md in cmp_modes else None}
        for kind, s in per_kind.items():
            extra = f" >= {thr:g}x: {s['ge_threshold']}/{s['n']} median ratio {s['median_ratio']:.2f}x" if md in cmp_modes else ""
            print(f"{md}: {kind}: n={s['n']} median={s['median']:.4f} max={s['max']:.4f}{extra}")
    # quant-1 GPU validation: down_proj and GDN in_proj errors fall >= 3x, on every such layer
    gate = {}
    for md in cmp_modes:
        pk = summary[md]["per_kind"]
        gk = [pk[kd] for kd in ("down_proj", "in_proj_qkvz") if kd in pk]
        gate[md] = bool(gk) and all(s["ge_threshold"] == s["n"] for s in gk)
    json.dump({"modes": modes, "threshold": thr, "layers": layers, "summary": summary, "gate_down_in_proj": gate},
              open(out_path, "w"), indent=1)
    parts = []
    for md in cmp_modes:
        pk = summary[md]["per_kind"]
        kinds = ",".join(f"{kd}={pk[kd]['ge_threshold']}/{pk[kd]['n']}" for kd in KINDS if kd in pk)
        parts.append(f"{md}:ge{thr:g}x={summary[md]['ge_threshold']}/{summary[md]['n']},{kinds},gate={'yes' if gate[md] else 'no'}")
    if act_quant.FLOOR in modes:
        fl = [v[act_quant.FLOOR]["out_err"] for v in layers.values() if act_quant.FLOOR in v]
        parts.append(f"floor:median={statistics.median(fl):.5f},max={max(fl):.5f}")
    print(f"ACT_CALIB layers={len(layers)} " + " ".join(parts), flush=True)
    sys.exit(0)
res = {k: {"out_err": v[0]/v[2], "act_err": v[1]/v[2], "n": v[2]} for k, v in stats.items()}
json.dump(res, open(out_path, "w"), indent=1)
rows = sorted(res.items(), key=lambda kv: -kv[1]["out_err"])
print("worst 25 layers by relative output error from per-token int8 activations:")
for k, v in rows[:25]: print(f"  {v['out_err']:.4f}  (act {v['act_err']:.4f})  {k}")
import statistics
for kind in ["gate_up_proj", "down_proj", "in_proj_qkvz", "out_proj", "qkv_proj", "o_proj"]:
    vals = [v["out_err"] for k, v in res.items() if k.endswith(kind)]
    if vals: print(f"{kind}: n={len(vals)} median={statistics.median(vals):.4f} max={max(vals):.4f}")
