"""quant-4 check: the online block-Hadamard-128 input rotation of
docs/research/deepinfer/experiments/patches/quant4-act-hadamard.patch (VLLM_ACT_HADAMARD_RE).

Kernel check (default; GPU, ~1 min, no model):
  CUDA_VISIBLE_DEVICES=0 venv/bin/python bench/test_act_hadamard.py
Synthetic AutoRound-signed int4 weights W and heavy-tailed inputs (bench/act_quant.py's
make_acts, down_proj-like and in_proj-like). A rotated layer holds the RTN requant of
W . H / 16 and has act_hadamard_block = 128 (what the scheme sets for matching layers). For the
bf16 and the int8 input path it checks:
  1. the rotated layer's output is bitwise equal to an unflagged layer with the same weights fed
     x . H / 8 by hand: the patch adds only the rotation;
  2. bf16 path: its error against the fp32 reference (x . H/8) . W'^T is at the bf16 floor (< 1%);
  3. int8 path: the error the int8 activations add, against each layer's own fp32 reference, falls
     >= 3x with the rotation (Hadamard-128 on the unchanged per-token Marlin W4A8 kernel).

Model check (GPU, one arm per process, ~3 min):
  [env of the arm] venv/bin/python bench/test_act_hadamard.py --model <dir> --ref <unrotated dir> \\
      --out arm.json [--expect q4_report.json]
Runs the model in-process (enforce_eager, like bench/act_calib.py) with whatever
VLLM_ACT_HADAMARD_RE / VLLM_MARLIN_INPUT_DTYPE / VLLM_MARLIN_INT8_INCLUDE_RE the caller set,
hooks every --layers-re layer, and compares its real output with y_ref = x . W^T computed in fp32
from the --ref checkpoint's int4 weights (x = the layer input before the rotation). Per layer that
is: unrotated W4A16 -> the bf16 floor; unrotated INT8 -> today's a_pt; rotated W4A16 -> the
requant error w; rotated INT8 -> the quant-4 arm tot. --expect compares the medians with the
builder's report (quant4_h128down.py --out). Ends with one TEST_ACT_HADAMARD line.

Exit 0 pass, 1 fail (or the patch is missing), 2 no CUDA device.
"""
import argparse, json, os, re, statistics, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import act_quant  # noqa: E402


def find_model(obj, depth=0, seen=None):
    """bench/act_calib.py's walk from the in-process LLM to the torch model."""
    seen = seen if seen is not None else set()
    if depth > 14 or id(obj) in seen:
        return None
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        return obj if any(n.endswith("down_proj") for n, _ in obj.named_modules()) else None
    for attr in ("llm_engine", "engine_core", "model_executor", "driver_worker", "worker", "model_runner",
                 "model", "engine", "_model", "runner"):
        if hasattr(obj, attr):
            try:
                r = find_model(getattr(obj, attr), depth + 1, seen)
            except Exception:
                continue
            if r is not None:
                return r
    if hasattr(obj, "get_model"):
        try:
            r = obj.get_model()
            if isinstance(r, torch.nn.Module):
                return r
        except Exception:
            pass
    return None


GROUP = 128
KNOB = "VLLM_ACT_HADAMARD_RE"


def hpm1(device):
    h = torch.ones(1, 1)
    while h.shape[0] < GROUP:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def sym_int4(w):
    """AutoRound-signed RTN: scale = -extreme/8 per group. Returns codes [N, K] int8, bf16 scales, dequant."""
    N, K = w.shape
    g = w.reshape(N, K // GROUP, GROUP)
    ext = g.gather(-1, g.abs().argmax(-1, keepdim=True))
    s = (-ext / 8).to(torch.bfloat16)
    q = torch.clamp(torch.round(g / s.float()), -8, 7)
    return q.reshape(N, K).to(torch.int8), s.reshape(N, K // GROUP), (q * s.float()).reshape(N, K)


def kernel_check():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
    from vllm.model_executor.kernels.linear.mixed_precision import MPLinearLayerConfig
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
    from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
    from vllm.scalar_type import scalar_types
    from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
    import inspect
    from vllm.model_executor.kernels.linear.mixed_precision import marlin as marlin_mod
    if "act_hadamard" not in inspect.getsource(marlin_mod.MarlinLinearKernel.apply_weights):
        print("FAIL: quant4-act-hadamard.patch is not applied to this vllm", flush=True)
        return False
    dev = "cuda"
    set_current_vllm_config(VllmConfig()).__enter__()
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29519", local_rank=0)
    ensure_model_parallel_initialized(1, 1)
    wl = lambda *a, **k: None
    H = hpm1(dev)

    def build(K, N, q, s, act, rotated):
        layer = torch.nn.Module()
        layer.params_dtype = torch.bfloat16
        layer.weight_packed = PackedvLLMParameter(data=pack_to_int32(q, 4, packed_dim=1).contiguous().to(dev),
                                                  input_dim=1, output_dim=0, packed_dim=1, packed_factor=8,
                                                  weight_loader=wl)
        layer.weight_scale = GroupQuantScaleParameter(data=s.contiguous().to(dev), output_dim=0, input_dim=1,
                                                      weight_loader=wl)
        if rotated:
            layer.act_hadamard_block = GROUP
        cfg = MPLinearLayerConfig(full_weight_shape=(K, N), partition_weight_shape=(K, N),
                                  weight_type=scalar_types.uint4b8, act_type=act, group_size=GROUP,
                                  zero_points=False)
        ok, why = MarlinLinearKernel.can_implement(cfg)
        assert ok, why
        k = MarlinLinearKernel(cfg, w_q_param_name="weight_packed", w_s_param_name="weight_scale",
                               w_zp_param_name="weight_zero_point")
        k.process_weights_after_loading(layer)
        assert rotated == (getattr(layer, "act_hadamard", None) is not None)
        return layer, k

    cases = [("down_proj-like", 17408, 5120, 2.5, 16, 30.0), ("in_proj-like", 5120, 12288, 5.0, 6, 40.0),
             ("gaussian", 5120, 6144, 1e6, 0, 1.0)]
    ok = True
    for i, (name, K, N, df, n_out, mag) in enumerate(cases):
        torch.manual_seed(i)
        w = torch.randn(N, K) * 0.02
        q, s, D = sym_int4(w)
        qr, sr, Dr = sym_int4(act_quant.had_block(D, GROUP, hpm1("cpu")) / 16)     # W' = RTN(W . H / 16)
        D, Dr = D.to(dev), Dr.to(dev)
        layers = {(act, rot): build(K, N, qr if rot else q, sr if rot else s, act, rot)
                  for act in (torch.bfloat16, torch.int8) for rot in (False, True)}
        for M in (1, 16, 64, 512):
            x = act_quant.make_acts(M, K, df, n_out, mag, 100 + M).to(dev).to(torch.bfloat16)
            xr = torch.matmul(x.unflatten(-1, (-1, GROUP)), (H / 8).to(torch.bfloat16)).flatten(-2)
            ref = x.float() @ D.t()
            ref_r = xr.float() @ Dr.t()
            out, line = {}, []
            for act in (torch.bfloat16, torch.int8):
                lr, kr = layers[(act, True)]
                lp, kp = layers[(act, False)]
                y_rot = kr.apply_weights(lr, x)
                lp_r, kp_r = build(K, N, qr, sr, act, False) if M == 1 else (None, None)
                if M == 1:
                    y_hand = kp_r.apply_weights(lp_r, xr)
                    same = torch.equal(y_rot, y_hand)
                    ok &= same
                    line.append(f"{'int8' if act == torch.int8 else 'bf16'} rotated==by-hand {same}")
                out[act] = (rel(y_rot, ref_r), rel(kp.apply_weights(lp, x), ref))
            e_bf16 = out[torch.bfloat16][0]
            e_rot, e_pt = out[torch.int8]
            good = e_bf16 < 0.01 and (e_pt / e_rot >= 3.0 or n_out == 0)
            ok &= good
            print(f"{name:15s} K={K:5d} N={N:5d} M={M:3d}  {'  '.join(line)}  bf16 rotated err {e_bf16:.4f}  "
                  f"int8 err: per-token {e_pt:.4f}  rotated {e_rot:.4f} ({e_pt / e_rot:.1f}x)  "
                  f"{'ok' if good else 'FAIL'}", flush=True)
        del layers
    return ok


def model_check(a):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from safetensors import safe_open
    from vllm import LLM, SamplingParams
    rotated_dir = os.path.exists(os.path.join(a.model, "quant4_hadamard.json"))
    knob = os.environ.get(KNOB, "")
    print(f"model {a.model} rotated dir: {rotated_dir}  {KNOB}={knob!r}  "
          f"VLLM_MARLIN_INPUT_DTYPE={os.environ.get('VLLM_MARLIN_INPUT_DTYPE')!r}  "
          f"VLLM_MARLIN_INT8_INCLUDE_RE={os.environ.get('VLLM_MARLIN_INT8_INCLUDE_RE')!r}", flush=True)
    llm = LLM(model=a.model, served_model_name="q", gpu_memory_utilization=0.85, max_model_len=2048,
              max_num_seqs=4, enforce_eager=True, language_model_only=True, kv_cache_dtype="fp8",
              mamba_ssm_cache_dtype="float16", max_num_batched_tokens=2048)
    model = find_model(llm)
    assert model is not None, "could not find the torch model"
    lre = re.compile(a.layers_re)
    caps = {}

    def make_hook(name):
        def hook(m, inputs, output):
            c = caps.setdefault(name, [])
            if len(c) >= a.calls:
                return
            x = inputs[0].reshape(-1, inputs[0].shape[-1])
            y = (output[0] if isinstance(output, tuple) else output).reshape(x.shape[0], -1)
            idx = torch.linspace(0, x.shape[0] - 1, min(x.shape[0], a.rows_per_call)).long().to(x.device)
            c.append((x[idx].cpu(), y[idx].cpu()))
        return hook
    n = 0
    for name, mod in model.named_modules():
        if lre.search(name) and "mtp" not in name and "Linear" in type(mod).__name__:
            mod.register_forward_hook(make_hook(name)); n += 1
    print("hooked layers:", n, flush=True)
    prompts = [json.loads(l)["prompt"] for l in open(os.path.join(HERE, "prompts_real.jsonl"))]
    llm.generate(prompts, SamplingParams(max_tokens=16, temperature=0))

    idx = json.load(open(os.path.join(a.ref, "model.safetensors.index.json")))["weight_map"]
    res = {}
    for name, c in sorted(caps.items(), key=lambda kv: int(re.search(r"layers\.(\d+)\.", kv[0]).group(1))):
        mi = re.search(r"layers\.(\d+)\.(.*)$", name)
        base = next(k[: -len(".weight_packed")] for k in idx if k.endswith(".weight_packed") and "mtp" not in k
                    and re.search(rf"layers\.{mi.group(1)}\.{re.escape(mi.group(2))}\.weight_packed$", k))
        with safe_open(os.path.join(a.ref, idx[base + ".weight_packed"]), "pt") as f:
            p = f.get_tensor(base + ".weight_packed").cuda()
            s = f.get_tensor(base + ".weight_scale").cuda().float()
        u = ((p.unsqueeze(-1) >> torch.arange(0, 32, 4, dtype=torch.int32, device=p.device)) & 0xF).reshape(p.shape[0], -1)
        W = ((u - 8).float().view(p.shape[0], -1, GROUP) * s[..., None]).view(p.shape[0], -1)
        x = torch.cat([t[0] for t in c]).cuda()
        y = torch.cat([t[1] for t in c]).cuda()
        res[name] = {"err": rel(y, x.float() @ W.t()), "rows": x.shape[0]}
        del p, s, u, W, x, y
    errs = [v["err"] for v in res.values()]
    med = statistics.median(errs)
    summary = {"layers": len(res), "median": med, "max": max(errs), "rotated_dir": rotated_dir, "knob": knob,
               "input_dtype": os.environ.get("VLLM_MARLIN_INPUT_DTYPE"),
               "int8_include_re": os.environ.get("VLLM_MARLIN_INT8_INCLUDE_RE")}
    for k, v in res.items():
        print(f"  {v['err']:.4f}  {k}")
    ok = bool(res) and max(errs) < 0.5          # a rotation mismatch (knob vs dir) reads ~1.4
    if rotated_dir != bool(knob):
        print(f"NOTE: {KNOB} {'set' if knob else 'unset'} on a {'rotated' if rotated_dir else 'plain'} dir: "
              "this arm is the negative control and should FAIL", flush=True)
    if a.expect and rotated_dir and knob:
        rep = json.load(open(a.expect))["summary"]["median"]
        key = "tot" if os.environ.get("VLLM_MARLIN_INPUT_DTYPE") == "int8" else "w"
        summary["expect"] = {key: rep[key]}
        agree = abs(med - rep[key]) <= 0.25 * rep[key]
        summary["expect_agree"] = agree
        print(f"builder report median {key} {rep[key]:.4f} vs real kernel {med:.4f}: "
              f"{'agree' if agree else 'DISAGREE'} (+-25%)", flush=True)
        ok &= agree
    if a.out:
        json.dump({"summary": summary, "layers": res}, open(a.out, "w"), indent=1)
    print(f"TEST_ACT_HADAMARD model layers={len(res)} median={med:.4f} max={max(errs):.4f} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description="quant-4 online Hadamard rotation check (see docstring)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--ref", default=None, help="unrotated source dir for the reference weights")
    ap.add_argument("--layers-re", default=r"layers\.\d+\.mlp\.down_proj$")
    ap.add_argument("--calls", type=int, default=8, help="forward calls captured per layer")
    ap.add_argument("--rows-per-call", type=int, default=32)
    ap.add_argument("--expect", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device", flush=True)
        sys.exit(2)
    if a.model:
        if not a.ref:
            ap.error("--model needs --ref")
        ok = model_check(a)
    else:
        ok = kernel_check()
        print(f"TEST_ACT_HADAMARD kernel {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
