#!/usr/bin/env python3
"""quant-1 route A probe: can vLLM 0.31's Humming kernels run this checkpoint's linears with
per-token x per-128 int8 activations on this GPU, correctly, and how fast against Marlin?

  # inside the image, on the GPU, with no server holding the card:
  docker run --rm --gpus all -v "$PWD/bench:/app/bench" -v "$PWD/models:/app/models:ro" \
    --entrypoint /app/venv/bin/python ghcr.io/syv-ai/hyperqwen:latest \
    /app/bench/humming_probe.py --out /app/bench/humming_probe.json

Four parts, each printed as it runs:
1. import: is `humming` importable, its version, vLLM's has_humming().
2. resolve: the input schema vLLM builds for a compressed-tensors `input_activations`
   {int8, strategy group, group_size 128, dynamic, symmetric} next to this checkpoint's
   int4 g128 symmetric weights, after vLLM's check_and_fallback_input_schema (the call
   prepare_humming_linear_layer_config makes, allow_fallback=True). With no GPU it runs
   with sm86 stubbed in; on a GPU, on the real device. If it resolves to bf16 activations,
   route A as shipped runs W4A16 even with `--linear-backend humming`.
3. routing (GPU): which mixed-precision kernel choose_mp_linear_kernel picks for a W4A16
   g128 layer with --linear-backend auto and with humming.
4. kernels (GPU): for each linear shape of the model (from its config.json) builds one
   random int4 g128 weight in the compressed-tensors layout and runs it through vLLM's own
   MPLinear kernels: marlin-w4a16, marlin-w4a8-int8 (HyperQwen's INT8_ACT), humming-w4a16,
   humming-g128-vllm (the int8 g128 input config, vLLM's fallback left on),
   humming-int8-g128 and humming-int8-token (fallback off, which is what a HyperQwen patch
   would do). Correctness on 64 rows of gaussian and of heavy-tailed activations (check a's
   in_proj-like generator): each arm against the fp32 GEMM of its own activation quantizer
   (bench/act_quant.py) on the same dequantized weight. humming-g128-vllm reads OFF on the
   heavy rows when vLLM's fallback has dropped the int8 config (it then runs bf16
   activations); it is reported, not gated. Timing: median us per call over M
   (CUDA graph replay; eager if capture fails), and the speed of the humming int8 arms
   against marlin-w4a16 and marlin-w4a8-int8 (>1 = Humming faster).

Exit: 0 measured (or `skip: no CUDA device` after parts 1-2); 1 route A blocked (humming
not importable, humming-int8-g128 fails to run, or an int8/w4a16 Humming arm is off its
reference by more than --bound); 2 invalid (no config, or marlin-w4a16 itself is off its
reference, so the weight layout or reference is wrong); 3 negative control failed (on the
heavy-tailed rows the g128 reference is within 1.5x --bound of bf16 activations, so the check
could not catch a kernel that ignored the input quant). Last line: HUMMING_PROBE ...
"""
import argparse, json, math, os, statistics, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_CONFIG = "/app/models/Qwen3.8-27B-W4A16-AutoRound-fast/config.json"
GROUP = 128
WEIGHT_QCFG = {"quant_method": "humming", "dtype": "int4", "group_size": GROUP, "has_zero_point": False}


def input_qcfg(strategy, group_size):
    """What CompressedTensorsWNA8Int._build_input_quant_config hands Humming."""
    return {"num_bits": 8, "type": "int", "strategy": strategy, "symmetric": True, "dynamic": True,
            "group_size": group_size, "quant_method": "compressed-tensors", "format": "int-quantized"}


G128_QCFG = input_qcfg("group", GROUP)
TOKEN_QCFG = input_qcfg("token", 0)


def model_shapes(cfg_path):
    """(name, N, K) of every linear the model runs per layer, plus lm_head."""
    c = json.load(open(cfg_path))
    t = c.get("text_config", c)
    H, I = t["hidden_size"], t["intermediate_size"]
    nk, dk = t["linear_num_key_heads"], t["linear_key_head_dim"]
    nv, dv = t["linear_num_value_heads"], t["linear_value_head_dim"]
    nh, nkv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    q_heads = nh * (2 if t.get("attn_output_gate") else 1)
    return [
        ("gdn.in_proj_qkvz", 2 * nk * dk + 2 * nv * dv, H),
        ("gdn.in_proj_ba", 2 * nv, H),
        ("gdn.out_proj", H, nv * dv),
        ("attn.qkv_proj", (q_heads + 2 * nkv) * hd, H),
        ("attn.o_proj", H, nh * hd),
        ("mlp.gate_up_proj", 2 * I, H),
        ("mlp.down_proj", H, I),
        ("lm_head", t["vocab_size"], H),
    ]


DEFAULT_SHAPES = "gdn.in_proj_qkvz,gdn.out_proj,attn.qkv_proj,attn.o_proj,mlp.gate_up_proj,mlp.down_proj"


def describe(schema):
    a = getattr(schema, "a_dtype", None)
    mode = getattr(schema, "input_quant_mode", None)
    g = getattr(schema, "input_scale_group_size", 0)
    mode = getattr(mode, "value", mode)
    return f"{a if a is not None else 'bf16'}/{mode or 'none'}/g{g}"


def resolve_part(on_gpu):
    """Part 2: what vLLM's fallback does to the int8 group input config."""
    from vllm.utils.humming import BaseInputSchema, BaseWeightSchema
    import vllm.model_executor.layers.quantization.utils.humming.schema as vs
    import humming.schema.humming as hs
    import torch
    restore = []
    if not on_gpu:   # sm86 stubbed in: the function only reads the capability and sm_version
        from types import SimpleNamespace
        from vllm.platforms.interface import DeviceCapability
        plat = vs.current_platform
        restore.append((plat, "get_device_capability", plat.get_device_capability))
        plat.get_device_capability = lambda device_id=0: DeviceCapability(8, 6)
        restore.append((hs, "current_device", hs.current_device))
        hs.current_device = SimpleNamespace(sm_version=86)
    out = {"device": "real" if on_gpu else "sm86 stub"}
    try:
        w = BaseWeightSchema.from_config(dict(WEIGHT_QCFG))
        for name, cfg in (("int8_g128", G128_QCFG), ("int8_token", TOKEN_QCFG)):
            i = BaseInputSchema.from_config(dict(cfg))
            requested = i.to_humming_schema(torch.bfloat16)
            out[name] = {
                "requested": describe(requested),
                "compatible": bool(requested.is_compatible_with(w.to_humming_schema(torch.bfloat16), torch.bfloat16)),
                "vllm_default": describe(vs.check_and_fallback_input_schema(w, i, torch.bfloat16)),
                "no_fallback": describe(vs.check_and_fallback_input_schema(w, i, torch.bfloat16, allow_fallback=False)),
            }
    finally:
        for obj, attr, val in reversed(restore):
            setattr(obj, attr, val)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--model-config", default=None,
                    help=f"model config.json for the shapes (default {DEFAULT_CONFIG}, else models/ next to bench/)")
    ap.add_argument("--shapes", default=DEFAULT_SHAPES,
                    help=f"comma list of shape names, or 'all' (adds gdn.in_proj_ba and lm_head). Default: {DEFAULT_SHAPES}")
    ap.add_argument("--m", default="1,16,64,256,2048", help="token counts M to time (default 1,16,64,256,2048)")
    ap.add_argument("--iters", type=int, default=50, help="calls per timing sample (5 samples; default 50)")
    ap.add_argument("--warmup", type=int, default=5, help="eager warmup calls per arm and M; the first also JIT-compiles Humming (default 5)")
    ap.add_argument("--no-graphs", action="store_true", help="time eager calls instead of CUDA graph replay")
    ap.add_argument("--bound", type=float, default=0.01,
                    help="max relative error of an arm against its own fp32 reference (default 0.01)")
    ap.add_argument("--cpu-only", action="store_true", help="run parts 1-2 only, even with a GPU")
    ap.add_argument("--out", default=None, help="write the full result as JSON here")
    args = ap.parse_args()

    res = {"import": {}, "resolve": None, "routing": None, "kernels": [], "verdict": None}
    summary = {}

    def finish(code, verdict):
        res["verdict"] = verdict
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)
        print("HUMMING_PROBE " + " ".join(f"{k}={v}" for k, v in summary.items()) + f" verdict={verdict}", flush=True)
        sys.exit(code)

    # 1. import
    print("== import", flush=True)
    try:
        import humming
        res["import"] = {"importable": True, "version": humming.__version__, "path": os.path.dirname(humming.__file__)}
    except Exception as e:
        res["import"] = {"importable": False, "error": f"{type(e).__name__}: {e}"}
    try:
        from vllm.utils.import_utils import has_humming
        res["import"]["vllm_has_humming"] = bool(has_humming())
    except Exception as e:
        res["import"]["vllm_has_humming"] = f"error: {e}"
    print(json.dumps(res["import"]), flush=True)
    summary["importable"] = "yes" if res["import"]["importable"] else "no"
    if not res["import"]["importable"]:
        finish(1, "BLOCKED-not-importable")
    summary["version"] = res["import"]["version"]

    import torch
    on_gpu = torch.cuda.is_available() and not args.cpu_only

    # 2. resolve
    print("== resolve (vLLM's input-schema fallback for int4 g128 weights)", flush=True)
    try:
        res["resolve"] = resolve_part(on_gpu)
        for name in ("int8_g128", "int8_token"):
            r = res["resolve"][name]
            print(f"  {name}: requested {r['requested']} compatible={r['compatible']}  "
                  f"vLLM default -> {r['vllm_default']}  allow_fallback=False -> {r['no_fallback']}", flush=True)
        summary["vllm_resolves_int8_g128"] = res["resolve"]["int8_g128"]["vllm_default"]
    except Exception as e:
        traceback.print_exc()
        res["resolve"] = {"error": f"{type(e).__name__}: {e}"}
        summary["vllm_resolves_int8_g128"] = "error"

    if not on_gpu:
        print("skip: no CUDA device (or --cpu-only); routing and kernel parts not run", flush=True)
        finish(0, "SKIP-no-gpu")

    cfg_path = args.model_config or next(
        (p for p in (DEFAULT_CONFIG, os.path.join(HERE, "..", "models", "Qwen3.8-27B-W4A16-AutoRound-fast", "config.json"))
         if os.path.exists(p)), None)
    if cfg_path is None or not os.path.exists(cfg_path):
        print(f"INVALID: no model config.json (tried {args.model_config or DEFAULT_CONFIG}); pass --model-config", flush=True)
        finish(2, "INVALID-no-config")
    all_shapes = model_shapes(cfg_path)
    want = [s for s in all_shapes] if args.shapes == "all" else [s for s in all_shapes if s[0] in args.shapes.split(",")]
    unknown = set(args.shapes.split(",")) - {s[0] for s in all_shapes} - {"all"}
    if unknown or not want:
        ap.error(f"unknown shape(s) {sorted(unknown)}; choose from {[s[0] for s in all_shapes]}")
    Ms = [int(m) for m in args.m.split(",")]

    import act_quant
    from vllm.config import VllmConfig, KernelConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
    from vllm.model_executor.kernels.linear.mixed_precision import MPLinearLayerConfig
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
    from vllm.model_executor.kernels.linear.mixed_precision.humming import HummingLinearKernel
    import vllm.model_executor.layers.quantization.utils.humming.linear as hl
    from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
    from vllm.scalar_type import scalar_types
    from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

    dev = "cuda"
    cap = torch.cuda.get_device_capability()
    summary["device"] = f"sm{cap[0]}{cap[1]}"
    print(f"== device {torch.cuda.get_device_name()} sm{cap[0]}{cap[1]}; shapes from {cfg_path}", flush=True)

    # vLLM's parameter classes ask for the TP rank: a 1-process group, as in vLLM's kernel tests.
    base_cfg = set_current_vllm_config(VllmConfig())
    base_cfg.__enter__()
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29518", local_rank=0)
    ensure_model_parallel_initialized(1, 1)

    def mp_cfg(K, N, act):
        return MPLinearLayerConfig(full_weight_shape=(K, N), partition_weight_shape=(K, N),
                                   weight_type=scalar_types.uint4b8, act_type=act, group_size=GROUP, zero_points=False)

    # 3. routing
    print("== routing (choose_mp_linear_kernel, W4A16 g128)", flush=True)
    routing = {}
    for backend in ("auto", "humming"):
        try:
            with set_current_vllm_config(VllmConfig(kernel_config=KernelConfig(linear_backend=backend))):
                routing[backend] = choose_mp_linear_kernel(mp_cfg(5120, 5120, torch.bfloat16)).__name__
        except Exception as e:
            routing[backend] = f"error: {type(e).__name__}: {e}"
        print(f"  --linear-backend {backend}: {routing[backend]}", flush=True)
    res["routing"] = routing
    summary["routing_auto"] = routing["auto"]
    summary["routing_humming"] = routing["humming"]

    # 4. kernels
    wl = lambda *a, **k: None

    def make_layer(packed, scale, K, N):
        layer = torch.nn.Module()
        layer.weight_packed = PackedvLLMParameter(data=packed.clone(), input_dim=1, output_dim=0,
                                                  packed_dim=1, packed_factor=8, weight_loader=wl)
        layer.weight_scale = GroupQuantScaleParameter(data=scale.clone(), output_dim=0, input_dim=1, weight_loader=wl)
        layer.input_size_per_partition = K
        layer.input_size = K
        layer.output_size = N
        layer.output_partition_sizes = [N]
        layer.params_dtype = torch.bfloat16
        layer.has_bias = False
        return layer

    ARMS = [  # name, kernel, act_type, humming input config, vLLM fallback on
        ("marlin-w4a16", MarlinLinearKernel, torch.bfloat16, None, True),
        ("marlin-w4a8-int8", MarlinLinearKernel, torch.int8, None, True),
        ("humming-w4a16", HummingLinearKernel, torch.bfloat16, None, True),
        ("humming-g128-vllm", HummingLinearKernel, torch.bfloat16, G128_QCFG, True),
        ("humming-int8-g128", HummingLinearKernel, torch.bfloat16, G128_QCFG, False),
        ("humming-int8-token", HummingLinearKernel, torch.bfloat16, TOKEN_QCFG, False),
    ]
    # the activation quantizer each arm should match
    EMUL = {"marlin-w4a16": "x", "humming-w4a16": "x", "humming-g128-vllm": "per-128",
            "marlin-w4a8-int8": "per-token", "humming-int8-g128": "per-128", "humming-int8-token": "per-token"}
    GATED = ("humming-w4a16", "humming-int8-g128", "humming-int8-token")   # off the bound -> route A FAIL

    def build(arm, packed, scale, K, N):
        name, kcls, act, icfg, fallback = arm
        layer = make_layer(packed, scale, K, N)
        if icfg is not None:
            layer._humming_input_quant_config = dict(icfg)
        k = kcls(mp_cfg(K, N, act), w_q_param_name="weight_packed", w_s_param_name="weight_scale")
        orig = hl.check_and_fallback_input_schema
        if not fallback:
            def no_fallback(*a, **kw):
                kw["allow_fallback"] = False
                return orig(*a, **kw)
            hl.check_and_fallback_input_schema = no_fallback
        try:
            k.process_weights_after_loading(layer)
        finally:
            hl.check_and_fallback_input_schema = orig
        resolved = None
        if hasattr(k, "layer_config"):
            resolved = describe(k.layer_config)
        return (lambda x: k.apply_weights(layer, x)), resolved

    def reference(x, q, s, K, N):
        """fp32 GEMM of x, per-token(x) and per-128(x) with the dequantized weight."""
        xf = x.float()
        ins = {"x": xf, "per-token": act_quant.int8_per_token(xf), "per-128": act_quant.int8_group(xf, GROUP)}
        outs = {k: torch.empty(x.shape[0], N, device=dev) for k in ins}
        for n0 in range(0, N, 8192):
            n1 = min(N, n0 + 8192)
            w = (q[n0:n1].float().view(n1 - n0, K // GROUP, GROUP) * s[n0:n1].float().unsqueeze(-1)).view(n1 - n0, K)
            for k, xi in ins.items():
                outs[k][:, n0:n1] = xi @ w.t()
        return outs

    def time_call(fn, x):
        for _ in range(args.warmup):
            fn(x)
        torch.cuda.synchronize()
        call, how = (lambda: fn(x)), "eager"
        if not args.no_graphs:
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        fn(x)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn(x)
                call, how = g.replay, "graph"
            except Exception as e:
                torch.cuda.synchronize()
                how = f"eager (graph capture failed: {type(e).__name__})"
        samples = []
        for _ in range(5):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(args.iters):
                call()
            b.record()
            torch.cuda.synchronize()
            samples.append(a.elapsed_time(b) * 1000.0 / args.iters)
        return statistics.median(samples), how

    fails, invalid, negctl_bad, ran_g128 = [], [], [], False
    for si, (sname, N, K) in enumerate(want):
        print(f"== {sname}: N={N} K={K}", flush=True)
        if K % GROUP:
            print(f"  skip: K={K} is not a multiple of {GROUP}", flush=True)
            continue
        torch.manual_seed(1000 + si)
        q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
        scale = (torch.rand(N, K // GROUP, device=dev) * 0.002 + 0.001).to(torch.bfloat16)
        packed = pack_to_int32(q, 4, packed_dim=1).contiguous()
        g = torch.Generator().manual_seed(1)
        inputs = {
            "gauss": torch.randn(64, K, generator=g).to(torch.bfloat16).to(dev),
            "heavy": act_quant.make_acts(64, K, 5.0, 6, 40.0, 1).to(torch.bfloat16).to(dev),
        }
        refs = {kind: reference(x, q, scale, K, N) for kind, x in inputs.items()}
        neg = act_quant.rel(refs["heavy"]["per-128"], refs["heavy"]["x"])
        neg_ok = neg > 1.5 * args.bound
        if not neg_ok:
            negctl_bad.append(sname)
        print(f"  negative control: heavy rows, per-128 reference vs bf16-activation reference {neg:.4f} "
              f"({'ok' if neg_ok else 'FAIL'}: must exceed {1.5 * args.bound:g})", flush=True)
        entry = {"shape": sname, "N": N, "K": K, "negative_control": neg, "arms": {}}
        outs_gauss = {}
        for arm in ARMS:
            name = arm[0]
            a = {}
            entry["arms"][name] = a
            try:
                with torch.inference_mode():
                    fn, resolved = build(arm, packed, scale, K, N)
                    a["resolved"] = resolved
                    errs = {}
                    for kind, x in inputs.items():
                        y = fn(x).float()
                        if kind == "gauss":
                            outs_gauss[name] = y
                        errs[kind] = {"vs_own_ref": act_quant.rel(y, refs[kind][EMUL[name]]),
                                      "vs_bf16_act_ref": act_quant.rel(y, refs[kind]["x"])}
                    a["err"] = errs
                    worst = max(e["vs_own_ref"] for e in errs.values())
                    a["ok"] = worst <= args.bound
                    a["time_us"] = {}
                    for M in Ms:
                        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                        t, how = time_call(fn, x)
                        a["time_us"][M] = t
                        a["timing"] = how
                    del fn
                if name == "humming-int8-g128":
                    ran_g128 = True
                if not a["ok"]:
                    (invalid if name == "marlin-w4a16" else fails if name in GATED else []).append(f"{sname}:{name}")
                print(f"  {name:20s} resolved={resolved or '-':22s} err vs own ref: gauss {errs['gauss']['vs_own_ref']:.4f} "
                      f"heavy {errs['heavy']['vs_own_ref']:.4f} (vs bf16-act ref: heavy {errs['heavy']['vs_bf16_act_ref']:.4f}) "
                      f"{'ok' if a['ok'] else 'OFF'}  [{a['timing']}] us: "
                      + " ".join(f"M{M}={a['time_us'][M]:.1f}" for M in Ms), flush=True)
            except Exception as e:
                a["error"] = f"{type(e).__name__}: {e}"
                print(f"  {name:20s} ERROR {a['error']}", flush=True)
                traceback.print_exc()
                if name == "marlin-w4a16":
                    invalid.append(f"{sname}:{name}")
                elif name in GATED:
                    fails.append(f"{sname}:{name}")
            torch.cuda.empty_cache()
        for x_, y_ in (("humming-w4a16", "marlin-w4a16"), ("humming-int8-g128", "marlin-w4a8-int8")):
            if x_ in outs_gauss and y_ in outs_gauss:
                d = act_quant.rel(outs_gauss[x_], outs_gauss[y_])
                entry.setdefault("diff_vs_marlin", {})[f"{x_}|{y_}"] = d
                print(f"  diff {x_} vs {y_} (gauss rows): {d:.4f}", flush=True)
        res["kernels"].append(entry)
        del q, scale, packed, refs, outs_gauss
        torch.cuda.empty_cache()

    # speed: per M, geometric mean over shapes of t(marlin arm) / t(humming arm); > 1 = Humming faster
    print("== speed (geomean over shapes of t_marlin / t_humming; >1 = Humming faster)", flush=True)
    speed = {}
    for hname in ("humming-w4a16", "humming-int8-g128", "humming-int8-token"):
        for mname in ("marlin-w4a16", "marlin-w4a8-int8"):
            row = {}
            for M in Ms:
                rs = [e["arms"][mname]["time_us"][M] / e["arms"][hname]["time_us"][M] for e in res["kernels"]
                      if "time_us" in e["arms"].get(hname, {}) and "time_us" in e["arms"].get(mname, {})]
                if rs:
                    row[M] = math.exp(sum(math.log(r) for r in rs) / len(rs))
            if row:
                speed[f"{hname}|{mname}"] = row
                print(f"  {hname:20s} vs {mname:18s} " + " ".join(f"M{M}={r:.2f}x" for M, r in row.items()), flush=True)
    res["speed"] = speed
    summary["int8_g128_runs"] = "yes" if ran_g128 else "no"
    summary["correct"] = "no" if fails else "yes"
    for key in ("humming-int8-g128|marlin-w4a16", "humming-int8-g128|marlin-w4a8-int8"):
        if key in speed:
            summary["speed_" + key.replace("|", "_vs_")] = ",".join(f"M{M}:{r:.2f}" for M, r in speed[key].items())

    if invalid:
        print(f"INVALID: reference arm off or broken: {invalid}", flush=True)
        finish(2, "INVALID-reference")
    if negctl_bad:
        print(f"NEGATIVE CONTROL FAILED on {negctl_bad}", flush=True)
        finish(3, "NEGCTL-FAILED")
    if not ran_g128 or fails:
        print(f"FAIL: {fails or 'humming-int8-g128 never ran'}", flush=True)
        finish(1, "BLOCKED-kernel")
    finish(0, "MEASURED")


if __name__ == "__main__":
    main()
