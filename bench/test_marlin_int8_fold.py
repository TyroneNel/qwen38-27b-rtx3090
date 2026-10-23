"""quant-2 fix A check: the lossless negative-scale remap (VLLM_MARLIN_INT8_LOSSLESS_FOLD=1,
docs/research/deepinfer/experiments/patches/quant2-lossless-fold.patch) against the default
sign fold (patches/marlin-int8-negative-scales.patch) on the Marlin W4A8-INT8 path.

  CUDA_VISIBLE_DEVICES=0 venv/bin/python bench/test_marlin_int8_fold.py

Builds a symmetric int4 g128 weight the way AutoRound signs it (scale = -extreme/8, so the
largest weight of every group lands on code -8 and half the scales are negative), in the
compressed-tensors pack-quantized layout, then checks:

  1. dequant (CPU): lossless_fold_remap's (code', |s|, zp) dequantizes bit-exactly to the
     signed (code - 8) * s; the fold's (16 - code clamped, -s) does not (it drops code -8).
  2. kernel (GPU): the knob-on layer loads as uint4 + zero points; its output is bitwise
     equal to a uint4 zero-point layer built directly from the remapped tensors through the
     checkpoint path (bench/test_marlin_int8_asym.py's); and its relative error against the
     float reference of the signed dequant is below the fold's at every shape.

Exit 0 all pass, 1 any failure (or the patch is missing), 2 step 1 passed but no CUDA device.
"""
import os, sys, torch
from vllm.model_executor.kernels.linear.mixed_precision import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision import marlin as marlin_mod
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.scalar_type import scalar_types
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

GROUP = 128
KNOB = "VLLM_MARLIN_INT8_LOSSLESS_FOLD"
SHAPES = [(5120, 17408), (17408, 5120), (5120, 12288), (5120, 6144), (5120, 5120), (5120, 1024)]

if not hasattr(marlin_mod, "lossless_fold_remap"):
    print("FAIL: quant2-lossless-fold.patch is not applied to this vllm", flush=True)
    sys.exit(1)


def make_weights(K, N, seed):
    """AutoRound-signed symmetric int4: signed codes q [N, K] in [-8, 7], bf16 scales [N, K/G]."""
    torch.manual_seed(seed)
    w = torch.randn(N, K, dtype=torch.float32) * 0.02
    g = w.reshape(N, K // GROUP, GROUP)
    ext = g.gather(-1, g.abs().argmax(-1, keepdim=True))        # signed extreme per group
    s = (-ext / 8).to(torch.bfloat16)                            # extreme -> code -8
    q = torch.clamp(torch.round(g / s.float()), -8, 7)
    deq = (q * s.float()).reshape(N, K)
    return q.reshape(N, K).to(torch.int8), s.reshape(N, K // GROUP), deq


def unpack_kn(packed_kn):
    """[K/8, N] int32 -> [K, N] int32 codes 0..15 (element i of a word at bits 4i)."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    nib = (packed_kn.unsqueeze(-1) >> shifts) & 0xF              # [K/8, N, 8]
    return nib.permute(0, 2, 1).reshape(-1, packed_kn.shape[1])


def dequant_check():
    ok = True
    for i, (K, N) in enumerate(SHAPES[3:]):          # the smaller shapes: CPU memory
        q, s, deq = make_weights(K, N, i)
        packed_kn = pack_to_int32(q, 4, packed_dim=1).t().contiguous()    # kernel layout [K/8, N]
        s_kn = s.t().contiguous()                                         # [K/G, N]
        code = unpack_kn(packed_kn)
        s_rows = s_kn.float().repeat_interleave(GROUP, dim=0)
        orig = (code - 8).float() * s_rows
        assert torch.equal(orig, deq.t()), "test unpack convention is wrong"
        q2, s2, zp = marlin_mod.lossless_fold_remap(packed_kn.clone(), s_kn.clone(), GROUP)
        remap = (unpack_kn(q2) - zp.repeat_interleave(GROUP, dim=0)).float() \
            * s2.float().repeat_interleave(GROUP, dim=0)
        neg = s_kn < 0
        neg_rows = neg.repeat_interleave(GROUP, dim=0)
        fold = torch.where(neg_rows, torch.clamp(16 - code, max=15), code)
        fold = (fold - 8).float() * torch.where(neg, -s_kn, s_kn).float().repeat_interleave(GROUP, dim=0)
        exact = torch.equal(remap, orig) and bool((s2 >= 0).all())
        e_fold = ((fold - orig).square().sum() / orig.square().sum()).item()
        e_remap = ((remap - orig).square().sum() / orig.square().sum()).item()
        n_neg8 = int(((code == 0) & neg_rows).sum())
        print(f"dequant K={K:6d} N={N:6d}  neg groups {neg.float().mean().item():.1%}  "
              f"neg-group code -8 weights {n_neg8}  remap exact {exact}  "
              f"error energy: fold {e_fold:.3e}  remap {e_remap:.3e}", flush=True)
        ok &= exact and e_fold > 0
    return ok


def kernel_check():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
    dev = "cuda"
    cfg_ctx = set_current_vllm_config(VllmConfig())
    cfg_ctx.__enter__()
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29518", local_rank=0)
    ensure_model_parallel_initialized(1, 1)
    wl = lambda *a, **k: None

    def build(K, N, q, s, zp_u, wtype, knob):
        os.environ[KNOB] = "1" if knob else "0"
        layer = torch.nn.Module()
        layer.weight_packed = PackedvLLMParameter(data=pack_to_int32(q, 4, packed_dim=1).contiguous().to(dev),
                                                  input_dim=1, output_dim=0, packed_dim=1, packed_factor=8,
                                                  weight_loader=wl)
        layer.weight_scale = GroupQuantScaleParameter(data=s.contiguous().to(dev), output_dim=0, input_dim=1,
                                                      weight_loader=wl)
        if zp_u is not None:
            layer.weight_zero_point = PackedvLLMParameter(
                data=pack_to_int32((zp_u - 8).to(torch.int8), 4, packed_dim=0).contiguous().to(dev),
                input_dim=1, output_dim=0, packed_dim=0, packed_factor=8, weight_loader=wl)
        cfg = MPLinearLayerConfig(full_weight_shape=(K, N), partition_weight_shape=(K, N),
                                  weight_type=wtype, act_type=torch.int8, group_size=GROUP,
                                  zero_points=zp_u is not None)
        ok, why = MarlinLinearKernel.can_implement(cfg)
        assert ok, why
        k = MarlinLinearKernel(cfg, w_q_param_name="weight_packed", w_s_param_name="weight_scale",
                               w_zp_param_name="weight_zero_point")
        k.process_weights_after_loading(layer)
        return layer, k

    ok = True
    worst_remap = 0.0
    for i, (K, N) in enumerate(SHAPES):
        q, s, deq = make_weights(K, N, i)
        neg = (s < 0).repeat_interleave(GROUP, dim=1)                       # [N, K]
        code = q.to(torch.int32) + 8
        code2 = torch.where(neg, 15 - code, code)                          # remap, [N, K] layout
        zp_u = 8 - (s < 0).to(torch.int32)                                 # [N, K/G]
        l_fold, k_fold = build(K, N, q, s, None, scalar_types.uint4b8, knob=False)
        l_remap, k_remap = build(K, N, q, s, None, scalar_types.uint4b8, knob=True)
        l_asym, k_asym = build(K, N, (code2 - 8).to(torch.int8), s.abs(), zp_u, scalar_types.uint4, knob=False)
        os.environ.pop(KNOB, None)
        loaded_zp = k_remap.config.weight_type == scalar_types.uint4 and k_remap.config.zero_points
        assert k_fold.config.weight_type == scalar_types.uint4b8
        deq = deq.to(dev)
        for M in (1, 5, 16, 256):
            torch.manual_seed(1000 + 10 * i + M)
            x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
            ref = x.float() @ deq.t()
            y_fold = k_fold.apply_weights(l_fold, x)
            y_remap = k_remap.apply_weights(l_remap, x)
            y_asym = k_asym.apply_weights(l_asym, x)
            bitwise = torch.equal(y_remap, y_asym)
            e_fold = ((y_fold.float() - ref).norm() / ref.norm()).item()
            e_remap = ((y_remap.float() - ref).norm() / ref.norm()).item()
            worst_remap = max(worst_remap, e_remap)
            good = loaded_zp and bitwise and e_remap < e_fold and e_remap < 0.05
            ok &= good
            print(f"kernel K={K:6d} N={N:6d} M={M:3d}  uint4+zp {loaded_zp}  remap==direct-asym {bitwise} "
                  f"(max |d| {(y_remap.float() - y_asym.float()).abs().max().item():.3g})  "
                  f"rel err fold {e_fold:.5f}  remap {e_remap:.5f}  {'ok' if good else 'FAIL'}", flush=True)
        del l_fold, l_remap, l_asym
    print("worst remap rel err:", worst_remap, flush=True)
    return ok


ok = dequant_check()
print("dequant:", "PASS" if ok else "FAIL", flush=True)
if not ok:
    sys.exit(1)
if not torch.cuda.is_available():
    print("SKIP: no CUDA device, kernel check not run", flush=True)
    sys.exit(2)
ok = kernel_check()
print("kernel:", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
