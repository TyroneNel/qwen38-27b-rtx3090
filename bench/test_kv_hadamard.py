"""attn-5 GPU test (docs/research/deepinfer/attention-kv.md attn-5; experiments/patches/attn5-kv-hadamard.patch):
the patch's rotation helpers around the real int8 write and read paths of mode D.

One D-arm layer geometry (Hq/Hkv/D 24/4/256, int8_per_token_head, block 816): K/V are written with vLLM's own
per-token-head writer (PthFlashAttnImpl.do_kv_cache_update, inherited from TritonAttentionImpl, exactly as
unified_kv_cache_update calls it) and read with the pthfa op (the CUSTOM backend's call), once unrotated and once
the way the patched ops do it (K H, V H written; q H read; output times H; the patch's _kv_hadamard_matrix and
_kv_rotate, imported from the image's vllm). Reference: fp64 attention on the unquantised bf16 q/k/v.

Data, two regimes: 'outlier' (K with per-channel spread, 6 channels x15 and a per-channel offset, attn_kvq_error.py's
regime B shape, where the Hadamard must help) and 'flat' (k_norm-like K: unit-RMS rows, mild spread).
Checks:
  1. the cache really holds rotated values (rotated arm's int8 K differs from the unrotated arm's);
  2. negative control: the rotated cache read with an UNrotated q is far off (>= 10x the rotated arm's error),
     so a frame mismatch anywhere in the patch cannot pass the server-level probes silently;
  3. 'outlier': rotated output rel-L2 < unrotated; 'flat': rotated <= 1.25 x unrotated (no regression).
Prints the rel-L2 of each arm at 4k / 32k / 128k. Exit 0 = all checks pass, 1 = a check failed,
0 with SKIP(by design) without CUDA, pthfa or the patch.
"""
import math
import sys

import torch

if not torch.cuda.is_available():
    print("kv hadamard: SKIP(by design): no CUDA device"); sys.exit(0)
try:
    import pth_flash_attn.ops as ops
    from pth_flash_attn.planner import plan_launch
    from pth_flash_attn.vllm_backend import PthFlashAttnImpl
    from vllm.model_executor.layers.attention.attention import _kv_hadamard_matrix, _kv_rotate
except ImportError as e:
    print(f"kv hadamard: SKIP(by design): {e}"); sys.exit(0)

DEV = "cuda"
HQ, HKV, D, BS = 24, 4, 256, 816
G = HQ // HKV
NSM = torch.cuda.get_device_properties(0).multi_processor_count
H = _kv_hadamard_matrix(D, torch.bfloat16, DEV)
RESULTS = []


def report(ok, what):
    print(f"  {'OK  ' if ok else 'FAIL'}  {what}", flush=True)
    RESULTS.append(ok)


def data(L, regime, seed, qlen=4):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if regime == "outlier":
        ch = torch.exp(torch.randn(HKV, D, generator=g) * 0.6)
        idx = torch.randint(0, D, (HKV, 6), generator=g)
        ch.scatter_(1, idx, ch.gather(1, idx) * 15.0)
        mean = torch.randn(HKV, D, generator=g) * ch * 0.5
        k = torch.randn(L, HKV, D, generator=g) * ch + mean
    else:
        ch = torch.exp(torch.randn(HKV, D, generator=g) * 0.2)
        k = torch.randn(L, HKV, D, generator=g) * ch
        k = k / k.pow(2).mean(-1, keepdim=True).sqrt()
    v = torch.randn(L, HKV, D, generator=g) * torch.exp(torch.randn(HKV, D, generator=g) * 0.3)
    tgt = torch.randint(0, L - qlen, (qlen, HQ), generator=g)
    q = k[tgt, torch.arange(HQ)[None, :] // G] * 2.0 + torch.randn(qlen, HQ, D, generator=g)
    return q.bfloat16().to(DEV), k.bfloat16().to(DEV), v.bfloat16().to(DEV)


def reference(q, k, v):
    L, qlen = k.shape[0], q.shape[0]
    out = torch.empty(q.shape, dtype=torch.float64, device=DEV)
    pos = L - qlen + torch.arange(qlen, device=DEV)
    vis = torch.arange(L, device=DEV)[None, :] <= pos[:, None]
    for h in range(HKV):
        qh = q[:, h * G:(h + 1) * G].double()
        p = (torch.einsum("qgd,kd->gqk", qh, k[:, h].double()) * D ** -0.5).masked_fill(~vis, -math.inf).softmax(-1)
        out[:, h * G:(h + 1) * G] = torch.einsum("gqk,kd->qgd", p, v[:, h].double())
    return out


def write_and_read(impl, kv, k, v, q):
    """vLLM's writer into the layer cache, then the pthfa op (decode-shaped launch: one request, q_len tokens)."""
    L, qlen = k.shape[0], q.shape[0]
    slot = torch.arange(L, device=DEV, dtype=torch.int64)
    impl.do_kv_cache_update(None, k, v, kv, slot)
    kc, vc, ks, vs = impl._caches(kv)
    nb = math.ceil(L / BS)
    bt = torch.arange(nb, device=DEV, dtype=torch.int32).view(1, nb)
    plan = plan_launch(num_tokens=qlen, num_reqs=1, max_query_len=qlen, G=G, Hkv=HKV, D=D, num_sms=NSM,
                       is_prefill=False, ws_token_splits=1 << 20)
    ws_o = torch.empty(plan.ws_o_shape, dtype=torch.float32, device=DEV)
    ws_lse = torch.empty(plan.ws_lse_shape, dtype=torch.float32, device=DEV)
    out = torch.empty_like(q)
    ops.pth_flash_attn_fwd(out, q, kc, vc, ks, vs, bt, torch.tensor([L], dtype=torch.int32, device=DEV),
                           torch.tensor([0, qlen], dtype=torch.int32, device=DEV), qlen, plan.wm, plan.num_splits,
                           impl.scale, ws_o, ws_lse, 0, True)
    return out, kc.clone()


def rel(o, ref):
    return (o.double() - ref).norm().item() / ref.norm().item()


print(f"== attn-5 kv hadamard, {torch.cuda.get_device_name(0)}", flush=True)
impl = PthFlashAttnImpl(num_heads=HQ, head_size=D, scale=D ** -0.5, num_kv_heads=HKV, alibi_slopes=None,
                        sliding_window=None, kv_cache_dtype="int8_per_token_head")
# ONE layer cache for every case: TritonAttentionImpl binds its fp32 scale views to the first kv_cache it sees
# (_ensure_scale_caches returns early once _k_scale_cache is set), as in serving, where a layer's cache never moves.
# A fresh cache per case left the scale views on the freed first one (the cudaErrorIllegalAddress of run 1).
LMAX = 131072
kv = torch.zeros((math.ceil(LMAX / BS), BS, HKV, 2 * (D + 4)), dtype=torch.int8, device=DEV).transpose(1, 2)
for regime in ("outlier", "flat"):
    for L in (4096, 32768, LMAX):
        q, k, v = data(L, regime, L)
        ref = reference(q, k, v)
        o0, k0 = write_and_read(impl, kv, k, v, q)
        orot, k1 = write_and_read(impl, kv, _kv_rotate(k, H), _kv_rotate(v, H), _kv_rotate(q, H))
        o1 = torch.mm(orot.view(-1, D), H).view_as(orot)
        oneg, _ = write_and_read(impl, kv, _kv_rotate(k, H), _kv_rotate(v, H), q)
        oneg = torch.mm(oneg.view(-1, D), H).view_as(oneg)
        e0, e1, en = rel(o0, ref), rel(o1, ref), rel(oneg, ref)
        print(f"ROW {regime:7s} L={L:>6d}: rel-L2 unrotated {e0:.5f}  rotated {e1:.5f}  (x{e0 / e1:.2f})  "
              f"negative control (q unrotated) {en:.3f}", flush=True)
        report(not torch.equal(k0, k1), f"{regime} L={L}: the cache holds rotated K")
        report(en >= 10 * e1, f"{regime} L={L}: frame mismatch is detected ({en:.3f} >= 10 x {e1:.5f})")
        if regime == "outlier":
            report(e1 < e0, f"{regime} L={L}: rotation helps ({e1:.5f} < {e0:.5f})")
        else:
            report(e1 <= 1.25 * e0, f"{regime} L={L}: no regression ({e1:.5f} <= 1.25 x {e0:.5f})")
        del ref
        torch.cuda.empty_cache()
ok = all(RESULTS)
print(f"KV HADAMARD {'OK' if ok else 'FAIL'} ({sum(RESULTS)}/{len(RESULTS)})")
sys.exit(0 if ok else 1)
