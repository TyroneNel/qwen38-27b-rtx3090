"""The pth-flash-attn op (pthfa/, PTH_FA=1) against an fp64 reference on the int8_per_token_head cache it reads.

Four parts, all on the cache views the CUSTOM backend passes to the op (PthFlashAttnImpl._caches over a packed
[blocks, ., ., 2 * (D + 4)] int8 cache, scales inline):
  1. shapes: head sizes 64/128/256, the target's GQA group at TP=1 (24 q / 4 kv) and TP=2 (12 / 2), decode,
     MTP verify (4 query tokens per request) and chunked prefill, both cache layouts, one and several KV splits,
     bf16 and fp16. Tolerance from the kernel's rounding-error analysis, per element:
     2^-8 * (p @ |v|) + u_out * |o| + 2^-24 (proven for L <= 4096).
  2. big pool (#86/#109): one request's blocks at the TOP of a 2,600-block pool of 816-token blocks (the hybrid
     group's block size), 4.4 GB of int8 cache, so block ids run past 2,383 and byte offsets past 2^32; every other
     block is poisoned, so a wrapped offset reads values ~1e5 off. L = 4,000 against the proven tolerance, and
     L = 20,000 / 131,072 against bench/test_spec_decode_bigpool.py's 0.05.
  3. CUDA-graph replay: a captured decode/verify launch, replayed after seq_lens grew on the device (as vLLM's
     FULL graphs replay), equals the eager launch bit for bit and the reference, with one and several splits.
  4. the DFlash2 drafter's layers: D = 128, 16 / 4 heads (TP=2) and 32 / 8 (TP=1), sliding window 2048, non-causal,
     8 query tokens per request (row r at position p sees keys [max(0, p - 2047), min(L - 1, p + 2047)]), block 864,
     L around the window edge and past it, both layouts, one and several splits, bf16 and fp16; once with the
     blocks below the window pointing to block 0 with NaN scales (vLLM's null block); and the window causal, and
     non-causal without a window, once each. The same tolerance (the visible keys are at most 2,055).

attn-1 knobs (docs/research/deepinfer/attention-kv.md, attn-1; read by pth_flash_attn.ops at import, default off):
PTHFA_TILE48=1 runs D = 256 row tiles of 33..48 rows (verify of 6..8 tokens at G = 6) on the swap-AB decode kernel
instead of the padded 64-row tile; it is lossless, so the tolerance above applies unchanged. The shapes include
verify lengths 6, 7 and 8 (36, 42 and 48 rows at G = 6) so that every arm exercises its kernel. Run once per arm.

PTHFA_MMA=pv16 (stage 2) accumulates the decode kernel's P''V of each 64-token tile in fp16 from zero (four
m16n8k16.f16 MMAs per output fragment) and adds the partial to the fp32 O. Re-derived bound, per element:
  (2^-8 + 2^-7) * (p @ |v|) + u_out * |o| + 2^-24 + 2^-20 * n_tiles * max(v_scale).
Derivation: a tile's partial is sum_t P''_t V_t with P''_t = fp16(p_t vs_t 2^-c) in [0, 1] and V_t integers, so
every intermediate is bounded by A = sum_t P''_t |V_t| <= 64 * 128 < 65504 (no overflow). Model (A100, arXiv
2512.07004; GA102 assumed the same family): each m16n8k16.f16 MMA is two 8-term block FMAs, each rounding its
fp16 result once (RNE, unit roundoff 2^-11; the in-adder truncation adds < 2^-19 relative). Four MMAs per tile,
so at most 8 roundings, each <= (2^-11 + 2^-19) * A: <= 2^-8 * 1.004 * A; the bound takes 2^-7 (2x margin, also
covering round-to-zero). Scaled to the output (x 2^c * prod(alpha) / l, with P'' 2^c V = p v and l >= 1) that is
2^-7 * (p @ |v|). Below 2^-14 the fp16 grid is absolute (2^-24, half that for RNE): <= 8 * 2^-25 per tile in P''V
units, x 2^c < 2 vmax_tile in the output, i.e. 2^-21 * max(v_scale) per tile, bound 2^-20 (n_tiles = ceil(L/64)).
The tile partial's one fp32 add replaces four fp32 MMA accumulations, so the fp32 term of the shipped bound still
covers it [inference: the bound.py analysis is not in this repo]. Applies to the decode-kernel launches (D = 256,
<= 32 rows, <= 48 with PTHFA_TILE48=1); the other launches meet the shipped bound, and every case also prints its
error against the shipped bound (err/tol_f32) for comparison.

PTHFA_MMA=s8pv16 (stage 3, lossy) is pv16 plus QK on int8: the kernel quantises each Q row (one token, one head,
D elements) to q8 = rn(q * fp32(127 / absmax)) and computes S exactly in int32. No bound against the true Q is
claimed. Each decode-kernel case is checked twice: (a) against the fp64 reference on the dequantised int8 Q
(q8 * fp32(absmax / 127), emulated bit for bit on the CPU) with the stage-2 bound, which catches a wrong fragment
layout or scale (those give errors of order |o|); and (b) error statistics against the reference on the true Q,
reported, not gated: rel-L2 of the output, the same for the reference on the int8 Q alone (the share Q's
quantisation explains), max |err| and the smallest per-row cosine. The summary line gives the worst of each.

Run inside the image or a venv with pthfa/install.sh done (production stopped; ~6 GB of GPU memory):
  venv/bin/python bench/test_pthfa_attn.py
  PTHFA_TILE48=1 venv/bin/python bench/test_pthfa_attn.py
  PTHFA_TILE48=1 PTHFA_MMA=pv16 venv/bin/python bench/test_pthfa_attn.py
  PTHFA_TILE48=1 PTHFA_MMA=s8pv16 venv/bin/python bench/test_pthfa_attn.py
Designed skips, exit 0: no CUDA device; pth_flash_attn not installed; a card other than sm_80/86/89 (the library
has no cubin for it, by design). Exit 0 = all OK, 1 = a case failed (named)."""
import math, sys
import torch
try:
    import pth_flash_attn.ops as ops
    from pth_flash_attn.planner import plan_launch, workspace_token_splits
    from pth_flash_attn.vllm_backend import PthFlashAttnImpl
except ImportError as e:
    print(f"pthfa: SKIP(by design): pth_flash_attn is not installed here ({e}); bash pthfa/install.sh"); sys.exit(0)
if not torch.cuda.is_available():
    print("pthfa: SKIP(by design): no CUDA device"); sys.exit(0)
cap = torch.cuda.get_device_capability()
if cap[0] != 8:
    print(f"pthfa: SKIP(by design): sm{cap[0]}{cap[1]}; the library has SASS for sm80/86/89 only"); sys.exit(0)
if not ops.is_available():
    print(f"pthfa: FAIL: the package is installed but its op library is not ({ops.LIBRARY_PATH})"); sys.exit(1)
DEV = "cuda"
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
OUT_ROUNDING = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11}
ATTN1 = ops.attn1_mode() if hasattr(ops, "attn1_mode") else 0
TILE48 = bool(ATTN1 & getattr(ops, "ATTN1_TILE48", 0))
MMA = ops.attn1_mma(ATTN1) if hasattr(ops, "attn1_mma") else "f32"
PV16 = MMA in ("pv16", "s8pv16")
S8 = MMA == "s8pv16"
PS8 = bool(ATTN1 & getattr(ops, "ATTN3_PREFILL_S8", 0))  # attn-3: int8 QK in the D = 256 64-row-tile kernel
ARM = (f"attn1 mode {ATTN1:#x} (PTHFA_TILE48={int(TILE48)}, PTHFA_MMA={MMA}, "
       f"PTHFA_PREFILL_QK={'s8' if PS8 else 'f16'})")
RESULTS = []
S8_STATS = []  # (rel_l2, rel_l2 of the int8-Q reference, max |err|, min cosine) per stage-3 case


def tolerance(ref, absv, dtype, vsmax, max_len, arm=True):
    """Per-element bound: the shipped one, plus the stage-2 terms when the arm accumulates PV in fp16."""
    tol = 2.0 ** -8 * absv + OUT_ROUNDING[dtype] * ref.abs() + 2.0 ** -24
    if arm and PV16:
        tol = tol + 2.0 ** -7 * absv + 2.0 ** -20 * math.ceil(max_len / 64) * vsmax
    return tol


def on_decode_kernel(d, g, wm, qmax):
    """True if the launch runs the swap-AB decode kernel, which the MMA modes change (launch.cu tile_tokens,
    instantiate_d256.cu)."""
    wm_for = lambda rows: 1 if rows <= 16 else (2 if rows <= 32 else 4)
    cap_t = 16 * wm // g
    bq = qmax if 1 <= qmax < cap_t and wm_for(qmax * g) == wm else cap_t
    rows = bq * g
    return d == 256 and (rows <= 32 or (TILE48 and rows <= 48))


def uses_s8(d, g, wm, qmax, window=0, causal=True):
    """True if the launch quantises Q to int8: attn-1 stage 3 on the decode kernel, or attn-3 on the D = 256
    64-row-tile kernel (instantiate_d256.cu)."""
    on_dec = on_decode_kernel(d, g, wm, qmax)
    return (S8 and on_dec) or (PS8 and d == 256 and not on_dec and not window and causal)


def q_int8(q):
    """The kernel's stage-3 Q (attn_decode.cuh stage_q_s8), emulated in fp32 on the CPU: q8 * fp32(absmax / 127)."""
    x = q.float().cpu()
    amax = x.abs().amax(-1, keepdim=True)
    inv = torch.where(amax > 0, torch.tensor(127.0) / amax, torch.zeros_like(amax))
    inv = torch.where(torch.isfinite(inv), inv, torch.zeros_like(inv))
    q8 = torch.round(x * inv).clamp(-127, 127)
    qs = torch.where(inv > 0, amax / torch.tensor(127.0), torch.zeros_like(amax))
    return (q8.double() * qs.double()).to(DEV)


def s8_stats(out, ref_true, ref_q8):
    """Stage-3 error statistics against the true-Q reference; recorded for the summary, not gated."""
    o = out.double()
    n = ref_true.norm().item()
    st = ((o - ref_true).norm().item() / n, (ref_q8 - ref_true).norm().item() / n, (o - ref_true).abs().max().item(),
          torch.nn.functional.cosine_similarity(o, ref_true, dim=-1).min().item())
    S8_STATS.append(st)
    return f"; vs true Q: rel-L2 {st[0]:.5f} (int8 Q alone {st[1]:.5f}), max|err| {st[2]:.4f}, min cos {st[3]:.6f}"


def report(ok, what):
    print(f"  {'OK  ' if ok else 'FAIL'}  {what}", flush=True)
    RESULTS.append(ok)


def impl_for(hq, hkv, d, window=0):
    return PthFlashAttnImpl(num_heads=hq, head_size=d, scale=d ** -0.5, num_kv_heads=hkv, alibi_slopes=None,
                            sliding_window=window or None, kv_cache_dtype="int8_per_token_head")


def make_cache(impl, nb, bs, hkv, d, layout, fill=True):
    shape = (nb, bs, hkv, 2 * (d + 4)) if layout == "NHD" else (nb, hkv, bs, 2 * (d + 4))
    kv = torch.empty(shape, dtype=torch.int8, device=DEV)
    if layout == "NHD":
        kv = kv.transpose(1, 2)
    k, v, ks, vs = impl._caches(kv)
    if fill:
        k.copy_(torch.randint_like(k, -127, 128)); v.copy_(torch.randint_like(v, -127, 128))
        ks.copy_(torch.rand_like(ks) * 0.5 / 127 + 0.5 / 127); vs.copy_(torch.rand_like(vs) * 0.5 / 127 + 0.5 / 127)
    return k, v, ks, vs


def reference(q, caches, bt, seq_lens, q_lens, scale, window=0, causal=True):
    """fp64 attention on the dequantized cache, causal or windowed / non-causal; returns (o, p @ |v|, max v_scale read).

    Row r at position p sees keys lo..hi: lo = max(0, p - W + 1) (0 without a window), hi = p if causal, else
    min(L - 1, p + W - 1) (L - 1 without a window). Only keys from the first row's lo on are read, so blocks
    wholly below the window (vLLM's null block) never enter it."""
    k, v, ks, vs = caches
    bs, hkv = k.shape[1], k.shape[2]
    g = q.shape[1] // hkv
    out = torch.empty(q.shape, dtype=torch.float64, device=DEV); absv = torch.empty_like(out)
    vsmax = 0.0
    start = 0
    for r, (L, ql) in enumerate(zip(seq_lens, q_lens)):
        pos = L - ql + torch.arange(ql, device=DEV)
        lo = (pos - window + 1).clamp(min=0) if window else torch.zeros_like(pos)
        hi = pos if causal else ((pos + window - 1).clamp(max=L - 1) if window else torch.full_like(pos, L - 1))
        slots = torch.arange(int(lo[0]), L, device=DEV); blocks = bt[r, slots // bs].long(); offs = slots % bs
        vis = (slots[None, :] >= lo[:, None]) & (slots[None, :] <= hi[:, None])
        for h in range(hkv):  # one KV head at a time keeps a 131k-token reference in a few hundred MB
            kd = k[blocks, offs, h].double() * ks[blocks, offs, h].double()[:, None]   # [L, D]
            vd = v[blocks, offs, h].double() * vs[blocks, offs, h].double()[:, None]
            vsmax = max(vsmax, torch.nan_to_num(vs[blocks, offs, h].double(), nan=0.0).max().item())
            qh = q[start:start + ql, h * g:(h + 1) * g].double()                        # [q, G, D]
            p = (torch.einsum("qgd,kd->gqk", qh, kd) * scale).masked_fill(~vis, -math.inf).softmax(-1)
            out[start:start + ql, h * g:(h + 1) * g] = torch.einsum("gqk,kd->qgd", p, vd)
            absv[start:start + ql, h * g:(h + 1) * g] = torch.einsum("gqk,kd->qgd", p, vd.abs())
        start += ql
    return out, absv, vsmax


def launch(q, caches, bt, seq_lens_t, qsl, q_lens, scale, hkv, d, ws_ts, out=None, window=0, causal=True):
    g = q.shape[1] // hkv; qmax = max(q_lens)
    plan = plan_launch(num_tokens=q.shape[0], num_reqs=len(q_lens), max_query_len=qmax, G=g, Hkv=hkv, D=d,
                       num_sms=NUM_SMS, is_prefill=qmax > 64 // g, ws_token_splits=ws_ts, window=window)
    ws_o = torch.empty(plan.ws_o_shape, dtype=torch.float32, device=DEV)
    ws_lse = torch.empty(plan.ws_lse_shape, dtype=torch.float32, device=DEV)
    out = torch.full_like(q, float("nan")) if out is None else out
    ops.pth_flash_attn_fwd(out, q, *caches, bt, seq_lens_t, qsl, qmax, plan.wm, plan.num_splits, scale, ws_o, ws_lse,
                           window, causal)
    return out, plan


def case(hq, hkv, d, seq_lens, q_lens, dtype, layout, bs, ws_ts, window=0, causal=True, null_block=False):
    impl = impl_for(hq, hkv, d, window)
    nbr = [math.ceil(L / bs) for L in seq_lens]
    nb = sum(nbr) + 1
    caches = make_cache(impl, nb, bs, hkv, d, layout)
    if null_block:  # block 0 is vLLM's null block: NaN scales, and every block wholly below the window maps to it
        caches[2][0] = float("nan"); caches[3][0] = float("nan")
        perm = torch.randperm(nb - 1, device=DEV, dtype=torch.int32) + 1
    else:
        perm = torch.randperm(nb, device=DEV, dtype=torch.int32)
    bt = torch.zeros(len(seq_lens), max(nbr), dtype=torch.int32, device=DEV); used = 0
    nulled = 0
    for r, n in enumerate(nbr):
        bt[r, :n] = perm[used:used + n]; used += n
        if null_block:
            below = max(0, seq_lens[r] - q_lens[r] - window + 1) // bs  # blocks wholly below the first row's lo
            bt[r, :below] = 0; nulled += below
    q = torch.randn(sum(q_lens), hq, d, dtype=dtype, device=DEV)
    qsl = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=DEV)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    out, plan = launch(q, caches, bt, sl, qsl, q_lens, impl.scale, hkv, d, ws_ts, window=window, causal=causal)
    torch.cuda.synchronize()
    s8 = uses_s8(d, hq // hkv, plan.wm, max(q_lens), window, causal)
    ref, absv, vsmax = reference(q_int8(q) if s8 else q, caches, bt, seq_lens, q_lens, impl.scale, window, causal)
    err = (out.double() - ref).abs()
    worst = (err / tolerance(ref, absv, dtype, vsmax, max(seq_lens))).max().item()
    worst_f32 = (err / tolerance(ref, absv, dtype, vsmax, max(seq_lens), arm=False)).max().item()
    ok = bool(torch.isfinite(out).all()) and worst <= 1.0 and (nulled > 0 or not null_block)
    mask = f" W={window} {'causal' if causal else 'non-causal'}" if window or not causal else ""
    mask += f" null-block x{nulled}" if null_block else ""
    stats = ""
    if s8:
        ref_t = reference(q, caches, bt, seq_lens, q_lens, impl.scale, window, causal)[0]
        stats = s8_stats(out, ref_t, ref)
    report(ok, f"D={d} Hq/Hkv={hq}/{hkv} q_lens={q_lens} L={seq_lens} {str(dtype)[6:]} {layout} bs={bs}{mask} "
               f"splits={plan.num_splits}: max err/tol {worst:.3f}{' vs the int8-Q reference' if s8 else ''} "
               f"(err/tol_f32 {worst_f32:.3f}){stats}")


print(f"== pthfa attention, {torch.cuda.get_device_name(0)} (sm{cap[0]}{cap[1]}, {NUM_SMS} SMs), {ARM}", flush=True)
torch.manual_seed(0)
print("== 1. shapes against the proven tolerance", flush=True)
SHAPES = {"decode": ([1, 17, 64, 1000, 4096], [1, 1, 1, 1, 1]),
          "verify": ([4, 333, 4096], [4, 4, 4]),
          "verify6": ([6, 900, 4096], [6, 6, 6]),       # 36 rows at G = 6: NT = 5 with PTHFA_TILE48=1
          "verify7": ([7, 2000, 4096], [7, 5, 7]),      # 42 rows (a 5-token request inside the tile): NT = 6
          "verify8": ([8, 333, 4096], [8, 8, 8]),       # 48 rows: the DFlash2 verify, NT = 6
          "prefill": ([512, 1500, 37], [512, 300, 37])}
for d in (64, 128, 256):
    for hq, hkv in ((24, 4), (12, 2)):
        for name, (sl, ql) in SHAPES.items():
            case(hq, hkv, d, sl, ql, torch.bfloat16, "NHD", 16, 1 << 20)
for layout in ("NHD", "HND"):
    for bs in (16, 816):
        for ws in (0, 1 << 20):  # ws 0 forces one split
            case(24, 4, 256, *SHAPES["verify"], torch.bfloat16, layout, bs, ws)
            case(24, 4, 256, *SHAPES["verify8"], torch.bfloat16, layout, bs, ws)
case(24, 4, 256, *SHAPES["verify"], torch.float16, "NHD", 816, 1 << 20)
case(24, 4, 256, *SHAPES["verify8"], torch.float16, "NHD", 816, 1 << 20)

print("== 2. big pool: block ids past 2,383, byte offsets past 2^32", flush=True)
NB, BS = 2600, 816
for hq, hkv in ((24, 4), (12, 2)):
    d = 256
    impl = impl_for(hq, hkv, d)
    caches = make_cache(impl, NB, BS, hkv, d, "NHD", fill=False)
    k, v, ks, vs = caches
    k.fill_(127); v.fill_(127); ks.fill_(1000.0); vs.fill_(1000.0)   # poison: a wrong block reads ~1.3e5
    ws_ts = workspace_token_splits(max_num_seqs=8, q_dec_max=4, Hkv=hkv, D=d, num_sms=NUM_SMS)
    for L, q_len, proven in ((4000, 4, True), (4000, 8, True), (20000, 4, False), (131072, 4, False),
                             (131072, 1, False), (131072, 8, False)):
        nbr = math.ceil(L / BS)
        ids = torch.arange(NB - nbr, NB, device=DEV, dtype=torch.int32)
        idl = ids.long()
        k[idl] = torch.randint_like(k[idl], -127, 128); v[idl] = torch.randint_like(v[idl], -127, 128)
        ks[idl] = torch.rand_like(ks[idl]) * 0.5 / 127 + 0.5 / 127; vs[idl] = torch.rand_like(vs[idl]) * 0.5 / 127 + 0.5 / 127
        bt = ids.reshape(1, nbr)
        q = torch.randn(q_len, hq, d, dtype=torch.bfloat16, device=DEV)
        qsl = torch.tensor([0, q_len], dtype=torch.int32, device=DEV); sl = torch.tensor([L], dtype=torch.int32, device=DEV)
        out, plan = launch(q, caches, bt, sl, qsl, [q_len], impl.scale, hkv, d, ws_ts)
        torch.cuda.synchronize()
        s8 = uses_s8(d, hq // hkv, plan.wm, q_len)
        ref, absv, vsmax = reference(q_int8(q) if s8 else q, caches, bt, [L], [q_len], impl.scale)
        err = (out.double() - ref).abs()
        worst = (err / tolerance(ref, absv, torch.bfloat16, vsmax, L)).max().item()
        if proven:
            ok = worst <= 1.0; how = f"max err/tol {worst:.3f}"
        else:
            e = err.max().item(); ok = e < 0.05
            how = f"max|ours-ref| {e:.4f} (< 0.05; err/tol {worst:.3f}, not proven at this L)"
        if s8:
            how += " vs the int8-Q reference" + s8_stats(out, reference(q, caches, bt, [L], [q_len], impl.scale)[0], ref)
        ok = ok and bool(torch.isfinite(out).all())
        byte_end = (int(ids[-1]) + 1) * k.stride(0)
        report(ok, f"Hq/Hkv={hq}/{hkv} L={L} q={q_len} blocks {NB - nbr}..{NB - 1} (last byte {byte_end:.3e} > 2^32 "
                   f"= {byte_end > 2 ** 32}) splits={plan.num_splits}: {how}")
        k[idl] = 127; v[idl] = 127; ks[idl] = 1000.0; vs[idl] = 1000.0
    del caches, k, v, ks, vs
    torch.cuda.empty_cache()

print("== 3. CUDA-graph replay after seq_lens grow on the device", flush=True)
for ws, qn in ((0, 4), (1 << 20, 4), (0, 8), (1 << 20, 8)):
    hq, hkv, d, bs = 24, 4, 256, 816
    impl = impl_for(hq, hkv, d)
    nbr = math.ceil(4096 / bs)
    caches = make_cache(impl, 2 * nbr, bs, hkv, d, "NHD")
    bt = torch.arange(2 * nbr, dtype=torch.int32, device=DEV).view(2, nbr)
    q = torch.randn(2 * qn, hq, d, dtype=torch.bfloat16, device=DEV)
    qsl = torch.tensor([0, qn, 2 * qn], dtype=torch.int32, device=DEV)
    sl = torch.tensor([100, 200], dtype=torch.int32, device=DEV)
    out_g = torch.empty_like(q)
    _, plan = launch(q, caches, bt, sl, qsl, [qn, qn], impl.scale, hkv, d, ws, out=out_g)   # warm-up, fixes the plan
    ws_o = torch.empty(plan.ws_o_shape, dtype=torch.float32, device=DEV)
    ws_lse = torch.empty(plan.ws_lse_shape, dtype=torch.float32, device=DEV)
    run = lambda o: ops.pth_flash_attn_fwd(o, q, *caches, bt, sl, qsl, qn, plan.wm, plan.num_splits, impl.scale,
                                           ws_o, ws_lse)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(out_g)
    sl.copy_(torch.tensor([1500, 4096], dtype=torch.int32))
    g.replay()
    out_e = torch.empty_like(q); run(out_e); torch.cuda.synchronize()
    same = torch.equal(out_g, out_e)
    s8 = uses_s8(d, hq // hkv, plan.wm, qn)
    ref, absv, vsmax = reference(q_int8(q) if s8 else q, caches, bt, [1500, 4096], [qn, qn], impl.scale)
    worst = ((out_g.double() - ref).abs() / tolerance(ref, absv, torch.bfloat16, vsmax, 4096)).max().item()
    report(same and worst <= 1.0, f"q={qn} splits={plan.num_splits}: L 100/200 -> 1500/4096 on the device; replay == "
                                  f"eager bit for bit: {same}; vs {'int8-Q ' if s8 else ''}reference max err/tol "
                                  f"{worst:.3f}")

print("== 4. the DFlash2 drafter's layers: D=128, sliding window 2048, non-causal, 8 query tokens", flush=True)
W, DRAFT_Q, DRAFT_BS = 2048, 8, 864
DRAFT_L = [8, 2047, 2048, 2049, 2055, 5000]  # one block, the window edge, past it
for hq, hkv in ((16, 4), (32, 8)):
    for layout in ("NHD", "HND"):
        for ws in (0, 1 << 20):  # ws 0 forces one split
            case(hq, hkv, 128, DRAFT_L, [DRAFT_Q] * len(DRAFT_L), torch.bfloat16, layout, DRAFT_BS, ws,
                 window=W, causal=False)
    case(hq, hkv, 128, DRAFT_L, [DRAFT_Q] * len(DRAFT_L), torch.float16, "NHD", DRAFT_BS, 1 << 20,
         window=W, causal=False)
    case(hq, hkv, 128, [5000, 9000], [DRAFT_Q] * 2, torch.bfloat16, "NHD", DRAFT_BS, 1 << 20,
         window=W, causal=False, null_block=True)
case(16, 4, 128, DRAFT_L, [DRAFT_Q] * len(DRAFT_L), torch.bfloat16, "NHD", DRAFT_BS, 1 << 20, window=W, causal=True)
case(16, 4, 128, [8, 3000], [DRAFT_Q] * 2, torch.bfloat16, "NHD", DRAFT_BS, 1 << 20, window=0, causal=False)
if PS8:
    print("== 5. attn-3: int8-QK prefill chunks at depth (D=256, the serving geometry)", flush=True)
    for sl, ql in (([8192], [1024]), ([3000, 9000], [2048, 700])):
        for dtype in (torch.bfloat16, torch.float16):
            case(24, 4, 256, sl, ql, dtype, "NHD", 816, 1 << 20)
ok = all(RESULTS)
if S8_STATS:
    w = [max(s[0] for s in S8_STATS), max(s[1] for s in S8_STATS), max(s[2] for s in S8_STATS),
         min(s[3] for s in S8_STATS)]
    print(f"== stage 3 (s8pv16) vs the true Q, {len(S8_STATS)} decode-kernel cases (reported, not gated): worst rel-L2 "
          f"{w[0]:.5f} (int8 Q alone {w[1]:.5f}), worst max|err| {w[2]:.4f}, min cosine {w[3]:.6f}", flush=True)
print(f"PTHFA ATTN {'OK' if ok else 'FAIL'} ({sum(RESULTS)}/{len(RESULTS)}) [{ARM}]"); sys.exit(0 if ok else 1)
