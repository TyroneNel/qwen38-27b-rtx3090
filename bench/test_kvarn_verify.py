"""KVarN verify attention (attn-2, KVARN_VERIFY_SHARED) against an fp64 reference on the dequantized cache.

docs/research/deepinfer/attention-kv.md, attn-2. Needs the image's vLLM with
docs/research/deepinfer/experiments/patches/attn2-kvarn-verify-shared.patch applied (it adds
kvarn_verify_shared_launch and the KVARN_VERIFY_SHARED route to vllm/v1/attention/ops/triton_kvarn_decode.py).

The cache is synthetic but in KVarN's exact tile layout (k4v2, group 128, head_dim 256, 26,880 B per block and
head): random 4-bit K / 2-bit V codes with per-channel and per-token fp16 scales and zero points, a permuted block
table one column wider than a multiple of 16 (as the runner's), the sink block, a mid block and the open tail block
in the fp16 tail pool through Block_to_slot, and poisoned spare blocks (NaN scales, block 0 included) that a
correct kernel never reads. The reference dequantizes in fp64 and runs a two-pass fp64 softmax in key chunks, so
any context length fits.

Parts:
  1. kernel (rotated frame): kvarn_verify_shared_launch on q_rot = fp16(q @ H), the exact input the driver feeds
     it, against the fp64 reference on the same q_rot. Per-element bound, from the kernel's roundings (K, Q-free,
     V and P rounded to fp16 once each; the dequant inputs bounded by their absolute terms so the KVARN_FP16_DEQUANT
     cancellation in code*s + zp is covered):
       e_s  = 3 * 2^-11 * scale * max_t sum_d |q_d| kabs_td           (score error, per row)
       tol  = 2 * (2 * e_s * (p@vabs + |o|) + 2^-9 * p@vabs + 2^-11 * |o|) + 2^-20
              (+ 2^-7 * p@vabs with KVARN_VERIFY_SHARED_PV16: chunk partials of <= 64 tokens accumulated in fp16)
     The per-token base path (tf32 MMA on the same fp16-rounded inputs) is held to the same bound when it runs.
     Cases: QLEN 2, 3, 4, 7, 8, 16 (M_PAD 16..128); contexts from sink-only to 131,072; peaky queries; a batch of
     three requests of different lengths plus a padded request (its rows must come out exactly 0); a sliding
     window; every launch config (BLOCK_N 16/32/64 x 4/8 warps x 1/2 stages); split counts 1..256 agree; PV16 and
     F16 on and off.
  2. driver: kvarn_verify_attention with the route on (module flag) against the route off and the unrotated fp64
     reference: rel-L2 on <= 1.25 x rel-L2 off + 1e-4, and <= 5e-3.

Run inside the patched image (production stopped; < 4 GB of GPU memory):
  /app/venv/bin/python bench/test_kvarn_verify.py
  KVARN_FP16_DEQUANT=1 /app/venv/bin/python bench/test_kvarn_verify.py
  --quick skips the 131,072-token cases and the config sweep; --dry-run builds one tiny case on the CPU, checks the
  reference and the patch's symbols, and exits; --interpret runs a few tiny cases through TRITON_INTERPRET on the CPU
  (slow; no GPU needed). Exit 0 = all OK, 1 = a case failed (named), 2 = the patch is not installed.
"""
import argparse
import importlib.util
import math
import os
import sys
import types

import torch

D, G, HQ, HK = 256, 128, 24, 4
RESULTS = []


def report(ok, what):
    print(f"  {'OK  ' if ok else 'FAIL'}  {what}", flush=True)
    RESULTS.append((ok, what))


def load_kernels(module_path="", interpret=False):
    """The patched triton_kvarn_decode module (installed, or a file), and KVarNConfig."""
    if interpret:
        os.environ["TRITON_INTERPRET"] = "1"
    if interpret or not torch.cuda.is_available():
        # Without a GPU vllm.triton_utils hands out placeholder decorators; the kernels need the real ones.
        import triton
        import triton.language as tl
        import vllm  # noqa: F401
        tu = types.ModuleType("vllm.triton_utils")
        tu.triton, tu.tl, tu.HAS_TRITON = triton, tl, True
        sys.modules["vllm.triton_utils"] = tu
    if module_path:
        spec = importlib.util.spec_from_file_location("triton_kvarn_decode_under_test", module_path)
        tkd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tkd)
    else:
        from vllm.v1.attention.ops import triton_kvarn_decode as tkd
    from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
    return tkd, KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", D)


def hadamard(n, device):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(device)


class Case:
    """A synthetic KVarN cache, block table, tail pool and verify plan for B requests of qlen tokens each.

    ctxs[b] is request b's committed context (its full length is ctxs[b] + qlen); ctxs[b] < 0 marks a padded
    request (seq_len 0, as uniform-batch graph replay pads)."""

    def __init__(self, cfg, ctxs, qlen, device, seed=0, qscale=1.0, window=0, hq=HQ, hk=HK):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        rnd = lambda *s: torch.rand(*s, generator=gen)
        self.cfg, self.qlen, self.device, self.window, self.hq, self.hk = cfg, qlen, device, window, hq, hk
        self.B = len(ctxs)
        self.seq = [c + qlen if c >= 0 else 0 for c in ctxs]
        nblk = [-(-s // G) for s in self.seq]
        n_live = sum(nblk)
        total = n_live + 4                                  # block 0 + 3 spares, poisoned
        tb = cfg.tile_bytes_aligned
        kv = torch.randint(0, 256, (total, hk, tb), dtype=torch.uint8, generator=gen)

        def put16(off, vals):                                 # vals [total, hk, n] -> fp16 bytes at off
            n = vals.shape[-1]
            kv[:, :, off:off + 2 * n] = vals.to(torch.float16).contiguous().view(torch.uint8)

        s_col_k = 0.05 + 0.10 * rnd(total, hk, D)
        put16(cfg.k_s_col_offset, s_col_k)
        put16(cfg.k_zp_offset, -7.5 * s_col_k + 0.05 * torch.randn(total, hk, D, generator=gen))
        put16(cfg.k_s_row_offset, 0.5 + rnd(total, hk, G))
        put16(cfg.v_s_col_offset, 0.5 + rnd(total, hk, D))
        s_row_v = 0.2 + 0.4 * rnd(total, hk, G)
        put16(cfg.v_s_row_offset, s_row_v)
        put16(cfg.v_zp_offset, -1.5 * s_row_v + 0.05 * torch.randn(total, hk, G, generator=gen))
        poison = [0] + list(range(n_live + 1, total))
        nan = torch.full((len(poison), hk, D), float("nan"))
        kv[poison, :, cfg.k_s_col_offset:cfg.k_s_col_offset + 2 * D] = nan.half().view(torch.uint8)
        kv[poison, :, cfg.v_s_col_offset:cfg.v_s_col_offset + 2 * D] = nan.half().view(torch.uint8)

        width = max(max(nblk), 1) + 1
        width += (17 - width % 16) % 16                      # == 1 (mod 16): not 16-divisible, like the runner's
        bt = torch.zeros(self.B, width, dtype=torch.int32)
        ids = (torch.randperm(n_live, generator=gen) + 1).tolist()
        b2s = torch.full((total,), -1, dtype=torch.int32)
        slot = 0
        for b in range(self.B):
            for k in range(nblk[b]):
                bt[b, k] = ids.pop()
            if nblk[b]:
                pooled = {0, nblk[b] - 1}                     # sink and open tail
                if nblk[b] > 3:
                    pooled.add(nblk[b] // 2)                  # an unflushed full block
                for k in sorted(pooled):
                    b2s[bt[b, k]] = slot
                    slot += 1
        pool_k = (0.6 * torch.randn(max(slot, 1), G, hk, D, generator=gen)).half()
        pool_v = (0.6 * torch.randn(max(slot, 1), G, hk, D, generator=gen)).half()

        nq = self.B * qlen
        vq_seqlen = torch.zeros(nq, dtype=torch.int32)
        vq_req = torch.zeros(nq, dtype=torch.int32)
        for b in range(self.B):
            for j in range(qlen):
                vq_req[b * qlen + j] = b
                vq_seqlen[b * qlen + j] = self.seq[b] - qlen + j + 1 if self.seq[b] else 0
        q = (qscale * torch.randn(nq, hq, D, generator=gen)).half()

        to = lambda t: t.to(device)
        self.kv, self.bt, self.b2s, self.pool_k, self.pool_v = map(to, (kv, bt, b2s, pool_k, pool_v))
        self.vq_seqlen, self.vq_req, self.q = map(to, (vq_seqlen, vq_req, q))
        self.nblk, self.total = nblk, total
        self.max_ctx_blocks = max(max(nblk), 1)
        self.scale = D ** -0.5
        self.H64 = hadamard(D, device)
        self.H16 = self.H64.half()
        self.q_rot = torch.mm(self.q.reshape(-1, D), self.H16).view(nq, hq, D)    # what the driver feeds stage 1
        self.impl = types.SimpleNamespace(
            _H_fp16=self.H16, _hadamard=lambda dev: self.H64.float(), _block_lookup_size=total,
            _block_to_slot_t=self.b2s, _tail_K_pool=self.pool_k, _tail_V_pool=self.pool_v,
            sliding_window=window, _sm_count=0)

    # ── fp64 reference ──────────────────────────────────────────────────────────────────────────────────────
    def _dequant(self, bids, hk):
        """K, V, kabs, vabs [n, G, D] fp64 for int4 blocks bids of kv head hk."""
        cfg, t = self.cfg, self.kv[bids, hk].to(torch.int32)                         # [n, tb]
        f16 = lambda off, n: self.kv[bids, hk, off:off + 2 * n].contiguous().view(torch.float16).double()
        s_col_k, zp_k = f16(cfg.k_s_col_offset, D), f16(cfg.k_zp_offset, D)
        s_row_k = f16(cfg.k_s_row_offset, G)
        kb = t[:, cfg.k_packed_offset:cfg.k_packed_offset + D * G // 2].view(-1, D, G // 2)
        ck = torch.stack([kb & 15, kb >> 4], -1).view(-1, D, G).double()              # [n, D, G]
        a = ck * s_col_k[:, :, None]
        K = ((a + zp_k[:, :, None]) * s_row_k[:, None, :]).transpose(1, 2)
        kabs = ((a.abs() + zp_k.abs()[:, :, None]) * s_row_k.abs()[:, None, :]).transpose(1, 2)
        s_col_v, s_row_v, zp_v = f16(cfg.v_s_col_offset, D), f16(cfg.v_s_row_offset, G), f16(cfg.v_zp_offset, G)
        vb = t[:, cfg.v_packed_offset:cfg.v_packed_offset + G * D // 4].view(-1, G, D // 4)
        cv = torch.stack([(vb >> (2 * i)) & 3 for i in range(4)], -1).view(-1, G, D).double()
        a = cv * s_row_v[:, :, None]
        V = (a + zp_v[:, :, None]) * s_col_v[:, None, :]
        vabs = (a.abs() + zp_v.abs()[:, :, None]) * s_col_v.abs()[:, None, :]
        return K.contiguous(), V, kabs.contiguous(), vabs

    def _keys(self, b, hk, k0, k1):
        """Blocks k0..k1-1 of request b, kv head hk: K, V, kabs, vabs [(k1-k0)*G, D] fp64 (pool or int4)."""
        bids = self.bt[b, k0:k1].long()
        slots = self.b2s[bids].long()
        K, V, KA, VA = self._dequant(bids, hk)
        pooled = slots >= 0
        if pooled.any():
            pk = self.pool_k[slots[pooled], :, hk].double()
            pv = self.pool_v[slots[pooled], :, hk].double()
            K[pooled], V[pooled], KA[pooled], VA[pooled] = pk, pv, pk.abs(), pv.abs()
        return [x.reshape(-1, D) for x in (K, V, KA, VA)]

    def reference(self, chunk_blocks=32):
        """o_rot [NQ, Hq, D] fp64 on q_rot, the bound's ingredients (p@vabs, e_s), and o (unrotated)."""
        qlen, qpk = self.qlen, self.hq // self.hk
        nq = self.B * qlen
        o = torch.zeros(nq, self.hq, D, dtype=torch.float64, device=self.device)
        pva = torch.zeros_like(o)
        e_s = torch.zeros(nq, self.hq, 1, dtype=torch.float64, device=self.device)
        for b in range(self.B):
            L = self.seq[b]
            if L == 0:
                continue
            rows = slice(b * qlen, (b + 1) * qlen)
            limit = torch.arange(qlen, device=self.device) + (L - qlen + 1)          # [qlen]
            for hk in range(self.hk):
                hs = slice(hk * qpk, (hk + 1) * qpk)
                qr = self.q_rot[rows, hs].double().reshape(qlen * qpk, D)           # row = j*qpk + lane
                lim = limit.repeat_interleave(qpk)[:, None]
                lo = (lim - self.window).clamp(min=0) if self.window else None
                n = self.nblk[b]
                m = torch.full((qlen * qpk, 1), -math.inf, dtype=torch.float64, device=self.device)
                sabs = torch.zeros_like(m)
                for pass_ in (0, 1):
                    if pass_:
                        num = torch.zeros(qlen * qpk, D, dtype=torch.float64, device=self.device)
                        numa, den = torch.zeros_like(num), torch.zeros_like(m)
                    for k0 in range(0, n, chunk_blocks):
                        k1 = min(n, k0 + chunk_blocks)
                        K, V, KA, VA = self._keys(b, hk, k0, k1)
                        pos = torch.arange(k0 * G, k1 * G, device=self.device)[None, :]
                        mask = pos < lim
                        if lo is not None:
                            mask &= pos >= lo
                        s = (qr @ K.T) * self.scale
                        s = torch.where(mask, s, torch.full_like(s, -math.inf))
                        if not pass_:
                            m = torch.maximum(m, s.amax(1, keepdim=True))
                            sa = torch.where(mask, (qr.abs() @ KA.T) * self.scale, torch.zeros_like(s))
                            sabs = torch.maximum(sabs, sa.amax(1, keepdim=True))
                        else:
                            p = torch.exp(s - m)
                            num += p @ V
                            numa += p @ VA
                            den += p.sum(1, keepdim=True)
                o[rows, hs] = (num / den).view(qlen, qpk, D)
                pva[rows, hs] = (numa / den).view(qlen, qpk, D)
                e_s[rows, hs] = (3 * 2.0 ** -11 * sabs).view(qlen, qpk, 1)
        return o, pva, e_s

    def bound(self, o, pva, e_s, pv16=False):
        tol = 2 * (2 * e_s * (pva + o.abs()) + 2.0 ** -9 * pva + 2.0 ** -11 * o.abs()) + 2.0 ** -20
        if pv16:
            tol = tol + 2.0 ** -7 * pva
        return tol


def check(case, out, ref, what, pv16=False):
    o, pva, e_s = ref
    out = out.double()
    tol = case.bound(o, pva, e_s, pv16)
    err = (out - o).abs()
    finite = bool(torch.isfinite(out).all())
    worst = (err / tol).max().item() if finite else math.inf
    live = [b for b in range(case.B) if case.seq[b]]
    pad_ok = True
    for b in range(case.B):
        if not case.seq[b]:
            pad_ok &= bool((out[b * case.qlen:(b + 1) * case.qlen] == 0).all())
    rel = ((out - o).norm() / o.norm().clamp_min(1e-30)).item()
    ok = finite and worst <= 1.0 and pad_ok
    report(ok, f"{what}: rel-L2 {rel:.2e}, max err/tol {worst:.3f}"
               f"{'' if pad_ok else ', PADDED ROWS NOT ZERO'}{'' if finite else ', NON-FINITE'}"
               f" ({len(live)} live req)")
    return rel


def run_shared(tkd, case, **kw):
    out = torch.empty_like(case.q_rot)
    tkd.kvarn_verify_shared_launch(case.q_rot, out, case.kv, case.bt, case.vq_seqlen, case.scale, case.cfg,
                                   case.impl, case.qlen, case.max_ctx_blocks, **kw)
    return out


def run_base_rot(tkd, case):
    """The stock per-token path (VQ_INDIRECT split-K), rotated frame: undo the driver's output rotation."""
    out = tkd.kvarn_verify_attention(case.q, case.kv, case.bt, case.scale, case.cfg, case.impl,
                                     case.vq_req, case.vq_seqlen, case.max_ctx_blocks)
    return (out.double().reshape(-1, D) @ case.H64).view_as(out)   # H is its own inverse


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--module", default="", help="load triton_kvarn_decode from this file instead of vllm")
    a = ap.parse_args()

    dev = "cpu" if (a.dry_run or a.interpret) else "cuda"
    if dev == "cuda" and not torch.cuda.is_available():
        print("test_kvarn_verify: SKIP(by design): no CUDA device (try --dry-run or --interpret)")
        return 0
    tkd, cfg = load_kernels(a.module, a.interpret)
    if not hasattr(tkd, "kvarn_verify_shared_launch"):
        print("test_kvarn_verify: the attn2-kvarn-verify-shared patch is not installed in this vLLM")
        return 2
    f16 = os.environ.get("KVARN_FP16_DEQUANT", "0") == "1"
    print(f"test_kvarn_verify: {dev}, KVARN_FP16_DEQUANT={int(f16)}, tile {cfg.tile_bytes_aligned} B")

    if a.dry_run:
        case = Case(cfg, [300], 8, "cpu")
        o, pva, e_s = case.reference()
        ok = bool(torch.isfinite(o).all()) and o.abs().max().item() > 0
        report(ok, f"dry run: Case + fp64 reference on the CPU (300 + 8 tokens), |o|max {o.abs().max().item():.3f}")
        for sym in ("_kvarn_verify_shared_stage1", "_kvarn_verify_shared_stage2", "verify_shared_splits",
                    "verify_shared_config", "verify_shared_m_pad", "_VERIFY_SHARED"):
            report(hasattr(tkd, sym), f"patch symbol {sym}")
        return 0 if all(r[0] for r in RESULTS) else 1

    if a.interpret:
        cases = [([200], 8, {"n_splits": 3, "config": (16, 4, 1)}),
                 ([40], 2, {"n_splits": 1, "config": (16, 4, 1)}),
                 ([150, -1, 60], 4, {"n_splits": 2, "config": (16, 4, 1)})]
        for ctxs, qlen, kw in cases:
            case = Case(cfg, ctxs, qlen, "cpu", seed=qlen)
            check(case, run_shared(tkd, case, f16=f16, **kw), case.reference(), f"interpret ctx {ctxs} qlen {qlen}")
        return 0 if all(r[0] for r in RESULTS) else 1

    print("1. kernel, rotated frame (shared vs fp64; the base path's numbers alongside, not gated: its output"
          " went through the driver's fp16 un-rotation)")
    big = [] if a.quick else [131072]
    for ctx in [5, 120, 1000, 20000] + big:
        for qscale in (1.0, 4.0):
            case = Case(cfg, [ctx], 8, dev, seed=ctx, qscale=qscale)
            ref = case.reference()
            check(case, run_shared(tkd, case), ref, f"E shape ctx {ctx} qlen 8 qscale {qscale}")
            if ctx >= 1000 and qscale == 1.0:
                out = run_base_rot(tkd, case).double()
                rel = ((out - ref[0]).norm() / ref[0].norm()).item()
                worst = ((out - ref[0]).abs() / case.bound(*ref)).max().item()
                print(f"  info  base per-token path, same case: rel-L2 {rel:.2e}, max err/tol {worst:.3f}")
    for qlen in (2, 3, 4, 7, 16):
        case = Case(cfg, [3000], qlen, dev, seed=qlen)
        check(case, run_shared(tkd, case), case.reference(),
              f"qlen {qlen} (M_PAD {tkd.verify_shared_m_pad(qlen, HQ // HK)}) ctx 3000")
    case = Case(cfg, [500, 9000, -1, 2000], 8, dev, seed=7)
    ref = case.reference()
    check(case, run_shared(tkd, case), ref, "batch [500, 9000, padded, 2000] qlen 8")
    outs = {}
    for s in (1, 3, 41, 64, 256):
        outs[s] = run_shared(tkd, case, n_splits=s)
        check(case, outs[s], ref, f"  splits {s}")
    for pv16, f16x in ((False, not f16), (True, f16), (True, not f16)):
        check(case, run_shared(tkd, case, pv16=pv16, f16=f16x), ref,
              f"  PV16={int(pv16)} F16={int(f16x)}", pv16=pv16)
    if not a.quick:
        case2 = Case(cfg, [20000], 8, dev, seed=3)
        ref2 = case2.reference()
        for bn in (16, 32, 64):
            for nw in (4, 8):
                for ns in (1, 2):
                    try:
                        out = run_shared(tkd, case2, config=(bn, nw, ns))
                    except Exception as e:   # e.g. out of shared memory / resources: not a numerics failure
                        print(f"  SKIP  config BN {bn} W {nw} S {ns}: {type(e).__name__}: {str(e)[:120]}")
                        continue
                    check(case2, out, ref2, f"  config BN {bn} W {nw} S {ns} ctx 20000")
    case = Case(cfg, [2000], 8, dev, seed=11, window=300)
    check(case, run_shared(tkd, case), case.reference(), "sliding window 300, ctx 2000 qlen 8")

    print("2. driver: kvarn_verify_attention, route on vs off, unrotated fp64 reference")
    for ctxs in ([1000], [20000], [500, 9000, 2000]):
        case = Case(cfg, ctxs, 8, dev, seed=sum(ctxs))
        o, _, _ = case.reference()
        ref = (o.reshape(-1, D) @ case.H64).view_as(o)
        res = {}
        for on in (False, True):
            tkd._VERIFY_SHARED = on
            out = tkd.kvarn_verify_attention(case.q, case.kv, case.bt, case.scale, case.cfg, case.impl,
                                             case.vq_req, case.vq_seqlen, case.max_ctx_blocks, qlen=case.qlen)
            res[on] = ((out.double() - ref).norm() / ref.norm()).item()
        tkd._VERIFY_SHARED = False
        ok = res[True] <= 1.25 * res[False] + 1e-4 and res[True] <= 5e-3
        report(ok, f"driver ctx {ctxs}: rel-L2 on {res[True]:.2e} vs off {res[False]:.2e}")

    bad = [w for ok, w in RESULTS if not ok]
    print(f"test_kvarn_verify: {len(RESULTS) - len(bad)}/{len(RESULTS)} OK" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
