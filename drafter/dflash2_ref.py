"""Pure-torch DFlash2 drafter (no vLLM import), for offline evaluation (eval_dflash2.py,
drafter-2) and training (drafter-3). Mirrors vLLM 0.31's model_executor/models/
qwen3_dflash.py + qwen3_dflash2.py and the drafting path of v1/worker/gpu/spec_decode/
dflash{,2}/speculator.py on this repo's patched image:

  context  rows 0..t-1 of the target: fc(concat of the hidden states after target layers
           target_layer_ids) -> rms_norm(hidden_norm) -> per-layer k_proj/v_proj, k_norm,
           RoPE (DFlashQwen3Model.precompute_and_store_context_kv)
  query    [anchor x_t, mask x K] at positions t..t+K; 5 Qwen3 layers, each wrapped in two
           dynamic grouped convolutions (attention_conv, mlp_conv) that reset at the block
           boundary; non-causal attention over context + block, symmetric sliding window
           (flash_attn._maybe_symmetrize_window)
  draft    hidden rows 1..K -> top-16 candidates of the target lm_head (unary logits) ->
           CandidateSelector edge scores unary + <pred_cb[prev] * proj(h), succ_cb[cand]>,
           walked left to right (prev = the token drafted at the previous slot, the anchor
           for slot 1)

The embedding and lm_head are the target's (the drafter checkpoint ships neither); packed
compressed-tensors weights (the W4A16 drafter, the target's int8/int4 embed and lm_head)
are dequantized to dense bf16 on load.

API (kept stable for drafter/repair_dflash2-style trainers):
  load_drafter(drafter_dir, target_dir=None, device, dtype, shared=None) -> DFlash2Ref
  DFlash2Ref.context_kv(taps[T, n_taps*H], positions[T]) -> [(K[T,nkv,hd], V[T,nkv,hd])] per layer
  DFlash2Ref.block_forward(ctx_kv, ctx_pos[T], anchor_ids[B], anchor_pos[B], k) -> [B, 1+k, H]
  DFlash2Ref.candidates(h[B,L,H]) -> (ids[B,L,C] (no grad), unary[B,L,C] (with grad))
  DFlash2Ref.edge_scores(cand, unary, h, prev_ids[B,L]) -> [B,L,C]   (arbitrary predecessor)
  DFlash2Ref.edge_scores_full(cand, unary, h, anchor_ids[B]) -> [B,L,C,C]   (vLLM _score_edges)
  DFlash2Ref.bf16_state_dict() -> {checkpoint name: tensor} of the non-quantized tensors
  make_tiny_fixture(root, seed=0, quantize=True) -> dict(bf16, w4a16, target, capture) dirs
No in-place ops on tensors that can carry autograd state.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------
# compressed-tensors pack-quantized layout (compressors/pack_quantized/helpers.py):
# int8 codes shifted to unsigned by 2^(b-1), 32//b codes per int32 along dim 1, code j of a
# word at bits [j*b, (j+1)*b).


def unpack_int32(packed, bits, shape):
    """packed int32 [N, ceil(K/(32/bits))] -> int8 codes [N, K]."""
    pf = 32 // bits
    shifts = torch.arange(pf, dtype=torch.int32, device=packed.device) * bits
    vals = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    vals = vals.reshape(packed.shape[0], -1)[:, : int(shape[1])]
    return (vals - (1 << (bits - 1))).to(torch.int8)


def pack_int32(q, bits):
    """int8 codes [N, K] -> packed int32 [N, ceil(K/(32/bits))] (pack_to_int32, packed_dim=1)."""
    pf = 32 // bits
    N, K = q.shape
    u = (q.to(torch.int64) + (1 << (bits - 1))) & ((1 << bits) - 1)
    pad = (-K) % pf
    if pad:
        u = F.pad(u, (0, pad))
    shifts = torch.arange(pf, dtype=torch.int64, device=q.device) * bits
    words = (u.reshape(N, -1, pf) << shifts).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32)


def _packed_bits(packed_cols, k):
    for b in (4, 8, 2, 1):
        if -(-k // (32 // b)) == packed_cols:
            return b
    raise ValueError(f"cannot infer bit width: {packed_cols} int32 columns for K={k}")


def dequant_packed(packed, scale, shape, zero_point=None, g_idx=None):
    """compressed-tensors pack-quantized (symmetric or with zero points, group along K,
    optional g_idx) -> fp32 dense [N, K]."""
    N, K = int(shape[0]), int(shape[1])
    bits = _packed_bits(packed.shape[1], K)
    q = unpack_int32(packed, bits, (N, K)).float()
    if zero_point is not None:
        zp = zero_point
        if zp.dtype == torch.int32 and zp.shape[0] != N:   # packed along dim 0
            zp = unpack_int32(zp.t().contiguous(), bits, (zp.shape[1], N)).t()
        zp = zp.float()
    if g_idx is not None:
        cols = g_idx.long()
        s = scale.float()[:, cols]
        z = zp[:, cols] if zero_point is not None else 0.0
        return (q - z) * s
    group = K // scale.shape[1]
    s = scale.float()[..., None]
    qg = q.reshape(N, K // group, group)
    if zero_point is not None:
        qg = qg - zp[..., None]
    return (qg * s).reshape(N, K)


# ----------------------------------------------------------------------------------------
# safetensors reading (single file or sharded index)


def _st_files(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        files = {}
        for k, f in wm.items():
            files.setdefault(f, []).append(k)
        return [(os.path.join(d, f), ks) for f, ks in files.items()]
    return [(os.path.join(d, "model.safetensors"), None)]


def read_tensors(d, want=None):
    """{name: tensor} for every tensor in d (optionally filtered by want(name) -> bool)."""
    from safetensors import safe_open
    out = {}
    for path, keys in _st_files(d):
        if keys is not None and want is not None and not any(want(k) for k in keys):
            continue
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if want is None or want(k):
                    out[k] = f.get_tensor(k)
    return out


def _dense_from(tensors, base, dtype=torch.float32, chunk=16384):
    """(dense weight in dtype, was_packed) for checkpoint module `base`. Packed weights are
    dequantized in row chunks so a 248k-row head never exists in fp32 at once."""
    if base + ".weight" in tensors:
        return tensors[base + ".weight"].to(dtype), False
    if base + ".weight_packed" in tensors:
        shape = tensors.get(base + ".weight_shape")
        packed = tensors[base + ".weight_packed"]
        scale = tensors[base + ".weight_scale"]
        if shape is None:
            raise ValueError(f"{base}: weight_shape missing")
        N, K = (int(x) for x in shape.tolist())
        zp, gi = tensors.get(base + ".weight_zero_point"), tensors.get(base + ".weight_g_idx")
        out = torch.empty(N, K, dtype=dtype)
        for a in range(0, N, chunk):
            b = min(N, a + chunk)
            z = None
            if zp is not None:                             # per row, or packed along dim 0
                rows_per = 1 if zp.shape[0] == N else -(-N // zp.shape[0])
                z = zp[a // rows_per: -(-b // rows_per)]
            out[a:b] = dequant_packed(packed[a:b], scale[a:b], (b - a, K), z, gi).to(dtype)
        return out, True
    raise KeyError(f"no weight for {base}")


def load_target_heads(target_dir, device="cpu", dtype=torch.bfloat16):
    """(embed [V,H], lm_head [V,H]) of the target, dequantized to dtype. Skips mtp.* tensors."""
    def is_head(k):
        if k.startswith("mtp") or ".mtp." in k:
            return False
        return ("embed_tokens." in k) or k.startswith("lm_head.")
    t = read_tensors(target_dir, is_head)
    emb_base = next((k.rsplit(".", 1)[0] for k in t if "embed_tokens." in k), None)
    lm_base = "lm_head" if any(k.startswith("lm_head.") for k in t) else None
    if emb_base is None:
        raise KeyError(f"no embed_tokens in {target_dir}")
    emb, _ = _dense_from(t, emb_base, dtype)
    if lm_base is None:   # tied embeddings
        lm = emb
    else:
        lm, _ = _dense_from(t, lm_base, dtype)
    return emb.to(device=device, dtype=dtype), lm.to(device=device, dtype=dtype)


# ----------------------------------------------------------------------------------------
# config


class DFlash2Config:
    def __init__(self, cfg):
        dc = dict(cfg.get("eagle_config") or {})
        dc.update(cfg.get("dflash_config") or {})
        self.raw = cfg
        self.H = int(cfg["hidden_size"])
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg.get("head_dim") or self.H // self.n_heads)
        self.inter = int(cfg["intermediate_size"])
        self.n_layers = int(cfg["num_hidden_layers"])
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        rp = cfg.get("rope_parameters") or {}
        self.rope_theta = float(rp.get("rope_theta", cfg.get("rope_theta", 1e6)))
        if rp.get("rope_type", "default") not in ("default", None):
            raise NotImplementedError(f"rope_type {rp.get('rope_type')}")
        self.neox = bool(cfg.get("is_neox_style", True))
        self.vocab = int(cfg["vocab_size"])
        self.block_size = int(dc.get("block_size", 1 << 30))
        self.conv_group = int(dc["conv_group_size"])
        self.conv_taps = int(dc["conv_kernel_size"])
        self.mask_token_id = int(dc["mask_token_id"])
        self.rank = int(dc["selector_rank"])
        self.top_k = int(dc["selector_top_k"])
        self.target_layer_ids = list(dc["target_layer_ids"])
        self.input_embedding_scale = float(dc.get("input_embedding_scale", 1.0))
        self.output_multiplier = float(dc.get("output_multiplier", 1.0))
        self.softcap = float(dc.get("final_logit_softcapping") or 0.0)
        self.v_scale = dc.get("attention_value_scale")
        if dc.get("attention_sink_bias", cfg.get("add_swa_attention_sink_bias", False)):
            raise NotImplementedError("attention sink bias")
        if dc.get("sample_from_anchor", False):
            raise NotImplementedError("sample_from_anchor")
        self.target_H = int(cfg.get("target_hidden_size") or self.H)
        # per-layer (sliding_window, causal), qwen3_dflash._resolve_layer_attention
        lt = cfg.get("layer_types")
        use_swa = dc.get("use_swa", False)
        any_sliding = bool(lt) and any(x == "sliding_attention" for x in lt)
        self.windows, self.causal = [], []
        for i in range(self.n_layers):
            if lt is None or (use_swa and not any_sliding):
                sliding = bool(use_swa)
            else:
                sliding = lt[i] == "sliding_attention"
            w = dc.get("swa_window_size", cfg.get("sliding_window")) if sliding else None
            self.windows.append(int(w) if w is not None else None)
            if cfg.get("is_causal") is not None:
                c = bool(cfg["is_causal"])
            elif dc.get("causal") is not None:
                c = bool(dc["causal"])
            else:
                c = bool(lt) and lt[i] == "sliding_attention"
            self.causal.append(c)

    @classmethod
    def from_dir(cls, d):
        return cls(json.load(open(os.path.join(d, "config.json"))))


# ----------------------------------------------------------------------------------------
# modules (parameter names == checkpoint keys)


class Lin(nn.Module):
    def __init__(self, n_out, n_in, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in, dtype=dtype), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


class Norm(nn.Module):
    def __init__(self, n, eps, dtype):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(n, dtype=dtype), requires_grad=False)

    def forward(self, x):
        # vLLM RMSNorm: normalize in fp32, cast back, then scale by the weight.
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        return y * self.weight


class GroupedConv(nn.Module):
    """qwen3_dflash2.DFlashGroupedConv on [B, L, H] blocks (fp32 accumulate, like the
    Triton kernel). Position within the block is l % block_size."""

    def __init__(self, H, taps, group, dtype):
        super().__init__()
        self.taps, self.group, self.G = taps, group, H // group
        self.base_kernel = nn.Parameter(torch.empty(2, taps, H, dtype=dtype), requires_grad=False)
        self.kernel_projection = Lin(2 * taps * self.G, H, dtype)

    def _convolve(self, x, delta, side, block_size):
        B, L, H = x.shape
        xs = x.float().reshape(B, L, self.G, self.group)
        base = self.base_kernel[side].float().reshape(self.taps, self.G, self.group)
        coef = base[None, None] + delta.float()[..., None]          # [B,L,taps,G,gs]
        out = coef[:, :, 0] * xs
        pos = torch.arange(L, device=x.device) % block_size
        for tap in range(1, self.taps):
            shifted = F.pad(xs[:, :-tap], (0, 0, 0, 0, tap, 0))
            keep = (pos >= tap).to(xs.dtype).view(1, L, 1, 1)
            out = out + coef[:, :, tap] * shifted * keep
        return out.reshape(B, L, H).to(x.dtype)

    def prepare(self, x, block_size):
        B, L, _ = x.shape
        c = self.kernel_projection(x).reshape(B, L, 2, self.taps, self.G)
        return self._convolve(x, c[:, :, 0], 0, block_size), c[:, :, 1]

    def finish(self, x, coef, block_size):
        return self._convolve(x, coef, 1, block_size)


class Attn(nn.Module):
    def __init__(self, c, dtype):
        super().__init__()
        self.c = c
        self.q_proj = Lin(c.n_heads * c.head_dim, c.H, dtype)
        self.k_proj = Lin(c.n_kv * c.head_dim, c.H, dtype)
        self.v_proj = Lin(c.n_kv * c.head_dim, c.H, dtype)
        self.o_proj = Lin(c.H, c.n_heads * c.head_dim, dtype)
        self.q_norm = Norm(c.head_dim, c.eps, dtype)
        self.k_norm = Norm(c.head_dim, c.eps, dtype)


class MLP(nn.Module):
    def __init__(self, c, dtype):
        super().__init__()
        self.gate_proj = Lin(c.inter, c.H, dtype)
        self.up_proj = Lin(c.inter, c.H, dtype)
        self.down_proj = Lin(c.H, c.inter, dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c, dtype):
        super().__init__()
        self.self_attn = Attn(c, dtype)
        self.mlp = MLP(c, dtype)
        self.input_layernorm = Norm(c.H, c.eps, dtype)
        self.post_attention_layernorm = Norm(c.H, c.eps, dtype)
        self.attention_conv = GroupedConv(c.H, c.conv_taps, c.conv_group, dtype)
        self.mlp_conv = GroupedConv(c.H, c.conv_taps, c.conv_group, dtype)


class Selector(nn.Module):
    def __init__(self, c, dtype):
        super().__init__()
        self.predecessor_codebook = nn.Parameter(torch.empty(c.vocab, c.rank, dtype=dtype), requires_grad=False)
        self.successor_codebook = nn.Parameter(torch.empty(c.vocab, c.rank, dtype=dtype), requires_grad=False)
        self.hidden_projection = Lin(c.rank, c.H, dtype)


def _rope(x, pos, theta, neox):
    """x [..., n, hd] with pos broadcastable to x[..., 0, 0]; vLLM keeps cos/sin in the
    model dtype, the rotation here runs in fp32 on those rounded values."""
    hd = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    ang = pos.to(torch.float32)[..., None] * inv                       # [..., hd/2]
    cos = ang.cos().to(x.dtype).float()[..., None, :]
    sin = ang.sin().to(x.dtype).float()[..., None, :]
    xf = x.float()
    if neox:
        x1, x2 = xf[..., : hd // 2], xf[..., hd // 2:]
        out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)
    else:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).flatten(-2)
    return out.to(x.dtype)


class DFlash2Ref(nn.Module):
    def __init__(self, c, dtype=torch.bfloat16):
        super().__init__()
        self.c = c
        self.dtype = dtype
        self.layers = nn.ModuleList([Layer(c, dtype) for _ in range(c.n_layers)])
        self.fc = Lin(c.H, c.target_H * len(c.target_layer_ids), dtype)
        self.hidden_norm = Norm(c.H, c.eps, dtype)
        self.norm = Norm(c.H, c.eps, dtype)
        self.candidate_selector = Selector(c, dtype)
        self.register_buffer("embed", torch.empty(0, dtype=dtype), persistent=False)
        self.register_buffer("lm_head", torch.empty(0, dtype=dtype), persistent=False)
        self.quantized_names = set()      # checkpoint module bases that were packed (dense here)
        self.source_dir = None

    # ---- weights ------------------------------------------------------------------------
    def bf16_state_dict(self):
        """{checkpoint key: tensor} for every tensor that is stored unquantized."""
        out = {}
        for n, p in self.named_parameters():
            base = n[: -len(".weight")] if n.endswith(".weight") else None
            if base in self.quantized_names:
                continue
            out[n] = p
        return out

    # ---- context ------------------------------------------------------------------------
    def context_states(self, taps):
        """fc over the concatenated tap rows (DFlashQwen3ForCausalLM.combine_hidden_states)."""
        return self.fc(taps.to(self.dtype))

    def context_kv(self, taps, positions):
        """Per-layer (K roped+normed, V) for context rows [T, n_taps*H] at positions [T]."""
        c = self.c
        h = self.hidden_norm(self.context_states(taps))
        T = h.shape[0]
        kv = []
        for layer in self.layers:
            a = layer.self_attn
            k = a.k_norm(a.k_proj(h).view(T, c.n_kv, c.head_dim))
            k = _rope(k, positions, c.rope_theta, c.neox)
            v = a.v_proj(h).view(T, c.n_kv, c.head_dim)
            if c.v_scale is not None:
                v = v * c.v_scale
            kv.append((k, v))
        return kv

    # ---- block --------------------------------------------------------------------------
    def embed_ids(self, ids):
        e = F.embedding(ids, self.embed)
        if self.c.input_embedding_scale != 1.0:
            e = e * self.c.input_embedding_scale
        return e

    def _attention(self, li, a, x, ctx_k, ctx_v, ctx_pos, anchor_pos, qpos):
        c = self.c
        B, L, _ = x.shape
        g = c.n_heads // c.n_kv
        q = a.q_norm(a.q_proj(x).view(B, L, c.n_heads, c.head_dim))
        k = a.k_norm(a.k_proj(x).view(B, L, c.n_kv, c.head_dim))
        v = a.v_proj(x).view(B, L, c.n_kv, c.head_dim)
        if c.v_scale is not None:
            v = v * c.v_scale
        q = _rope(q, qpos, c.rope_theta, c.neox)
        k = _rope(k, qpos, c.rope_theta, c.neox)
        qg = q.float().view(B, L, c.n_kv, g, c.head_dim)
        scale = c.head_dim ** -0.5
        s_ctx = torch.einsum("blkgd,tkd->bkglt", qg, ctx_k.float()) * scale       # [B,kv,g,L,T]
        s_blk = torch.einsum("blkgd,bmkd->bkglm", qg, k.float()) * scale          # [B,kv,g,L,L]
        W = c.windows[li]
        diff = qpos[:, :, None] - ctx_pos[None, None, :]                          # [B,L,T]
        ok_ctx = ctx_pos[None, None, :] < anchor_pos[:, None, None]
        if W is not None:
            ok_ctx = ok_ctx & (diff < W)
        ok_blk = torch.ones(L, L, dtype=torch.bool, device=x.device)
        if c.causal[li]:
            ok_blk = torch.tril(ok_blk)
        if W is not None:
            idx = torch.arange(L, device=x.device)
            ok_blk = ok_blk & ((idx[:, None] - idx[None, :]).abs() < W)
        s_ctx = s_ctx.masked_fill(~ok_ctx[:, None, None], float("-inf"))
        s_blk = s_blk.masked_fill(~ok_blk[None, None, None], float("-inf"))
        p = torch.softmax(torch.cat((s_ctx, s_blk), -1), -1)
        T = ctx_k.shape[0]
        o = torch.einsum("bkglt,tkd->blkgd", p[..., :T], ctx_v.float()) + \
            torch.einsum("bkglm,bmkd->blkgd", p[..., T:], v.float())
        return a.o_proj(o.reshape(B, L, c.n_heads * c.head_dim).to(x.dtype))

    def block_forward(self, ctx_kv, ctx_pos, anchor_ids, anchor_pos, k=7):
        """Final normed drafter hidden [B, 1+k, H] for B anchors of one sequence. Anchor b
        (token anchor_ids[b] at position anchor_pos[b]) sees context rows with position <
        anchor_pos[b] inside each layer's window; row 0 is the anchor slot, row j predicts
        position anchor_pos[b] + j."""
        c = self.c
        B = anchor_ids.shape[0]
        L = 1 + k
        cb = min(L, c.block_size)
        ids = torch.full((B, L), c.mask_token_id, dtype=torch.long, device=anchor_ids.device)
        ids = torch.cat((anchor_ids.view(B, 1).long(), ids[:, 1:]), 1)
        qpos = anchor_pos.view(B, 1).long() + torch.arange(L, device=anchor_ids.device)
        x = self.embed_ids(ids)
        residual = x
        h = self.layers[0].input_layernorm(x)
        for li, layer in enumerate(self.layers):
            if li > 0:
                residual = h + residual
                h = layer.input_layernorm(residual)
            h, coef = layer.attention_conv.prepare(h, cb)
            h = self._attention(li, layer.self_attn, h, ctx_kv[li][0], ctx_kv[li][1],
                                ctx_pos, anchor_pos.long(), qpos)
            h = layer.attention_conv.finish(h, coef, cb)
            residual = h + residual
            h = layer.post_attention_layernorm(residual)
            h, coef = layer.mlp_conv.prepare(h, cb)
            h = layer.mlp(h)
            h = layer.mlp_conv.finish(h, coef, cb)
        return self.norm(h + residual)

    # ---- candidates and selector --------------------------------------------------------
    def _scale_logits(self, x):
        c = self.c
        if c.softcap > 0:
            x = torch.tanh(x / c.softcap) * c.softcap
        if c.output_multiplier != 1.0:
            x = x * c.output_multiplier
        return x

    def candidates(self, h):
        """Top-C candidate ids of the target lm_head (no grad) and their unary logits (with
        grad through h), LogitsProcessor.get_top_k_tokens without return_log_probs."""
        with torch.no_grad():
            logits = F.linear(h.detach(), self.lm_head)
            ids = logits.topk(self.c.top_k, dim=-1).indices
        unary = torch.einsum("...d,...cd->...c", h.float(), self.lm_head[ids].float()).to(h.dtype)
        return ids, self._scale_logits(unary)

    def edge_scores(self, cand, unary, h, prev_ids):
        """[B,L,C] scores of each slot's candidates given an arbitrary predecessor token per
        slot (prev_ids[:, 0] = the anchor). Rounds like vLLM's bf16 einsum."""
        sel = self.candidate_selector
        hp = sel.hidden_projection(h)                                       # [B,L,R]
        pred = (sel.predecessor_codebook[prev_ids] * hp).to(h.dtype)        # [B,L,R]
        succ = sel.successor_codebook[cand]                                 # [B,L,C,R]
        edge = torch.einsum("blr,blcr->blc", pred.float(), succ.float()).to(h.dtype)
        return unary + edge

    def edge_scores_full(self, cand, unary, h, anchor_ids):
        """[B,L,C,C] exactly as qwen3_dflash2._score_edges: predecessor p of slot l is
        candidate p of slot l-1 (the anchor, broadcast, for slot 0)."""
        sel = self.candidate_selector
        B, L, C = cand.shape
        hp = sel.hidden_projection(h)
        prev = torch.cat((anchor_ids.view(B, 1, 1).expand(B, 1, C), cand[:, :-1]), 1)
        pred = sel.predecessor_codebook[prev]                               # [B,L,C,R]
        succ = sel.successor_codebook[cand]
        prod = (pred * hp[:, :, None]).to(h.dtype)
        edge = torch.einsum("blpr,blcr->blpc", prod.float(), succ.float()).to(h.dtype)
        return unary[:, :, None] + edge


def load_drafter(drafter_dir, target_dir=None, device="cpu", dtype=torch.bfloat16, shared=None):
    """DFlash2Ref with the drafter's weights (packed ones dequantized) and the target's
    embed/lm_head (from target_dir, or reused from `shared`)."""
    c = DFlash2Config.from_dir(drafter_dir)
    with torch.device("meta"):
        m = DFlash2Ref(c, dtype)
    t = read_tensors(drafter_dir)
    params = dict(m.named_parameters())
    state, quantized = {}, set()
    for name in params:
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        if name in t:
            state[name] = t[name]
        elif base is not None and base + ".weight_packed" in t:
            w, _ = _dense_from(t, base, dtype)
            state[name] = w
            quantized.add(base)
        else:
            raise KeyError(f"{drafter_dir}: missing {name}")
    m = m.to_empty(device=device)
    for name, p in m.named_parameters():
        want = tuple(p.shape)
        got = state[name]
        if tuple(got.shape) != want:
            raise ValueError(f"{name}: checkpoint {tuple(got.shape)} vs model {want}")
        p.data = got.to(device=device, dtype=dtype)
    m.quantized_names = quantized
    m.source_dir = drafter_dir
    if shared is not None:
        m.embed, m.lm_head = shared.embed, shared.lm_head
    else:
        if target_dir is None:
            raise ValueError("load_drafter needs target_dir (embed/lm_head) or shared=")
        m.embed, m.lm_head = load_target_heads(target_dir, device, dtype)
    if m.embed.shape[0] < c.vocab or m.lm_head.shape[1] != c.H:
        raise ValueError(f"target heads {tuple(m.lm_head.shape)} do not fit drafter vocab {c.vocab} / H {c.H}")
    return m.eval()


# ----------------------------------------------------------------------------------------
# capture directory (written by capture_dflash2.py --dump-target)
#   meta.json     {taps, hidden, n_seq, n_tok, dtype, topk, window, target, sampling{...}}
#   index.jsonl   {seq, off, len, prompt_len, start_pos, domain, think, mode, seed, src}
#   taps.bin      bf16 (stored as uint16) [n_tok, n_taps*hidden]  fc-input order
#   ids.bin       int32 [n_tok]
#   topk_ids.bin  int32 [n_tok, topk]   target top-k at row i = prediction of token i+1,
#   topk_logits.bin fp32 [n_tok, topk]  raw pre-temperature logits, sorted descending
#   lse.bin       fp32 [n_tok]          logsumexp of the full raw logits row


class Capture:
    def __init__(self, d):
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.seqs = [json.loads(l) for l in open(os.path.join(d, "index.jsonl")) if l.strip()]
        n, w, k = self.meta["n_tok"], len(self.meta["taps"]) * self.meta["hidden"], self.meta["topk"]
        mm = lambda f, dt, shape: np.memmap(os.path.join(d, f), dtype=dt, mode="r", shape=shape)
        self.taps = mm("taps.bin", np.uint16, (n, w))
        self.ids = mm("ids.bin", np.int32, (n,))
        self.topk_ids = mm("topk_ids.bin", np.int32, (n, k))
        self.topk_logits = mm("topk_logits.bin", np.float32, (n, k))
        self.lse = mm("lse.bin", np.float32, (n,))

    def seq(self, s, device="cpu"):
        a, n = s["off"], s["len"]
        taps = torch.from_numpy(np.array(self.taps[a:a + n])).view(torch.bfloat16).to(device)
        g = lambda arr: torch.from_numpy(np.array(arr[a:a + n])).to(device)
        return dict(taps=taps, ids=g(self.ids).long(), topk_ids=g(self.topk_ids).long(),
                    topk_logits=g(self.topk_logits), lse=g(self.lse),
                    pos=torch.arange(n, device=device) + int(s["start_pos"]))


class CaptureWriter:
    def __init__(self, d, meta):
        os.makedirs(d, exist_ok=True)
        for f in ("taps.bin", "ids.bin", "topk_ids.bin", "topk_logits.bin", "lse.bin", "index.jsonl", "meta.json"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                raise FileExistsError(f"{p} exists; capture directories are write-once")
        self.d, self.meta, self.n_tok, self.n_seq = d, dict(meta), 0, 0
        self.f = {k: open(os.path.join(d, k + ".bin"), "wb") for k in ("taps", "ids", "topk_ids", "topk_logits", "lse")}
        self.index = open(os.path.join(d, "index.jsonl"), "w")

    def add(self, rec, taps, ids, topk_ids, topk_logits, lse):
        n = int(ids.shape[0])
        w = len(self.meta["taps"]) * self.meta["hidden"]
        k = self.meta["topk"]
        assert tuple(taps.shape) == (n, w) and tuple(topk_ids.shape) == (n, k) and lse.shape[0] == n
        self.f["taps"].write(taps.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes())
        self.f["ids"].write(ids.to(torch.int32).cpu().numpy().tobytes())
        self.f["topk_ids"].write(topk_ids.to(torch.int32).cpu().numpy().tobytes())
        self.f["topk_logits"].write(topk_logits.to(torch.float32).cpu().numpy().tobytes())
        self.f["lse"].write(lse.to(torch.float32).cpu().numpy().tobytes())
        rec = dict(rec, seq=self.n_seq, off=self.n_tok, len=n)
        self.index.write(json.dumps(rec) + "\n")
        self.index.flush()
        self.n_tok += n
        self.n_seq += 1

    def close(self):
        for f in self.f.values():
            f.close()
        self.index.close()
        meta = dict(self.meta, n_seq=self.n_seq, n_tok=self.n_tok)
        json.dump(meta, open(os.path.join(self.d, "meta.json"), "w"), indent=1)


def topk_rows(logits, k):
    """(topk ids, topk logits sorted descending, logsumexp) of raw logit rows (fp32 math)."""
    lf = logits.float()
    v, i = lf.topk(k, dim=-1)
    return i, v, torch.logsumexp(lf, -1)


# ----------------------------------------------------------------------------------------
# tiny synthetic fixture (CPU dry runs, trainer smoke tests)

TINY = dict(H=128, heads=4, kv=2, hd=32, inter=256, layers=2, vocab=512, taps=[1, 3, 5], rank=16,
            top_k=8, block=8, conv_group=16, conv_taps=2, window=48)


def _tiny_config(t):
    return {
        "architectures": ["DFlash2DraftModel"], "is_causal": False, "model_type": "qwen3",
        "dflash_config": {"block_size": t["block"], "conv_group_size": t["conv_group"],
                          "conv_kernel_size": t["conv_taps"], "mask_token_id": t["vocab"] - 2,
                          "selector_rank": t["rank"], "selector_top_k": t["top_k"],
                          "target_layer_ids": t["taps"]},
        "dtype": "bfloat16", "head_dim": t["hd"], "hidden_act": "silu", "hidden_size": t["H"],
        "intermediate_size": t["inter"], "layer_types": ["sliding_attention"] * t["layers"],
        "max_position_embeddings": 4096, "num_attention_heads": t["heads"],
        "num_hidden_layers": t["layers"], "num_key_value_heads": t["kv"], "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": t["window"], "use_sliding_window": True, "vocab_size": t["vocab"],
    }


def _rtn(W, bits, group=128):
    N, K = W.shape
    qmax = 2 ** (bits - 1) - 1
    g = W.float().reshape(N, K // group, group)
    s = torch.clamp(g.abs().amax(-1) / qmax, min=1e-8)
    q = torch.clamp(torch.round(g / s[..., None]), -qmax - 1, qmax).to(torch.int8).reshape(N, K)
    return q, s


def _save_packed(out, base, W, bits, scale_dtype=torch.float16):
    q, s = _rtn(W, bits)
    out[base + ".weight_packed"] = pack_int32(q, bits)
    out[base + ".weight_scale"] = s.to(scale_dtype)
    out[base + ".weight_shape"] = torch.tensor(list(W.shape), dtype=torch.int64)


def make_tiny_fixture(root, seed=0, quantize=True, n_seq=6, prompt_len=40, resp_len=24):
    """Synthetic bf16 drafter, its W4A16 requant (quant_dflash2.py layout, RTN), a target
    dir holding only int8 embed / int4 lm_head (compressed-tensors packed), and a capture dir
    with random taps and random target logits. Returns dict(bf16, w4a16, target, capture)."""
    from safetensors.torch import save_file
    t = TINY
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s, std=1.0: (torch.randn(*s, generator=g) * std).to(torch.bfloat16)
    cfg = _tiny_config(t)
    H, I, V, R = t["H"], t["inter"], t["vocab"], t["rank"]
    qd, kd = t["heads"] * t["hd"], t["kv"] * t["hd"]
    G = H // t["conv_group"]
    w = {"fc.weight": rn(H, H * len(t["taps"]), std=(H * len(t["taps"])) ** -0.5),
         "hidden_norm.weight": torch.ones(H, dtype=torch.bfloat16),
         "norm.weight": torch.ones(H, dtype=torch.bfloat16),
         "candidate_selector.predecessor_codebook": rn(V, R, std=0.5),
         "candidate_selector.successor_codebook": rn(V, R, std=0.5),
         "candidate_selector.hidden_projection.weight": rn(R, H, std=H ** -0.5)}
    for i in range(t["layers"]):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = rn(qd, H, std=H ** -0.5)
        w[p + "self_attn.k_proj.weight"] = rn(kd, H, std=H ** -0.5)
        w[p + "self_attn.v_proj.weight"] = rn(kd, H, std=H ** -0.5)
        w[p + "self_attn.o_proj.weight"] = rn(H, qd, std=qd ** -0.5)
        w[p + "self_attn.q_norm.weight"] = torch.ones(t["hd"], dtype=torch.bfloat16)
        w[p + "self_attn.k_norm.weight"] = torch.ones(t["hd"], dtype=torch.bfloat16)
        w[p + "mlp.gate_proj.weight"] = rn(I, H, std=H ** -0.5)
        w[p + "mlp.up_proj.weight"] = rn(I, H, std=H ** -0.5)
        w[p + "mlp.down_proj.weight"] = rn(H, I, std=I ** -0.5)
        w[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        w[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        for conv in ("attention_conv", "mlp_conv"):
            base = torch.zeros(2, t["conv_taps"], H)
            base[:, 0] = 1.0
            base[:, 1:] = torch.randn(2, t["conv_taps"] - 1, H, generator=g) * 0.1
            w[p + conv + ".base_kernel"] = base.to(torch.bfloat16)
            w[p + conv + ".kernel_projection.weight"] = rn(2 * t["conv_taps"] * G, H, std=0.02)
    dirs = {k: os.path.join(root, k) for k in ("bf16", "w4a16", "target", "capture")}
    os.makedirs(dirs["bf16"], exist_ok=True)
    save_file(w, os.path.join(dirs["bf16"], "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(dirs["bf16"], "config.json"), "w"), indent=1)
    if quantize:
        os.makedirs(dirs["w4a16"], exist_ok=True)
        q = {}
        lin = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        for name, W in w.items():
            base = name[: -len(".weight")]
            if name == "fc.weight" or (name.endswith(".weight") and base.rsplit(".", 1)[-1] in lin):
                _save_packed(q, base, W, 4)
            else:
                q[name] = W
        save_file(q, os.path.join(dirs["w4a16"], "model.safetensors"), metadata={"format": "pt"})
        qcfg = dict(cfg)
        qcfg["quantization_config"] = {"config_groups": {"group_0": {
            "format": "pack-quantized", "input_activations": None, "output_activations": None,
            "targets": ["Linear"], "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                                               "group_size": 128, "num_bits": 4, "observer": "minmax",
                                               "observer_kwargs": {}, "strategy": "group", "symmetric": True,
                                               "type": "int"}}},
            "format": "pack-quantized", "ignore": ["re:.*kernel_projection$", "re:.*candidate_selector.*",
                                                   "re:.*hidden_projection$"],
            "quant_method": "compressed-tensors", "quantization_status": "compressed"}
        json.dump(qcfg, open(os.path.join(dirs["w4a16"], "config.json"), "w"), indent=1)
    # target heads: int8 embed (bf16 scales), int4 lm_head (fp16 scales), like the fast variant
    os.makedirs(dirs["target"], exist_ok=True)
    th = {}
    _save_packed(th, "model.language_model.embed_tokens", rn(V, H), 8, torch.bfloat16)
    _save_packed(th, "lm_head", rn(V, H, std=H ** -0.5 * 4), 4)
    save_file(th, os.path.join(dirs["target"], "model.safetensors"), metadata={"format": "pt"})
    json.dump({"vocab_size": V, "hidden_size": H, "tie_word_embeddings": False},
              open(os.path.join(dirs["target"], "config.json"), "w"))
    json.dump({"temperature": 1.0, "top_k": 20, "top_p": 0.95},
              open(os.path.join(dirs["target"], "generation_config.json"), "w"))
    # capture: random taps / ids / target logits; half greedy, half sample, two domains
    topk = 64
    wr = CaptureWriter(dirs["capture"], {
        "taps": t["taps"], "hidden": H, "dtype": "bfloat16", "topk": topk, "window": t["window"],
        "target": dirs["target"], "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        "synthetic": True})
    for s in range(n_seq):
        n = prompt_len + resp_len
        logits = torch.randn(n, V, generator=g) * 3.0
        ids = torch.randint(0, V - 2, (n,), generator=g)
        mode = "greedy" if s % 2 == 0 else "sample"
        if mode == "greedy":   # greedy reference: next token = target argmax (alignment check)
            ids[prompt_len:] = logits[prompt_len - 1:-1].argmax(-1)
        else:                  # sampled reference: next token from the top-20 of the target
            top = logits[prompt_len - 1:-1].topk(20, -1)
            pick = torch.multinomial(torch.softmax(top.values, -1), 1, generator=g).squeeze(-1)
            ids[prompt_len:] = top.indices.gather(-1, pick[:, None]).squeeze(-1)
        ti, tv, lse = topk_rows(logits, topk)
        wr.add({"prompt_len": prompt_len, "start_pos": 3 * s, "domain": ["dom_a", "dom_b"][(s // 2) % 2],
                "think": False, "mode": mode, "seed": 0, "src": "synthetic"},
               rn(n, H * len(t["taps"])), ids, ti, tv, lse)
    wr.close()
    return dirs
