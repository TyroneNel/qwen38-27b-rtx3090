# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Python entry point for the ``pth_flash_attn_fwd`` op and its fake kernel.

The op itself is C++ (stable ABI, ``csrc/pth_attn/torch_bindings.cpp``); this
module loads that library, registers the op's *fake* (meta) kernel and exposes
a typed wrapper. The C++ op validates every argument, then runs the attention
kernel and (for ``num_splits > 1``) the split combine on the current CUDA
stream (``csrc/pth_attn/launch.cu``); it allocates nothing and reads no device
data on the host, so a call can be captured in a CUDA graph.

Why a fake kernel: ``torch.compile`` (which vLLM uses for the model) traces the
op with FakeTensors, which have shapes but no data. The op is an *out-variant*
(it writes ``out``, ``ws_o`` and ``ws_lse`` in place and returns nothing), so
its fake has no output to construct: it returns None. What it does do is run
the metadata-only subset of the C++ ``TORCH_CHECK`` list (shapes, dtypes,
strides, devices, scalars; ``check_metadata`` in ``csrc/pth_attn/launch.cu``),
with the same messages, so a bad call fails at trace time with the same reason
it would fail with at run time. Checks that need data pointers (alignment,
int64 extents) or a device (compute capability) cannot run on FakeTensors and
stay in C++ only.

Importing this module loads ``_C.abi3.so`` if it has been built (it is a pure
op library without ``PyInit``, so it is loaded with ``torch.ops.load_library``,
not imported). There is no torch.cuda call here: device questions are
answered by the C++ side or by vLLM's ``current_platform``.
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import torch

from pth_flash_attn.planner import (
    MAX_ROWS_PER_CTA,
    MMA_ROWS_PER_WARP,
    SUPPORTED_HEAD_SIZES,
    WINDOWED_HEAD_SIZES,
    WM_CHOICES,
)

OP_NAMESPACE = "pth_flash_attn"
OP_NAME = "pth_flash_attn_fwd"
OP_QUALNAME = f"{OP_NAMESPACE}::{OP_NAME}"
# Frozen interface (docs/api.md), identical to the m.def() text in
# csrc/pth_attn/torch_bindings.cpp (tests parse all three and compare).
# `Tensor!` marks the three tensors the op writes in place (out-variant).
OP_SCHEMA = (
    "pth_flash_attn_fwd(Tensor! out, Tensor q, Tensor k_cache, "
    "Tensor v_cache, Tensor k_scale, Tensor v_scale, Tensor block_table, "
    "Tensor seq_lens, Tensor query_start_loc, int max_query_len, int wm, "
    "int num_splits, float softmax_scale, Tensor! ws_o, Tensor! ws_lse, "
    "int window=0, bool causal=True) -> ()"
)
LIBRARY_PATH = Path(__file__).with_name("_C.abi3.so")

# Ranks of the op's tensors, from the layouts in docs/api.md:
# q/out [T, Hq, D]; caches [NB, BS, Hkv, D]; scales [NB, BS, Hkv];
# block_table [num_reqs, max_blocks]; seq_lens [num_reqs];
# query_start_loc [num_reqs + 1]; ws_o [S, T, Hq, D]; ws_lse [S, T, Hq].
_RANK_TOKENS_HEADS_DIM = 3
_RANK_CACHE = 4
_RANK_SCALE = 3
_RANK_BLOCK_TABLE = 2
_RANK_PER_REQUEST = 1
_RANK_WS_O = 4
_RANK_WS_LSE = 3

# dtype names exactly as the C++ checks print them (launch.cu dtype_name()).
_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int8: "int8",
    torch.int32: "int32",
    torch.int64: "int64",
}


def _fail(*parts: object) -> None:
    raise RuntimeError("pth_flash_attn_fwd: " + "".join(str(p) for p in parts))


def _dtype_name(dtype: torch.dtype) -> str:
    return _DTYPE_NAMES.get(dtype, "an unsupported dtype")


def _check_dtype(t: torch.Tensor, want: torch.dtype, name: str) -> None:
    if t.dtype != want:
        _fail(name, " must be ", _dtype_name(want), ", got ", _dtype_name(t.dtype))


def _check_dim(t: torch.Tensor, want: int, name: str) -> None:
    if t.dim() != want:
        _fail(name, " must be ", want, "-D, got ", t.dim(), "-D")


def _check_size(t: torch.Tensor, dim: int, want: int, name: str, what: str) -> None:
    if t.size(dim) != want:
        _fail(
            name,
            ".size(",
            dim,
            ") must equal ",
            what,
            " = ",
            want,
            ", got ",
            t.size(dim),
        )


def _check_inner_stride_one(t: torch.Tensor, name: str) -> None:
    if t.stride(-1) != 1:
        _fail(name, " must have stride 1 in its last dimension, got ", t.stride(-1))


def _check_same_device(t: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    if t.device != ref.device:
        _fail(name, " must be on the same device as q")


def _check_metadata(
    out: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    wm: int,
    num_splits: int,
    softmax_scale: float,
    ws_o: torch.Tensor,
    ws_lse: torch.Tensor,
    window: int = 0,
    causal: bool = True,
) -> None:
    # Same order and messages as check_metadata() in csrc/pth_attn/launch.cu,
    # so the first failing check, and so the message, is the same on both
    # sides (tests/cpu/test_ops_fake.py checks that pairwise, case by case).
    _check_dim(q, _RANK_TOKENS_HEADS_DIM, "q")
    if q.dtype not in (torch.bfloat16, torch.float16):
        _fail("q must be bfloat16 or float16, got ", _dtype_name(q.dtype))
    num_tokens, num_q_heads, head_size = q.shape
    if head_size not in SUPPORTED_HEAD_SIZES:
        _fail("head size D = q.size(2) must be 64, 128 or 256, got ", head_size)
    _check_inner_stride_one(q, "q")

    _check_dim(out, _RANK_TOKENS_HEADS_DIM, "out")
    _check_dtype(out, q.dtype, "out (same dtype as q)")
    _check_size(out, 0, num_tokens, "out", "T")
    _check_size(out, 1, num_q_heads, "out", "Hq")
    _check_size(out, 2, head_size, "out", "D")
    _check_inner_stride_one(out, "out")

    _check_dim(k_cache, _RANK_CACHE, "k_cache")
    _check_dim(v_cache, _RANK_CACHE, "v_cache")
    _check_dtype(k_cache, torch.int8, "k_cache")
    _check_dtype(v_cache, torch.int8, "v_cache")
    num_blocks, block_size, num_kv_heads = k_cache.shape[:3]
    _check_size(k_cache, 3, head_size, "k_cache", "D")
    _check_size(v_cache, 0, num_blocks, "v_cache", "NB")
    _check_size(v_cache, 1, block_size, "v_cache", "BS")
    _check_size(v_cache, 2, num_kv_heads, "v_cache", "Hkv")
    _check_size(v_cache, 3, head_size, "v_cache", "D")
    _check_inner_stride_one(k_cache, "k_cache")
    _check_inner_stride_one(v_cache, "v_cache")

    _check_dim(k_scale, _RANK_SCALE, "k_scale")
    _check_dim(v_scale, _RANK_SCALE, "v_scale")
    _check_dtype(k_scale, torch.float32, "k_scale")
    _check_dtype(v_scale, torch.float32, "v_scale")
    for scale, name in ((k_scale, "k_scale"), (v_scale, "v_scale")):
        _check_size(scale, 0, num_blocks, name, "NB")
        _check_size(scale, 1, block_size, name, "BS")
        _check_size(scale, 2, num_kv_heads, name, "Hkv")

    if num_kv_heads < 1 or num_q_heads % num_kv_heads != 0:
        _fail(
            "Hq = ", num_q_heads, " must be a positive multiple of Hkv = ", num_kv_heads
        )
    group = num_q_heads // num_kv_heads
    if wm not in WM_CHOICES:
        _fail("wm must be 1, 2 or 4, got ", wm)
    if group > MAX_ROWS_PER_CTA:
        _fail(
            "G = Hq/Hkv = ", group, " exceeds the supported maximum ", MAX_ROWS_PER_CTA
        )
    if group > MMA_ROWS_PER_WARP * wm:
        _fail("G = Hq/Hkv = ", group, " must be <= 16*wm = ", MMA_ROWS_PER_WARP * wm)

    _check_dim(block_table, _RANK_BLOCK_TABLE, "block_table")
    _check_dim(seq_lens, _RANK_PER_REQUEST, "seq_lens")
    _check_dim(query_start_loc, _RANK_PER_REQUEST, "query_start_loc")
    _check_dtype(block_table, torch.int32, "block_table")
    _check_dtype(seq_lens, torch.int32, "seq_lens")
    _check_dtype(query_start_loc, torch.int32, "query_start_loc")
    _check_inner_stride_one(block_table, "block_table")
    _check_inner_stride_one(seq_lens, "seq_lens")
    _check_inner_stride_one(query_start_loc, "query_start_loc")
    num_reqs = seq_lens.size(0)
    _check_size(query_start_loc, 0, num_reqs + 1, "query_start_loc", "num_reqs + 1")
    if block_table.size(0) < num_reqs:
        _fail(
            "block_table has ",
            block_table.size(0),
            " rows, fewer than num_reqs = ",
            num_reqs,
        )

    if not 0 <= max_query_len <= num_tokens:
        _fail("max_query_len must be in [0, T = ", num_tokens, "], got ", max_query_len)
    if num_splits < 1:
        _fail("num_splits must be >= 1, got ", num_splits)
    if not (math.isfinite(softmax_scale) and softmax_scale > 0.0):
        # :g prints a double the way C++ ostream does by default (-1 not -1.0).
        _fail("softmax_scale must be finite and > 0, got ", f"{softmax_scale:g}")
    if window < 0:
        _fail("window must be >= 0, got ", window)
    if (window > 0 or not causal) and head_size not in WINDOWED_HEAD_SIZES:
        _fail(
            "sliding-window and non-causal attention need head size D = 128, got ",
            head_size,
        )

    for t, name in (
        (out, "out"),
        (k_cache, "k_cache"),
        (v_cache, "v_cache"),
        (k_scale, "k_scale"),
        (v_scale, "v_scale"),
        (block_table, "block_table"),
        (seq_lens, "seq_lens"),
        (query_start_loc, "query_start_loc"),
    ):
        _check_same_device(t, q, name)

    # Workspace is read only when S > 1 (docs/api.md). Its leading sizes may
    # exceed S and T (a larger view still works); Hq and D must match.
    if num_splits > 1:
        _check_dim(ws_o, _RANK_WS_O, "ws_o")
        _check_dim(ws_lse, _RANK_WS_LSE, "ws_lse")
        _check_dtype(ws_o, torch.float32, "ws_o")
        _check_dtype(ws_lse, torch.float32, "ws_lse")
        if ws_o.size(0) < num_splits or ws_o.size(1) < num_tokens:
            _fail(
                "ws_o must have size(0) >= num_splits = ",
                num_splits,
                " and size(1) >= T = ",
                num_tokens,
                ", got [",
                ws_o.size(0),
                ", ",
                ws_o.size(1),
                ", ...]",
            )
        _check_size(ws_o, 2, num_q_heads, "ws_o", "Hq")
        _check_size(ws_o, 3, head_size, "ws_o", "D")
        if ws_lse.size(0) < num_splits or ws_lse.size(1) < num_tokens:
            _fail(
                "ws_lse must have size(0) >= num_splits = ",
                num_splits,
                " and size(1) >= T = ",
                num_tokens,
                ", got [",
                ws_lse.size(0),
                ", ",
                ws_lse.size(1),
                ", ...]",
            )
        _check_size(ws_lse, 2, num_q_heads, "ws_lse", "Hq")
        _check_inner_stride_one(ws_o, "ws_o")
        _check_inner_stride_one(ws_lse, "ws_lse")
        _check_same_device(ws_o, q, "ws_o")
        _check_same_device(ws_lse, q, "ws_lse")


def pth_flash_attn_fwd_fake(
    out: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    wm: int,
    num_splits: int,
    softmax_scale: float,
    ws_o: torch.Tensor,
    ws_lse: torch.Tensor,
    window: int = 0,
    causal: bool = True,
) -> None:
    """Fake (meta) kernel: validate metadata like C++, write nothing, return None.

    An out-variant op has no outputs for the fake to create; the in-place
    writes to ``out``/``ws_o``/``ws_lse`` are declared by ``Tensor!`` in the
    schema, which is all the compiler needs to order them correctly.
    """
    _check_metadata(
        out,
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_table,
        seq_lens,
        query_start_loc,
        max_query_len,
        wm,
        num_splits,
        softmax_scale,
        ws_o,
        ws_lse,
        window,
        causal,
    )


def _load_library() -> bool:
    # Absent library = not built yet (CPU-only checkouts run the planner and
    # reference without it); is_available() reports it, the wrapper raises.
    if not LIBRARY_PATH.is_file():
        return False
    torch.ops.load_library(str(LIBRARY_PATH))
    torch.library.register_fake(OP_QUALNAME)(pth_flash_attn_fwd_fake)
    return True


_LIBRARY_LOADED = _load_library()


def is_available() -> bool:
    """Return True if the op library was built and loaded (fake registered)."""
    return _LIBRARY_LOADED


# attn-1 experiments (docs/research/deepinfer/attention-kv.md, attn-1): bits of
# the library's process-wide launch mode (csrc/pth_attn/attn_launch.h). 0, the
# default, is the shipped kernels. Environment knobs, read once at import:
#   PTHFA_TILE48=1  D = 256 row tiles of 33..48 rows (6..8 query tokens at
#                   G = 6) on the swap-AB decode kernel instead of the padded
#                   64-row tile (stage 1, lossless).
#   PTHFA_MMA=f32|pv16|s8pv16
#                   the decode kernel's MMA mode: f32 (default, shipped), pv16
#                   = PV with fp16 accumulate per 64-token tile (stage 2,
#                   near-lossless), s8pv16 = pv16 plus QK on int8 with each Q
#                   row quantised by its absmax (stage 3, lossy). Applies to
#                   launches on the decode kernel: <= 32 rows, or <= 48 rows
#                   with PTHFA_TILE48=1.
ATTN1_TILE48 = 1
ATTN1_MMA_SHIFT = 1
ATTN1_MMA_MODES = {"f32": 0, "pv16": 1, "s8pv16": 2}
ATTN1_PV16 = ATTN1_MMA_MODES["pv16"] << ATTN1_MMA_SHIFT
ATTN1_S8PV16 = ATTN1_MMA_MODES["s8pv16"] << ATTN1_MMA_SHIFT
_ATTN1_KNOWN_BITS = ATTN1_TILE48 | (3 << ATTN1_MMA_SHIFT)
# Arm names for benchmarks and tests ("base", "tile48", "pv16", "s8pv16", joined with "+").
ATTN1_ARMS = {"base": 0, "tile48": ATTN1_TILE48, "pv16": ATTN1_PV16, "s8pv16": ATTN1_S8PV16}


def attn1_mma(mode: int) -> str:
    """Return the MMA-mode name (``ATTN1_MMA_MODES``) of mode bits."""
    field = (mode >> ATTN1_MMA_SHIFT) & 3
    return next((n for n, v in ATTN1_MMA_MODES.items() if v == field), f"unknown({field})")


def attn1_mode_from_env(environ=os.environ) -> int:
    """Return the attn-1 mode bits the environment asks for (0 if unset).

    Raises:
        ValueError: a knob has a value other than the documented ones.
    """
    mode = 0
    tile48 = environ.get("PTHFA_TILE48", "0").strip()
    if tile48 not in ("0", "1", ""):
        raise ValueError(f"PTHFA_TILE48 must be 0 or 1, got {tile48!r}")
    if tile48 == "1":
        mode |= ATTN1_TILE48
    mma = environ.get("PTHFA_MMA", "").strip() or "f32"
    if mma not in ATTN1_MMA_MODES:
        raise ValueError(f"PTHFA_MMA must be one of {'|'.join(ATTN1_MMA_MODES)}, got {mma!r}")
    mode |= ATTN1_MMA_MODES[mma] << ATTN1_MMA_SHIFT
    return mode


def attn1_mode_from_arm(arm: str) -> int:
    """Return the mode bits of an arm name such as ``"tile48"`` (``"base"`` = 0).

    Raises:
        ValueError: an unknown name.
    """
    mode = 0
    for name in arm.strip().split("+"):
        if name.strip() not in ATTN1_ARMS:
            raise ValueError(f"unknown attn-1 arm {name!r} (known: {', '.join(ATTN1_ARMS)})")
        mode |= ATTN1_ARMS[name.strip()]
    return mode


def attn1_arm_name(mode: int) -> str:
    """Return the arm name of mode bits (the inverse of ``attn1_mode_from_arm``)."""
    names = [n for n, b in ATTN1_ARMS.items() if b and mode & b == b]
    return "+".join(names) or "base"


_ATTN1_LIB = None


def _attn1_lib() -> ctypes.CDLL:
    # The same file torch.ops.load_library opened, so dlopen returns that
    # handle: the setter changes the state the op's launches read.
    global _ATTN1_LIB
    if _ATTN1_LIB is None:
        lib = ctypes.CDLL(str(LIBRARY_PATH))
        lib.pth_flash_attn_set_attn1_mode.argtypes = [ctypes.c_int32]
        lib.pth_flash_attn_set_attn1_mode.restype = ctypes.c_int32
        lib.pth_flash_attn_attn1_mode.argtypes = []
        lib.pth_flash_attn_attn1_mode.restype = ctypes.c_int32
        _ATTN1_LIB = lib
    return _ATTN1_LIB


def set_attn1_mode(mode: int) -> int:
    """Set the library's attn-1 mode bits for every later launch; return the previous bits.

    A CUDA graph keeps the kernel of the mode it was captured with.

    Raises:
        RuntimeError: the library is not built.
        ValueError: ``mode`` has a bit this library does not know.
    """
    if not _LIBRARY_LOADED:
        raise RuntimeError(f"pth_flash_attn: op library {LIBRARY_PATH} is not built")
    if mode & ~_ATTN1_KNOWN_BITS or attn1_mma(mode).startswith("unknown"):
        raise ValueError(f"attn-1 mode {mode:#x} has unknown bits (known {_ATTN1_KNOWN_BITS:#x}, "
                         f"MMA modes {ATTN1_MMA_MODES})")
    prev = _attn1_lib().pth_flash_attn_set_attn1_mode(mode)
    if prev < 0:
        raise ValueError(f"the op library refused attn-1 mode {mode:#x} (an older build?)")
    return prev


def attn1_mode() -> int:
    """Return the library's current attn-1 mode bits (0 = the shipped kernels)."""
    if not _LIBRARY_LOADED:
        return 0
    return _attn1_lib().pth_flash_attn_attn1_mode()


if _LIBRARY_LOADED:
    _ENV_ATTN1 = attn1_mode_from_env()
    if _ENV_ATTN1:
        set_attn1_mode(_ENV_ATTN1)


def pth_flash_attn_fwd(
    out: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    wm: int,
    num_splits: int,
    softmax_scale: float,
    ws_o: torch.Tensor,
    ws_lse: torch.Tensor,
    window: int = 0,
    causal: bool = True,
) -> None:
    """Run paged int8 attention for one launch, writing ``out`` in place.

    Arguments follow the frozen schema (docs/api.md). ``wm`` and ``num_splits``
    come from ``planner.plan_launch`` for this slice of the batch; ``ws_o`` and
    ``ws_lse`` are views of the load-time workspace with the launch plan's
    ``ws_o_shape``/``ws_lse_shape`` (may be empty when ``num_splits == 1``).
    ``window`` (0 = none) and ``causal`` are layer constants (vLLM's sliding
    window and non-causal flag): the query at position p sees keys ``[lo, hi]`` with
    ``lo = max(0, p - W + 1)`` (0 when W = 0) and ``hi = p`` if causal, else
    ``min(seq_len - 1, p + W - 1)`` (``seq_len - 1`` when W = 0).
    ``window > 0`` or ``causal=False`` needs D = 128. The defaults
    ``window=0, causal=True`` (also in the schema, so a 15-argument call from
    before the window existed still resolves) are the causal kernel unchanged.
    The call allocates nothing and reads no device data on the host.

    Raises:
        RuntimeError: the library is not built, or an argument is invalid (or
            the tensors are not CUDA tensors: the CPU registration exists only
            to test the checks).
    """
    if not _LIBRARY_LOADED:
        raise RuntimeError(
            f"pth_flash_attn_fwd: op library {LIBRARY_PATH} is not built; run "
            "tools/build_in_image.sh"
        )
    torch.ops.pth_flash_attn.pth_flash_attn_fwd.default(
        out,
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_table,
        seq_lens,
        query_start_loc,
        max_query_len,
        wm,
        num_splits,
        softmax_scale,
        ws_o,
        ws_lse,
        window,
        causal,
    )
