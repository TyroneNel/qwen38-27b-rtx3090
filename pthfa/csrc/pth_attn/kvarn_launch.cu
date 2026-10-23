// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// The op pth_flash_attn::pth_kvarn_verify_fwd (research/deepinfer attn-2):
// argument checks, then the KVarN verify kernel and, for num_splits > 1, its
// combine on the current CUDA stream (attn_decode_kvarn.cuh,
// instantiate_kvarn.cu). Built only with PTHFA_KVARN=1 (tools/build_flags.py);
// registered as a FRAGMENT of the pth_flash_attn library, so the shipped op
// and its registration (torch_bindings.cpp) are untouched. Like launch.cu it
// allocates nothing, reads no device data on the host and does not
// synchronise: a call can be captured in a CUDA graph.

#include <cuda_runtime.h>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cmath>
#include <cstdint>
#include <vector>

#define PTH_KVARN_PARAMS_ONLY  // Params and the launcher only, no kernels here
#include "attn_decode_kvarn.cuh"

// As launch.cu: the stable-ABI current-stream call, declared here.
extern "C" int32_t aoti_torch_get_current_cuda_stream(int32_t device_index,
                                                      void** ret_stream);

namespace pth_attn {
namespace kvarn {
namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

#define KV_CHECK(cond, ...) \
  STD_TORCH_CHECK(cond, "pth_kvarn_verify_fwd: ", __VA_ARGS__)

void check_tensor(const Tensor& t, const Tensor& ref, ScalarType dtype,
                  int64_t dim, const char* name) {
  KV_CHECK(t.dim() == dim, name, " must be ", dim, "-D, got ", t.dim(), "-D");
  KV_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
  KV_CHECK(t.is_cuda() && t.get_device_index() == ref.get_device_index(), name,
           " must be a CUDA tensor on q_rot's device");
  KV_CHECK(t.stride(dim - 1) == 1, name,
           " must have stride 1 in its last dimension");
  for (int64_t i = 0; i < dim; ++i) {
    KV_CHECK(t.stride(i) >= 0, name, " has a negative stride");
  }
}

void check_align(const Tensor& t, int64_t bytes, const char* name) {
  KV_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0, name,
           " data pointer must be ", bytes, "-byte aligned");
  const int64_t es = static_cast<int64_t>(t.element_size());
  for (int64_t i = 0; i + 1 < t.dim(); ++i) {
    KV_CHECK((t.stride(i) * es) % bytes == 0, name, ".stride(", i,
             ") must be a multiple of ", bytes, " bytes");
  }
}

int64_t bstride(const Tensor& t, int64_t dim) {
  return t.stride(dim) * static_cast<int64_t>(t.element_size());
}

}  // namespace

// Schema (kvarn_launch.cu STABLE_TORCH_LIBRARY_FRAGMENT below):
//   out_rot       [NQ, Hq, 256] fp16, written (rotated frame).
//   q_rot         [NQ, Hq, 256] fp16, NQ = B * qlen, request b = rows
//                 b qlen .. b qlen + qlen - 1; 16-B aligned rows.
//   kv_cache      [NB, Hk, 26880] uint8 KVarN k4v2 g128 tiles.
//   tail_k_pool / tail_v_pool  [POOL, 128, Hk, 256] fp16.
//   block_to_slot [lookup] int32: pool slot of a block id, < 0 = quantised.
//   block_table   [>= B, max_blocks] int32.
//   vq_seqlen     [NQ] int32: token i's causal length; request b's length
//                 is entry b qlen + qlen - 1 (<= 0: padded request).
//   qlen          query tokens per request, qlen * Hq / Hk <= 48.
//   num_splits    KV splits S >= 1; S > 1 needs ws_o [>= S, >= NQ, Hq, 256]
//                 and ws_lse [>= S, >= NQ, Hq] fp32.
//   tile_offsets  KVarNConfig's 8 tile offsets (k_packed, k_s_col, k_zp,
//                 k_s_row, v_packed, v_s_col, v_s_row, v_zp): must equal the
//                 compiled layout, which is the only one implemented.
//   window        sliding window (0 = none), as the Triton kernel's.
//   mma           0 = fp32 accumulate, 1 = PV16, 2 = QK16 + PV16.
void pth_kvarn_verify_fwd(Tensor& out_rot, const Tensor& q_rot,
                          const Tensor& kv_cache, const Tensor& tail_k_pool,
                          const Tensor& tail_v_pool,
                          const Tensor& block_to_slot,
                          const Tensor& block_table, const Tensor& vq_seqlen,
                          int64_t qlen, int64_t num_splits,
                          double softmax_scale, Tensor& ws_o, Tensor& ws_lse,
                          std::vector<int64_t> tile_offsets, int64_t window,
                          int64_t mma) {
  check_tensor(q_rot, q_rot, ScalarType::Half, 3, "q_rot");
  const int64_t nq = q_rot.size(0), hq = q_rot.size(1);
  KV_CHECK(q_rot.size(2) == kD, "head size must be ", kD, ", got ",
           q_rot.size(2));
  check_align(q_rot, 16, "q_rot");
  check_tensor(out_rot, q_rot, ScalarType::Half, 3, "out_rot");
  KV_CHECK(out_rot.size(0) == nq && out_rot.size(1) == hq &&
               out_rot.size(2) == kD,
           "out_rot must have q_rot's shape");
  check_align(out_rot, 2, "out_rot");

  check_tensor(kv_cache, q_rot, ScalarType::Byte, 3, "kv_cache");
  const int64_t hk = kv_cache.size(1);
  KV_CHECK(kv_cache.size(2) == kTileBytes, "kv_cache.size(2) must be the ",
           kTileBytes, "-byte k4v2 g128 D 256 tile, got ", kv_cache.size(2));
  check_align(kv_cache, 16, "kv_cache");
  KV_CHECK(hk >= 1 && hq % hk == 0, "Hq = ", hq,
           " must be a positive multiple of Hk = ", hk);
  const int64_t group = hq / hk;
  KV_CHECK(qlen >= 1 && qlen * group <= kMaxRows, "qlen * G = ", qlen, " * ",
           group, " must be in [1, ", kMaxRows, "]");
  KV_CHECK(nq % qlen == 0, "NQ = ", nq, " must be a multiple of qlen = ", qlen);

  check_tensor(tail_k_pool, q_rot, ScalarType::Half, 4, "tail_k_pool");
  check_tensor(tail_v_pool, q_rot, ScalarType::Half, 4, "tail_v_pool");
  for (const Tensor* t : {&tail_k_pool, &tail_v_pool}) {
    KV_CHECK(t->size(1) == kBlockTokens && t->size(2) == hk && t->size(3) == kD,
             "tail pools must be [POOL, ", kBlockTokens, ", Hk, ", kD, "]");
    check_align(*t, 4, "tail pool");
  }
  check_tensor(block_to_slot, q_rot, ScalarType::Int, 1, "block_to_slot");
  check_tensor(block_table, q_rot, ScalarType::Int, 2, "block_table");
  check_tensor(vq_seqlen, q_rot, ScalarType::Int, 1, "vq_seqlen");
  KV_CHECK(vq_seqlen.size(0) == nq, "vq_seqlen must have NQ = ", nq,
           " entries, got ", vq_seqlen.size(0));
  const int64_t num_reqs = nq / qlen;
  KV_CHECK(block_table.size(0) >= num_reqs, "block_table has ",
           block_table.size(0), " rows, fewer than B = ", num_reqs);

  KV_CHECK(num_splits >= 1 && num_splits <= 65535,
           "num_splits must be in [1, 65535], got ", num_splits);
  KV_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
           "softmax_scale must be finite and > 0");
  KV_CHECK(window >= 0 && window <= INT32_MAX, "window must be >= 0");
  KV_CHECK(mma >= 0 && mma < kNumModes, "mma must be 0 (f32), 1 (pv16) or 2 "
           "(qk16), got ", mma);
  KV_CHECK(static_cast<int64_t>(tile_offsets.size()) == kNumTileOffsets,
           "tile_offsets must have ", kNumTileOffsets, " entries");
  for (int i = 0; i < kNumTileOffsets; ++i) {
    KV_CHECK(tile_offsets[i] == kTileOffsets[i], "tile offset ", i, " is ",
             tile_offsets[i], "; only the k4v2 g128 D 256 layout (offset ",
             kTileOffsets[i], ") is implemented");
  }
  if (num_splits > 1) {
    check_tensor(ws_o, q_rot, ScalarType::Float, 4, "ws_o");
    check_tensor(ws_lse, q_rot, ScalarType::Float, 3, "ws_lse");
    KV_CHECK(ws_o.size(0) >= num_splits && ws_o.size(1) >= nq &&
                 ws_o.size(2) == hq && ws_o.size(3) == kD,
             "ws_o must be [>= S, >= NQ, Hq, ", kD, "]");
    KV_CHECK(ws_lse.size(0) >= num_splits && ws_lse.size(1) >= nq &&
                 ws_lse.size(2) == hq,
             "ws_lse must be [>= S, >= NQ, Hq]");
  }
  if (nq == 0) return;

  int major = 0;
  const cudaError_t qerr = cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, q_rot.get_device_index());
  KV_CHECK(qerr == cudaSuccess && major == 8,
           "needs compute capability 8.x (sm_80/86/89)");

  Params p{};
  p.q = static_cast<const char*>(q_rot.data_ptr());
  p.q_s_tok = bstride(q_rot, 0);
  p.q_s_head = bstride(q_rot, 1);
  p.kv = static_cast<const uint8_t*>(kv_cache.data_ptr());
  p.kv_s_block = bstride(kv_cache, 0);
  p.kv_s_head = bstride(kv_cache, 1);
  p.pool_k = static_cast<const char*>(tail_k_pool.data_ptr());
  p.pk_s_slot = bstride(tail_k_pool, 0);
  p.pk_s_tok = bstride(tail_k_pool, 1);
  p.pk_s_head = bstride(tail_k_pool, 2);
  p.pool_v = static_cast<const char*>(tail_v_pool.data_ptr());
  p.pv_s_slot = bstride(tail_v_pool, 0);
  p.pv_s_tok = bstride(tail_v_pool, 1);
  p.pv_s_head = bstride(tail_v_pool, 2);
  p.block_to_slot = static_cast<const int32_t*>(block_to_slot.data_ptr());
  p.lookup = static_cast<int32_t>(block_to_slot.size(0));
  p.block_table = static_cast<const int32_t*>(block_table.data_ptr());
  p.bt_s_row = block_table.stride(0);
  p.vq_seqlen = static_cast<const int32_t*>(vq_seqlen.data_ptr());
  p.qlen = static_cast<int32_t>(qlen);
  p.group = static_cast<int32_t>(group);
  p.num_reqs = static_cast<int32_t>(num_reqs);
  p.num_kv_heads = static_cast<int32_t>(hk);
  p.num_q_heads = static_cast<int32_t>(hq);
  p.num_splits = static_cast<int32_t>(num_splits);
  p.window = static_cast<int32_t>(window);
  p.score_scale_log2 =
      static_cast<float>(softmax_scale) * 1.4426950408889634f;
  p.out = static_cast<char*>(out_rot.data_ptr());
  p.out_s_tok = bstride(out_rot, 0);
  p.out_s_head = bstride(out_rot, 1);
  if (num_splits > 1) {
    p.ws_o = static_cast<float*>(ws_o.data_ptr());
    p.wo_s_split = bstride(ws_o, 0);
    p.wo_s_tok = bstride(ws_o, 1);
    p.wo_s_head = bstride(ws_o, 2);
    p.ws_lse = static_cast<float*>(ws_lse.data_ptr());
    p.wl_s_split = bstride(ws_lse, 0);
    p.wl_s_tok = bstride(ws_lse, 1);
    p.wl_s_head = bstride(ws_lse, 2);
  }

  const int32_t device = static_cast<int32_t>(q_rot.get_device_index());
  const torch::stable::accelerator::DeviceGuard guard(device);
  void* stream = nullptr;
  KV_CHECK(aoti_torch_get_current_cuda_stream(device, &stream) == 0,
           "cannot get the current CUDA stream");
  const cudaError_t err =
      launch_kvarn_verify(p, static_cast<int>(mma), static_cast<cudaStream_t>(stream));
  KV_CHECK(err == cudaSuccess, "launch: ", cudaGetErrorString(err));
}

}  // namespace kvarn
}  // namespace pth_attn

STABLE_TORCH_LIBRARY_FRAGMENT(pth_flash_attn, m) {
  m.def(
      "pth_kvarn_verify_fwd(Tensor! out_rot, Tensor q_rot, Tensor kv_cache, "
      "Tensor tail_k_pool, Tensor tail_v_pool, Tensor block_to_slot, "
      "Tensor block_table, Tensor vq_seqlen, int qlen, int num_splits, "
      "float softmax_scale, Tensor! ws_o, Tensor! ws_lse, int[] tile_offsets, "
      "int window=0, int mma=0) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(pth_flash_attn, CUDA, m) {
  m.impl("pth_kvarn_verify_fwd",
         TORCH_BOX(&pth_attn::kvarn::pth_kvarn_verify_fwd));
}
