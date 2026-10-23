// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Instantiations of the KVarN verify kernel (attn_decode_kvarn.cuh; research/
// deepinfer attn-2): n-tiles 1..6 (QLEN G <= 48 rows: QLEN 1..8 at G = 6) x
// the three MMA modes, and the combine. Built only with PTHFA_KVARN=1
// (tools/build_flags.py); the op is kvarn_launch.cu.

#include "attn_decode_kvarn.cuh"

namespace pth_attn {
namespace kvarn {
namespace {

// ------------------------------------------------------------- combine
// One warp per output row (token, head): attn_combine.cu's arithmetic (M over
// the live splits, empty splits skipped before their o is read, fixed order),
// without the query_start_loc bound (every row of the launch is real; a
// padded request's rows have no live split and come out 0).
struct CombineParams {
  PartialsView v;
  char* out;
  int64_t s_out_tok, s_out_head;
  int32_t num_rows, num_q_heads, num_splits;
};
constexpr int kCombineWarps = 4;
constexpr int kCombineBatch = 8;

__global__ void __launch_bounds__(kCombineWarps * 32, 1)
    kvarn_combine_kernel(const CombineParams c) {
  constexpr int kCols = kD / 32;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kCombineWarps + threadIdx.x / 32;
  if (row >= c.num_rows) return;
  const int64_t token = row / c.num_q_heads;
  const int32_t head = static_cast<int32_t>(row % c.num_q_heads);
  const int lane = threadIdx.x & 31;
  const int32_t S = c.num_splits;
  float mx = -INFINITY;
  for (int32_t s0 = 0; s0 < S; s0 += kCombineBatch) {
  #pragma unroll
    for (int32_t j = 0; j < kCombineBatch; ++j) {
      const float x = s0 + j < S ? c.v.lse(s0 + j, token, head) : -INFINITY;
      mx = x > mx ? x : mx;
    }
  }
  float num[kCols] = {};
  float den = 0.f;
  if (mx > -INFINITY) {
    for (int32_t s0 = 0; s0 < S; s0 += kCombineBatch) {
      float x[kCombineBatch];
      float ov[kCombineBatch][kCols];
  #pragma unroll
      for (int32_t j = 0; j < kCombineBatch; ++j)
        x[j] = s0 + j < S ? c.v.lse(s0 + j, token, head) : -INFINITY;
  #pragma unroll
      for (int32_t j = 0; j < kCombineBatch; ++j) {
        const float* ob = reinterpret_cast<const float*>(
                              c.v.ws_o + static_cast<int64_t>(s0 + j) * c.v.s_o_split +
                              token * c.v.s_o_token +
                              static_cast<int64_t>(head) * c.v.s_o_head) +
                          lane;
  #pragma unroll
        for (int32_t q = 0; q < kCols; ++q)
          ov[j][q] = x[j] > -INFINITY ? ob[32 * q] : 0.f;
      }
  #pragma unroll
      for (int32_t j = 0; j < kCombineBatch; ++j) {
        if (!(x[j] > -INFINITY)) continue;
        const float w = exp2f(x[j] - mx);
        den += w;
  #pragma unroll
        for (int32_t q = 0; q < kCols; ++q) num[q] = fmaf(w, ov[j][q], num[q]);
      }
    }
  }
  char* orow = c.out + token * c.s_out_tok + static_cast<int64_t>(head) * c.s_out_head;
  #pragma unroll
  for (int32_t q = 0; q < kCols; ++q) {
    store_out(orow + static_cast<int64_t>(lane + 32 * q) * 2, PTH_ATTN_F16,
              mx > -INFINITY ? num[q] / den : 0.f);
  }
}


template <int NT>
cudaError_t launch_nt(const Params& p, int mode, cudaStream_t stream) {
  switch (mode) {
    case kModeF32:
      return launch_split_kernel<NT, kModeF32>(p, stream);
    case kModePv16:
      return launch_split_kernel<NT, kModePv16>(p, stream);
    case kModeQk16:
      return launch_split_kernel<NT, kModeQk16>(p, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

cudaError_t launch_kvarn_verify(const Params& p, int mode, cudaStream_t stream) {
  const int rows = p.qlen * p.group;
  if (rows < 1 || rows > kMaxRows || p.num_reqs < 1 || p.num_splits < 1)
    return cudaErrorInvalidValue;
  cudaError_t err = cudaErrorInvalidValue;
  switch ((rows + kRowsPerNt - 1) / kRowsPerNt) {
    case 1: err = launch_nt<1>(p, mode, stream); break;
    case 2: err = launch_nt<2>(p, mode, stream); break;
    case 3: err = launch_nt<3>(p, mode, stream); break;
    case 4: err = launch_nt<4>(p, mode, stream); break;
    case 5: err = launch_nt<5>(p, mode, stream); break;
    case 6: err = launch_nt<6>(p, mode, stream); break;
    default: break;
  }
  if (err != cudaSuccess || p.num_splits == 1) return err;
  CombineParams c{};
  c.v.ws_o = reinterpret_cast<const char*>(p.ws_o);
  c.v.s_o_split = p.wo_s_split;
  c.v.s_o_token = p.wo_s_tok;
  c.v.s_o_head = p.wo_s_head;
  c.v.ws_lse = reinterpret_cast<const char*>(p.ws_lse);
  c.v.s_l_split = p.wl_s_split;
  c.v.s_l_token = p.wl_s_tok;
  c.v.s_l_head = p.wl_s_head;
  c.out = p.out;
  c.s_out_tok = p.out_s_tok;
  c.s_out_head = p.out_s_head;
  c.num_rows = p.num_reqs * p.qlen * p.num_q_heads;
  c.num_q_heads = p.num_q_heads;
  c.num_splits = p.num_splits;
  const unsigned blocks =
      static_cast<unsigned>((c.num_rows + kCombineWarps - 1) / kCombineWarps);
  kvarn_combine_kernel<<<blocks, kCombineWarps * 32, 0, stream>>>(c);
  return cudaGetLastError();
}

}  // namespace kvarn
}  // namespace pth_attn
