// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_prefill_s8.cuh: attn-3 (docs/research/deepinfer/attention-kv.md,
// attn-3; research/deepinfer, opt-in, lossy). QK on int8 in the 64-row-tile
// kernel at D = 256 (the prefill launches of the target's full-attention
// layers), reached only when the library's mode has kAttn3PrefillS8
// (attn_launch.h; PTHFA_PREFILL_QK=s8 in pth_flash_attn.ops).
//
// The cache's K is already int8 per token and head, so no smoothing or
// re-quantisation pass is needed: each Q row is quantised once per CTA by
// its absmax (attn-1 stage 3's stage_q_s8, attn_decode.cuh, unchanged:
// q8 = rn(q 127 / absmax), d in the cache's order, q8_swz layout) and
// S = Q8 K^T runs on m16n8k32.s32.s8.s8.s32 straight from the stage's K
// bytes: no cvt4 dequant of K and one int8 MMA per 8 tokens and 32 d instead
// of two fp16 MMAs with fp32 accumulate (4x the rate on GA10x). The row
// factor becomes fp32(softmax_scale log2 e) * fp32(absmax / 127); the int32
// scores (|S| <= 127^2 * 256 < 2^24) convert to fp32 exactly, and the rest
// of the kernel (k-scale, mask, softmax, PV, merge, epilogue) is unchanged.
// The only new error is Q's int8 rounding.
//
// Fragments (PTX ISA 9.7.16.5.8, m16n8k32 .s8, lane = 4 g + q):
//   A (16 x 32, row): a0 = (row g, k 4q..4q+3), a1 = (row g+8, same k),
//                     a2 = (row g, k 16+4q..), a3 = (row g+8, k 16+4q..)
//                     = ldmatrix.x4 of Q8 rows row0 + 0..7 / 8..15 at the
//                     16-B chunks 2s and 2s+1.
//   B (32 x 8, col):  b0 = (k 4q..4q+3, token g), b1 = (k 16+4q.., token g)
//                     = ldmatrix of K tokens 8n + 0..7 at chunks 2s, 2s+1
//                     (register of lane (g, q) = bytes 4q..4q+3 of token g).
//   C: as the fp16 MMA (c0, c1 = row g, tokens 2q, 2q+1; c2, c3 = row g+8).

#pragma once

#include "attn_decode.cuh"

namespace pth_attn {

#if defined(__CUDACC__)

// S[16 rows x SUB tokens] = Q8[rows row0..row0+15] . K^T for the SUB tokens
// t0.. of a stage, as fp32 (exact) in qk_subtile's layout.
template <int D, int SUB, class R>
__device__ __forceinline__ void qk_subtile_s8(float (&s)[SUB / kMmaN][4],
                                              uint32_t q_smem,
                                              uint32_t stage_smem, int t0,
                                              int row0, int lane) {
  constexpr int kNt = SUB / kMmaN;
  constexpr int kK32 = D / (2 * kMmaK);  // k32 steps over d
  static_assert(kNt % 2 == 0, "one K ldmatrix.x4 feeds two n-tiles");
  int32_t acc[kNt][4];
  #pragma unroll
  for (int n = 0; n < kNt; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0;
  const int mi = lane >> 3, l8 = lane & 7;
  #pragma unroll
  for (int ks = 0; ks < kK32; ++ks) {
    // Q8 matrix mi: rows row0 + 8 (mi & 1) + 0..7 at chunk 2 ks + (mi >> 1)
    // -> registers a0 (rows 0..7, chunk 0), a1 (rows 8..15, chunk 0), a2, a3.
    uint32_t a[4];
    ldmatrix_x4(a, q_smem + q8_swz(row0 + 8 * (mi & 1) + l8,
                                   (2 * ks + (mi >> 1)) * kSmemChunkBytes, D));
  #pragma unroll
    for (int np = 0; np < kNt; np += 2) {
      // K matrix mi: tokens of n-tile np + (mi >> 1) at chunk 2 ks + (mi & 1)
      // -> registers (b0, b1) of n-tile np, then of np + 1.
      uint32_t kb[4];
      ldmatrix_x4(kb, stage_smem + R::k(t0 + (np + (mi >> 1)) * kMmaN + l8,
                                        (2 * ks + (mi & 1)) * kSmemChunkBytes));
      const uint32_t b0[2] = {kb[0], kb[1]};
      const uint32_t b1[2] = {kb[2], kb[3]};
      mma_m16n8k32_s8s32(acc[np], a, b0);
      mma_m16n8k32_s8s32(acc[np + 1], a, b1);
    }
  }
  #pragma unroll
  for (int n = 0; n < kNt; ++n) {
  #pragma unroll
    for (int i = 0; i < 4; ++i) s[n][i] = __int2float_rn(acc[n][i]);
  }
}

// The int8-QK twin of attn_fwd_kernel (a separate name: the shipped kernels
// keep their symbols and machine code).
template <int D, int WM, int SUB, typename QT, bool kCell>
__global__ void __launch_bounds__(kAttnThreads<D>, ctas_per_sm(D))
    attn_fwd_s8_kernel(const __grid_constant__ AttnParams ap) {
  attn_fwd_body<D, WM, SUB, QT, kCell, false, true>(ap);
}

template <int D, int WM, typename QT, bool kCell>
cudaError_t launch_attn_s8_one(const AttnLaunch& L, cudaStream_t stream) {
  constexpr int kSub = default_sub<D, WM>();
  return launch_with_smem<attn_fwd_s8_kernel<D, WM, kSub, QT, kCell>,
                          AttnSmem<D, WM, kSub, kCell>::kBytes,
                          kAttnThreads<D>>(L.grid, stream, L.ap);
}

// launch_attn_wm's choice of q dtype and loader, causal launches only.
template <int D, int WM>
cudaError_t launch_attn_s8_wm(const AttnLaunch& L, cudaStream_t stream) {
  if (L.windowed) return cudaErrorInvalidValue;
  const bool bf16 = L.q_dtype == PTH_ATTN_BF16;
  if constexpr (kAttnHasCell<D>) {
    if (L.cell) {
      return bf16 ? launch_attn_s8_one<D, WM, __nv_bfloat16, true>(L, stream)
                  : launch_attn_s8_one<D, WM, __half, true>(L, stream);
    }
  }
  if (L.cell) return cudaErrorInvalidValue;
  return bf16 ? launch_attn_s8_one<D, WM, __nv_bfloat16, false>(L, stream)
              : launch_attn_s8_one<D, WM, __half, false>(L, stream);
}

#endif  // __CUDACC__

}  // namespace pth_attn
