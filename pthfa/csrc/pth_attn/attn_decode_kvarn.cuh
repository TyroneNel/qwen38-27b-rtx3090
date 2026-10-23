// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_decode_kvarn.cuh: KVarN's uniform speculative verify (research/deepinfer
// attn-2, docs/research/deepinfer/attention-kv.md; build-time opt-in,
// PTHFA_KVARN=1, see tools/build_flags.py). Not part of the shipped library:
// nothing in the default build includes this header.
//
// What it computes (the semantics of triton_kvarn_decode.py's
// _kvarn_verify_shared_stage1 + _stage2, attn2-kvarn-verify-shared.patch):
// for request b of QLEN query tokens and kv head hk, the QLEN x G query rows
// (row r = token r / G, q head hk G + r % G; q in the ROTATED frame) attend
// the request's KVarN k4v2 g128 cache at D = 256. Request length L =
// vq_seqlen[b QLEN + QLEN - 1] (L <= 0: a padded request, all-zero output,
// no cache read); token j sees kv positions < L - QLEN + j + 1 (bottom-right
// causal), and >= that limit - window when window > 0. Block k of the
// request is block_table[b][k]; if block_to_slot[block] >= 0 its 128 tokens
// are fp16 rows of the tail pools [slot, 128, Hk, D] instead (sink block,
// open tail block, unflushed blocks). Output: out_rot [NQ, Hq, D] fp16, still
// rotated (the driver multiplies by the Hadamard outside).
//
// Tile layout (per block and kv head, 26880 B; KVarNConfig k4v2 g128 D 256):
//   K codes     u4 [d][t]  at     0 (16384 B; byte d*64 + t/2, nibble t%2)
//   s_col_K     f16 [256]  at 16384      zp_K f16 [256] at 16896
//   s_row_K     f16 [128]  at 17408
//   V codes     u2 [t][d]  at 17664 (8192 B; byte t*64 + d/4, shift 2(d%4))
//   s_col_V     f16 [256]  at 25856      s_row_V f16 [128] at 26368
//   zp_V        f16 [128]  at 26624
//   K[t][d] = (code s_col_K[d] + zp_K[d]) s_row_K[t]
//   V[t][d] = (code s_row_V[t] + zp_V[t]) s_col_V[d]
//
// Design. One CTA of 8 warps per (request, kv head, KV split) serves all
// QLEN G <= 48 rows, so every cache byte is read and dequantised once per
// split. Swap-AB as attn_decode.cuh: the MMA (m16n8k16, fp16 operands)
// computes S^T = K Q^T (M = 16 tokens, N = 8 rows, K = 16 d) and
// O^T = V^T P^T (M = 16 d, N = 8 rows, K = 16 tokens), so the row tile is
// NT = ceil(QLEN G / 8) n-tiles of 8 rows (QLEN 8 at G 6: NT 6, no padding;
// QLEN 4: NT 3, none either). Per 128-token block (one tile):
//   0  barrier; cp.async (16 B, .cg) of the NEXT block's whole tile into the
//      other of two stages (double buffer), K and V rows chunk-swizzled so
//      that the fragment loads below are bank-conflict free;
//   1  QK, warp w: tokens 16w .. 16w + 15, all NT n-tiles. The A fragment is
//      built straight from the stage's code words: four LDS.32 (d0, d0 + 1,
//      d0 + 8, d0 + 9 at the warp's token word), nibble -> fp16 1024 + code
//      (shift + LOP3), PRMT pairs the two d of a register, then
//      fp16(code s_col + zp) = HFMA2(HSUB2(x, 1024), s_col, zp): one rounding,
//      exactly the reference's fp16 K. s_row_K and softmax_scale log2(e) are
//      applied to the fp32 scores. Q^T comes from shared memory (ldmatrix).
//   2  causal / window mask (only on blocks that reach a row's limit), row
//      max over the warp's 16 tokens (3 xor shuffles), exchange [row][warp];
//      barrier A;
//   3  online softmax (log2 domain), O^T rescale, P = fp16(2^(s - m)) to a
//      shared [row][token] buffer; barrier B;
//   4  PV, warp w: d 32w .. 32w + 31 (two m-tiles), all 128 tokens: A from
//      V code words (LDS.32 of tokens t0, t0 + 1, t0 + 8, t0 + 9, 2-bit crumb
//      -> 1024 + code, PRMT token pairing), V = fp16(fp16(code s_row + zp)
//      s_col) (HSUB2, HFMA2, HMUL2: two roundings), B = P via ldmatrix.
// Tail-pool blocks (fp16) skip the stage: their A fragments are loaded from
// global (L2) directly, already fp16 (LDG.32 pairs along d for K, LDG.16
// pairs along tokens for V), tokens past the block's end read as 0. They are
// few (sink + open tail + unflushed) and slower per block than a quantised one.
// End of split: l summed over lanes and warps; S = 1 writes the normalised
// fp16 output, S > 1 writes fp32 partials + log2-sum-exp ([S][T][Hq][D],
// [S][T][Hq], pthfa's ws layout) and the combine kernel (instantiate_kvarn.cu)
// merges them (combine_kernel.cuh's rule: empty splits skipped before their o
// is read).
//
// MMA modes (template kMode; op argument mma, vLLM knob KVARN_PTHFA_MMA):
//   kModeF32  QK and PV accumulate in fp32 (m16n8k16.f32.f16.f16.f32).
//   kModePv16 PV accumulates each 64-token half block in fp16 from zero
//             (m16n8k16.f16), added to the fp32 O. |P| <= 1, so a partial is
//             <= 64 max|V|: overflow needs |V| > 1023. Bound: + 2^-7 p@|v|
//             (test_kvarn_verify's PV16 term).
//   kModeQk16 kModePv16 plus QK on fp16 accumulate per 16-d k-step (one MMA
//             from zero, converted and added to the fp32 score): two fp16
//             roundings of a 16-term partial, so the score error doubles to
//             ~3 * 2^-11 sum_d |q| |K| (the test gates it at 2 e_s).
//
// Arithmetic budget per 128-token tile and kv head (QLEN 8, G 6: NT = 6; the
// MMA count scales with NT, bytes and dequant do not):
//   bytes    26880 from DRAM, once (1680 cp.async of 16 B, 6.6 per thread);
//            shared reads ~16 KB K + 8 KB V codes x1, 2.3 KB scales,
//            Q 24 KB x 8 warps (ldmatrix), P 12 KB x 8 warps.
//   MMAs     QK 8 m-tiles x NT x 16 k-steps = 768, PV 16 x NT x 8 = 768.
//            GA102 (3090): 256 dense fp16 FMA/clk/SM with fp32 accumulate,
//            512 with fp16 accumulate; an m16n8k16 is 2048 FMA, so a tile
//            costs 1536 x 8 = 12288 SM clocks (kModeF32), 768 x 8 + 768 x 4
//            = 9216 (kModePv16), 1536 x 4 = 6144 (kModeQk16).
//   DRAM     936 GB/s / 82 SMs = 6.7 B/clk/SM at 1.695 GHz: 3994 clocks per
//            tile. So at NT = 6 the kernel is MMA-bound, 3.1x (F32), 2.3x
//            (PV16), 1.5x (QK16) its DRAM floor; at NT = 3 (QLEN 4) 1.5x /
//            1.15x / 0.77x; at NT = 2, 1.0x / 0.77x / 0.5x. At 240k tokens
//            (7500 tiles per layer, 91.5 per SM): F32 0.66 ms, PV16 0.50 ms,
//            QK16 0.33 ms vs the 215 us floor (1.695 GHz; [INFERENCE]: the
//            clock under the 3090's power cap is lower, and nothing here is
//            measured).
//   dequant  K: 16384 f16x2 registers, per register 2 LDS.32 (code word +
//            scale pair, halved by sharing), SHF, LOP3, PRMT, HSUB2, HFMA2;
//            V: 16384 registers, LDS.32, SHF, LOP3, PRMT, HSUB2, HFMA2, HMUL2
//            (+ the scale pairs): ~7.4k warp instructions per tile, ~1.9k
//            issue slots per scheduler against 12288 (F32) / 6144 (QK16) MMA
//            clocks. kModeQk16 adds 8 instructions per QK MMA (two
//            HADD2.F32-class conversions, four FADD): ~1.5k slots more.
//   registers  per thread: O^T 2 m-tiles x NT x 4 = 48 fp32, S^T NT x 4 =
//            24, m and l 2 NT each = 24, A fragments 8 (QK) / 16 (PV); the
//            launch bound is 256 threads, 1 CTA/SM (<= 255 registers).
//   smem     2 x 26880 (stages) + 8 NT x 512 (Q) + 8 NT x 272 (P) +
//            8 NT x 8 x 4 (exchange) + 8 NT x 8 (limits) = 93312 B at NT 6:
//            one CTA per SM (sm_86 allows 101376 per block), 8 warps.
// Split count: one wave, num_splits = SMs / (B Hk) (20 at B = 1, 82 SMs);
// pthfa.ops.kvarn_verify_splits.

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "combine_kernel.cuh"
#include "ptx.cuh"

namespace pth_attn {
namespace kvarn {

// ---------------------------------------------------------------- layout
constexpr int kD = 256;
constexpr int kBlockTokens = 128;
constexpr int kKPacked = 0;
constexpr int kKSCol = 16384;
constexpr int kKZp = 16896;
constexpr int kKSRow = 17408;
constexpr int kVPacked = 17664;
constexpr int kVSCol = 25856;
constexpr int kVSRow = 26368;
constexpr int kVZp = 26624;
constexpr int kTileBytes = 26880;
// The order the op's tile_offsets argument lists them in (KVarNConfig
// attribute order: k_packed, k_s_col, k_zp, k_s_row, v_packed, v_s_col,
// v_s_row, v_zp).
constexpr int kNumTileOffsets = 8;
constexpr int64_t kTileOffsets[kNumTileOffsets] = {
    kKPacked, kKSCol, kKZp, kKSRow, kVPacked, kVSCol, kVSRow, kVZp};

constexpr int kCodeRowBytes = 64;  // K: one d (128 x 4 bit); V: one t (256 x 2)
constexpr int kChunk = 16;
constexpr int kTileChunks = kTileBytes / kChunk;          // 1680
constexpr int kKPackedEnd = kKSCol / kChunk;              // 1024
constexpr int kVPackedBegin = kVPacked / kChunk;          // 1104
constexpr int kVPackedEnd = kVSCol / kChunk;              // 1616
static_assert(kVPacked % kChunk == 0 && kVSCol % kChunk == 0 &&
                  kTileBytes % kChunk == 0,
              "regions are 16-B aligned");

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kRowsPerNt = 8;
constexpr int kMaxNt = 6;
constexpr int kMaxRows = kMaxNt * kRowsPerNt;  // 48
constexpr int kQPitch = kD * 2;                         // 512 B, chunk-swizzled
constexpr int kPPitch = kBlockTokens * 2 + kChunk;      // 272 B: 8 rows, 8 bank groups

enum Mode : int { kModeF32 = 0, kModePv16 = 1, kModeQk16 = 2 };
constexpr int kNumModes = 3;
PTH_HD constexpr bool mode_pv16(int m) { return m != kModeF32; }
PTH_HD constexpr bool mode_qk16(int m) { return m == kModeQk16; }
// Unroll widths of the quantised QK (8 k-step pairs) and PV (4) loops;
// overridable at compile time for register/spill tuning. ptxas -v, sm_86:
// full QK unroll spills at NT 5/6 in kModeQk16 (236 B at NT 6); 4 (and 2)
// is spill-free at every instantiation.
#ifndef PTH_KVARN_QK_UNROLL
#define PTH_KVARN_QK_UNROLL 4
#endif
#ifndef PTH_KVARN_PV_UNROLL
#define PTH_KVARN_PV_UNROLL 4
#endif
constexpr int kQkUnroll = PTH_KVARN_QK_UNROLL;
constexpr int kPvUnroll = PTH_KVARN_PV_UNROLL;

template <int NT>
struct Smem {
  static constexpr int kRows = kRowsPerNt * NT;
  static constexpr int kStage = kTileBytes;  // stage s at s * kStage
  static constexpr int kQ = 2 * kTileBytes;
  static constexpr int kP = kQ + kRows * kQPitch;
  static constexpr int kXchg = kP + kRows * kPPitch;
  static constexpr int kLim = kXchg + kRows * kWarps * 4;
  static constexpr int kLo = kLim + kRows * 4;
  static constexpr int kBytes = kLo + kRows * 4;
  static_assert(kQ % 128 == 0 && kP % 16 == 0 && kXchg % 16 == 0,
                "ldmatrix / float4 regions are 16-B aligned");
  static_assert(kBytes <= 101376, "one CTA per SM must fit (sm_86)");
};

// Launch parameters. Byte strides throughout (pthfa binding rule 5), except
// the block table's row stride (int32 elements).
struct Params {
  const char* q;  // [NQ, Hq, D] fp16, rotated
  int64_t q_s_tok, q_s_head;
  const uint8_t* kv;  // [NB, Hk, 26880] uint8
  int64_t kv_s_block, kv_s_head;
  const char* pool_k;  // [POOL, 128, Hk, D] fp16
  int64_t pk_s_slot, pk_s_tok, pk_s_head;
  const char* pool_v;
  int64_t pv_s_slot, pv_s_tok, pv_s_head;
  const int32_t* block_to_slot;  // [lookup]
  int32_t lookup;
  const int32_t* block_table;  // [>= B, max_blocks]
  int64_t bt_s_row;
  const int32_t* vq_seqlen;  // [NQ]
  int32_t qlen, group, num_reqs, num_kv_heads, num_q_heads, num_splits;
  int32_t window;
  float score_scale_log2;  // softmax_scale * log2(e)
  char* out;               // [NQ, Hq, D] fp16
  int64_t out_s_tok, out_s_head;
  float* ws_o;  // [>= S, >= NQ, Hq, D] fp32 (S > 1 only)
  int64_t wo_s_split, wo_s_tok, wo_s_head;
  float* ws_lse;  // [>= S, >= NQ, Hq] fp32, log2-sum-exp
  int64_t wl_s_split, wl_s_tok, wl_s_head;
};

// Host entry (instantiate_kvarn.cu): the split kernel and, for S > 1, the
// combine, on `stream`. mode is a Mode. Returns the first CUDA error.
cudaError_t launch_kvarn_verify(const Params& p, int mode, cudaStream_t stream);

// The op TU (kvarn_launch.cu) needs only Params and the launcher's
// declaration: it defines PTH_KVARN_PARAMS_ONLY.
#if defined(__CUDACC__) && !defined(PTH_KVARN_PARAMS_ONLY)

// ------------------------------------------------------ local instructions
// The f16x2 arithmetic, ex2 and scalar cvt this kernel needs beyond ptx.cuh.
// Kept here, not in ptx.cuh, so that the shipped headers stay byte-identical
// with the opt-in build off.
PTH_DEVICE uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
PTH_DEVICE uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
PTH_DEVICE float ex2(float x) {  // 2^x, 2^-22 relative; ex2(-inf) = +0
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
PTH_DEVICE float h2f(uint16_t h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
PTH_DEVICE uint16_t f2h(float f) {
  uint16_t h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
PTH_DEVICE float lo_f(uint32_t x) { return h2f(static_cast<uint16_t>(x)); }
PTH_DEVICE float hi_f(uint32_t x) { return h2f(static_cast<uint16_t>(x >> 16)); }

PTH_DEVICE uint32_t lds32(const uint8_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
PTH_DEVICE uint16_t lds16(const uint8_t* p) {
  return *reinterpret_cast<const uint16_t*>(p);
}
PTH_DEVICE uint32_t ldg32(const char* p) {
  return __ldg(reinterpret_cast<const unsigned int*>(p));
}
PTH_DEVICE uint32_t ldg16(const char* p) {
  return __ldg(reinterpret_cast<const unsigned short*>(p));
}

// fp16 1024 + code in each half: the magic-number int -> fp16 conversion.
constexpr uint32_t kF16x2Magic = 0x64006400u;  // (1024, 1024)
PTH_DEVICE uint32_t nib2(uint32_t w, int sh) {  // codes at bits sh, 16 + sh
  return ((w >> sh) & 0x000F000Fu) | kF16x2Magic;
}
PTH_DEVICE uint32_t crumb2(uint32_t w, int sh) {
  return ((w >> sh) & 0x00030003u) | kF16x2Magic;
}
constexpr uint32_t kPrmtLo = 0x5410u;  // (a.lo, b.lo)
constexpr uint32_t kPrmtHi = 0x7632u;  // (a.hi, b.hi)

// Physical tokens of the QK A-fragment rows g and g + 8 of warp w: the
// nibble pair (j, j + 4) of the warp's token word 2w + (g >> 2), j = g & 3.
PTH_DEVICE int qk_token_a(int warp, int g) {
  return 16 * warp + 8 * (g >> 2) + (g & 3);
}

// cp.async of one whole tile into a stage; K rows (64 B per d) and V rows
// (64 B per token) have their four 16-B chunks XOR-swizzled by (row >> 1) & 3.
PTH_DEVICE void issue_tile(uint32_t stage_smem, const uint8_t* tile, int tid) {
  #pragma unroll
  for (int i = 0; i < (kTileChunks + kThreads - 1) / kThreads; ++i) {
    const int ci = tid + i * kThreads;
    if (ci < kTileChunks) {
      int dst = ci;
      if (ci < kKPackedEnd) {
        const int r = ci >> 2;
        dst = (r << 2) | ((ci & 3) ^ ((r >> 1) & 3));
      } else if (ci >= kVPackedBegin && ci < kVPackedEnd) {
        const int v = ci - kVPackedBegin;
        const int r = v >> 2;
        dst = kVPackedBegin + ((r << 2) | ((v & 3) ^ ((r >> 1) & 3)));
      }
      cp_async<16>(stage_smem + dst * kChunk, tile + ci * kChunk);
    }
  }
}

// S^T[n] = K[16 tokens of the warp] . Q^T[rows 8n .. 8n + 7].
template <int NT, int kMode, bool kPool>
PTH_DEVICE void qk(float (&s)[NT][4], const uint8_t* st, uint32_t q_smem,
                   const char* pk_a, const char* pk_b, bool ok_a, bool ok_b,
                   int warp, int lane) {
  const int g = lane >> 2, q4 = lane & 3, mi = lane >> 3, l8 = lane & 7;
  #pragma unroll
  for (int n = 0; n < NT; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
  // Quantised: the thread's code words at d = 2 q4 (+ 16 ks + {0, 1, 8, 9}).
  const uint8_t* kb = st + kKPacked + (2 * q4) * kCodeRowBytes +
                      ((((warp >> 1) ^ q4)) << 4) +
                      (((2 * warp + (g >> 2)) & 3) << 2);
  const int sh = 4 * (g & 3);
  const uint8_t* scol = st + kKSCol + 4 * q4;
  const uint8_t* zp = st + kKZp + 4 * q4;
  // The pool path's global loads are not hoisted across k-step pairs (each
  // pair's 16 LDG.32 are already in flight together); the quantised path is
  // unrolled kQkUnroll-wide.
  #pragma unroll (kPool ? 1 : kQkUnroll)
  for (int ks2 = 0; ks2 < kD / 32; ++ks2) {
    uint32_t a[2][4];
  #pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int ks = 2 * ks2 + e;
      if constexpr (kPool) {
        const int off = 2 * (16 * ks + 2 * q4);
        a[e][0] = ok_a ? ldg32(pk_a + off) : 0u;
        a[e][1] = ok_b ? ldg32(pk_b + off) : 0u;
        a[e][2] = ok_a ? ldg32(pk_a + off + 16) : 0u;
        a[e][3] = ok_b ? ldg32(pk_b + off + 16) : 0u;
      } else {
        const uint8_t* kr = kb + ks * 16 * kCodeRowBytes;
        const uint32_t x0 = nib2(lds32(kr), sh);
        const uint32_t x1 = nib2(lds32(kr + kCodeRowBytes), sh);
        const uint32_t x8 = nib2(lds32(kr + 8 * kCodeRowBytes), sh);
        const uint32_t x9 = nib2(lds32(kr + 9 * kCodeRowBytes), sh);
        const uint32_t s01 = lds32(scol + 32 * ks), s89 = lds32(scol + 32 * ks + 16);
        const uint32_t z01 = lds32(zp + 32 * ks), z89 = lds32(zp + 32 * ks + 16);
        a[e][0] = hfma2(sub_f16x2(prmt(x0, x1, kPrmtLo), kF16x2Magic), s01, z01);
        a[e][1] = hfma2(sub_f16x2(prmt(x0, x1, kPrmtHi), kF16x2Magic), s01, z01);
        a[e][2] = hfma2(sub_f16x2(prmt(x8, x9, kPrmtLo), kF16x2Magic), s89, z89);
        a[e][3] = hfma2(sub_f16x2(prmt(x8, x9, kPrmtHi), kF16x2Magic), s89, z89);
      }
    }
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
      // Matrix mi: rows 8n + 0..7, d chunk 4 ks2 + mi = (k-step mi >> 1,
      // half mi & 1); the Q rows are chunk-swizzled by row & 7.
      uint32_t qb[4];
      ldmatrix_x4(qb, q_smem + (8 * n + l8) * kQPitch +
                          (((4 * ks2 + mi) ^ l8) << 4));
  #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const uint32_t b[2] = {qb[2 * e], qb[2 * e + 1]};
        if constexpr (mode_qk16(kMode)) {
          uint32_t h[2] = {0u, 0u};
          mma_m16n8k16_f16f16(h, a[e], b);
          s[n][0] += lo_f(h[0]);
          s[n][1] += hi_f(h[0]);
          s[n][2] += lo_f(h[1]);
          s[n][3] += hi_f(h[1]);
        } else {
          mma_m16n8k16_f16f32(s[n], a[e], b);
        }
      }
    }
  }
}

// O^T[mt][n] += V^T[d of m-tile 2w + mt] . P^T[rows 8n .. 8n + 7].
template <int NT, int kMode, bool kPool>
PTH_DEVICE void pv(float (&o)[2][NT][4], const uint8_t* st, uint32_t p_smem,
                   const char* pv_base, int64_t pv_s_tok, int n_tok, int warp,
                   int lane) {
  const int g = lane >> 2, q4 = lane & 3, mi = lane >> 3, l8 = lane & 7;
  // Quantised: code words of tokens 2 q4 (+ 16 ks + {0, 1, 8, 9}) at word
  // u = 2w + mt; s_col_V of d = 16u + g and 16u + g + 8, as broadcast pairs.
  const uint8_t* vb = st + kVPacked + (2 * q4) * kCodeRowBytes +
                      (((warp >> 1) ^ q4) << 4);
  const int sh = 2 * g;
  uint32_t sc_lo[2], sc_hi[2];
  int woff[2];
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int u = 2 * warp + mt;
    woff[mt] = (u & 3) << 2;
    if constexpr (!kPool) {
      const uint32_t lo = lds16(st + kVSCol + 2 * (16 * u + g));
      const uint32_t hi = lds16(st + kVSCol + 2 * (16 * u + g + 8));
      sc_lo[mt] = lo | (lo << 16);
      sc_hi[mt] = hi | (hi << 16);
    }
  }
  [[maybe_unused]] uint32_t hacc[2][NT][2];
  if constexpr (mode_pv16(kMode)) {
  #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
  #pragma unroll
      for (int n = 0; n < NT; ++n) hacc[mt][n][0] = hacc[mt][n][1] = 0u;
  }
  #pragma unroll (kPool ? 1 : kPvUnroll)
  for (int ks2 = 0; ks2 < kBlockTokens / 32; ++ks2) {
    uint32_t a[2][2][4];  // [k-step e][m-tile][reg]
  #pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int ks = 2 * ks2 + e;
      const int t0 = 16 * ks + 2 * q4;
      if constexpr (kPool) {
        const char* r0 = pv_base + t0 * pv_s_tok;
        const char* r1 = r0 + pv_s_tok;
        const char* r8 = r0 + 8 * pv_s_tok;
        const char* r9 = r0 + 9 * pv_s_tok;
        const bool k0 = t0 < n_tok, k1 = t0 + 1 < n_tok, k8 = t0 + 8 < n_tok,
                   k9 = t0 + 9 < n_tok;
  #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const int dl = 2 * (16 * (2 * warp + mt) + g), dh = dl + 16;
          a[e][mt][0] = (k0 ? ldg16(r0 + dl) : 0u) | ((k1 ? ldg16(r1 + dl) : 0u) << 16);
          a[e][mt][1] = (k0 ? ldg16(r0 + dh) : 0u) | ((k1 ? ldg16(r1 + dh) : 0u) << 16);
          a[e][mt][2] = (k8 ? ldg16(r8 + dl) : 0u) | ((k9 ? ldg16(r9 + dl) : 0u) << 16);
          a[e][mt][3] = (k8 ? ldg16(r8 + dh) : 0u) | ((k9 ? ldg16(r9 + dh) : 0u) << 16);
        }
      } else {
        // Token pairs (t0, t0 + 1) and (t0 + 8, t0 + 9): s_row_V and zp_V
        // pairs, zeroed past the block's end so that a stale scale there can
        // never reach the MMA (P is 0 there, but 0 * NaN is NaN).
        const uint32_t m01 = (t0 < n_tok ? 0x0000FFFFu : 0u) |
                             (t0 + 1 < n_tok ? 0xFFFF0000u : 0u);
        const uint32_t m89 = (t0 + 8 < n_tok ? 0x0000FFFFu : 0u) |
                             (t0 + 9 < n_tok ? 0xFFFF0000u : 0u);
        const uint32_t sr01 = lds32(st + kVSRow + 2 * t0) & m01;
        const uint32_t sr89 = lds32(st + kVSRow + 2 * t0 + 16) & m89;
        const uint32_t z01 = lds32(st + kVZp + 2 * t0) & m01;
        const uint32_t z89 = lds32(st + kVZp + 2 * t0 + 16) & m89;
        const uint8_t* vr = vb + ks * 16 * kCodeRowBytes;
  #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const uint32_t y0 = crumb2(lds32(vr + woff[mt]), sh);
          const uint32_t y1 = crumb2(lds32(vr + kCodeRowBytes + woff[mt]), sh);
          const uint32_t y8 = crumb2(lds32(vr + 8 * kCodeRowBytes + woff[mt]), sh);
          const uint32_t y9 = crumb2(lds32(vr + 9 * kCodeRowBytes + woff[mt]), sh);
          a[e][mt][0] = hmul2(hfma2(sub_f16x2(prmt(y0, y1, kPrmtLo), kF16x2Magic), sr01, z01), sc_lo[mt]);
          a[e][mt][1] = hmul2(hfma2(sub_f16x2(prmt(y0, y1, kPrmtHi), kF16x2Magic), sr01, z01), sc_hi[mt]);
          a[e][mt][2] = hmul2(hfma2(sub_f16x2(prmt(y8, y9, kPrmtLo), kF16x2Magic), sr89, z89), sc_lo[mt]);
          a[e][mt][3] = hmul2(hfma2(sub_f16x2(prmt(y8, y9, kPrmtHi), kF16x2Magic), sr89, z89), sc_hi[mt]);
        }
      }
    }
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
      uint32_t pb[4];
      ldmatrix_x4(pb, p_smem + (8 * n + l8) * kPPitch + (4 * ks2 + mi) * kChunk);
  #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const uint32_t b[2] = {pb[2 * e], pb[2 * e + 1]};
  #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          if constexpr (mode_pv16(kMode)) {
            mma_m16n8k16_f16f16(hacc[mt][n], a[e][mt], b);
          } else {
            mma_m16n8k16_f16f32(o[mt][n], a[e][mt], b);
          }
        }
      }
    }
    if constexpr (mode_pv16(kMode)) {
      if (ks2 & 1) {  // flush every 64 tokens
  #pragma unroll
        for (int mt = 0; mt < 2; ++mt)
  #pragma unroll
          for (int n = 0; n < NT; ++n) {
            o[mt][n][0] += lo_f(hacc[mt][n][0]);
            o[mt][n][1] += hi_f(hacc[mt][n][0]);
            o[mt][n][2] += lo_f(hacc[mt][n][1]);
            o[mt][n][3] += hi_f(hacc[mt][n][1]);
            hacc[mt][n][0] = hacc[mt][n][1] = 0u;
          }
      }
    }
  }
}

template <int NT, int kMode>
__global__ void __launch_bounds__(kThreads, 1)
    kvarn_verify_kernel(const Params p) {
  using L = Smem<NT>;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int b = blockIdx.x, hk = blockIdx.y, split = blockIdx.z;
  const int qlen = p.qlen, group = p.group;
  const int rows_real = qlen * group;

  const int seq = max(p.vq_seqlen[b * qlen + qlen - 1], 0);
  const int n_blocks = (seq + kBlockTokens - 1) / kBlockTokens;
  const int per = (n_blocks + p.num_splits - 1) / p.num_splits;
  const int blk_lo = split * per;
  const int blk_hi = min(blk_lo + per, n_blocks);
  const int causal0 = seq - qlen + 1;  // the first token's limit

  const uint32_t smem_base = smem_u32(smem);
  const uint32_t q_smem = smem_base + L::kQ;
  const uint32_t p_smem = smem_base + L::kP;
  float* xchg = reinterpret_cast<float*>(smem + L::kXchg);
  int* lim_s = reinterpret_cast<int*>(smem + L::kLim);
  int* lo_s = reinterpret_cast<int*>(smem + L::kLo);
  const int32_t* bt = p.block_table + static_cast<int64_t>(b) * p.bt_s_row;

  auto slot_of = [&](int32_t bid) -> int32_t {
    return (bid >= 0 && bid < p.lookup) ? p.block_to_slot[bid] : -1;
  };
  auto tile_of = [&](int32_t bid) -> const uint8_t* {
    return p.kv + static_cast<int64_t>(bid) * p.kv_s_block +
           static_cast<int64_t>(hk) * p.kv_s_head;
  };

  float o[2][NT][4];
  float m[NT][2], l[NT][2];
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
  #pragma unroll
    for (int i = 0; i < 2; ++i) {
      m[n][i] = -INFINITY;
      l[n][i] = 0.f;
    }
  #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
      o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
  }

  int32_t cur_bid = 0, cur_slot = -1;
  if (blk_lo < blk_hi) {
    // Prologue: row limits, Q (zero rows past QLEN G), the first tile.
    for (int r = tid; r < L::kRows; r += kThreads) {
      const int lim = r < rows_real ? causal0 + r / group : 0;
      lim_s[r] = lim;
      lo_s[r] = p.window > 0 ? max(lim - p.window, 0) : 0;
    }
    for (int c = tid; c < L::kRows * (kQPitch / kChunk); c += kThreads) {
      const int r = c >> 5, ch = c & 31;
      const bool valid = r < rows_real;
      const char* src =
          valid ? p.q + static_cast<int64_t>(b * qlen + r / group) * p.q_s_tok +
                      static_cast<int64_t>(hk * group + r % group) * p.q_s_head +
                      ch * kChunk
                : p.q;
      cp_async_zfill<16>(q_smem + r * kQPitch + ((ch ^ (r & 7)) << 4), src,
                         valid ? 16u : 0u);
    }
    cur_bid = bt[blk_lo];
    cur_slot = slot_of(cur_bid);
    if (cur_slot < 0) issue_tile(smem_base, tile_of(cur_bid), tid);
    cp_async_commit();
  }

  const int ta = qk_token_a(warp, g), tb = ta + 4;
  for (int k = blk_lo; k < blk_hi; ++k) {
    const int stage = (k - blk_lo) & 1;
    cp_async_wait<0>();
    __syncthreads();  // barrier 0: this tile and Q landed; the other stage, P
                      // and the exchange are free
    int32_t nxt_bid = 0, nxt_slot = -1;
    if (k + 1 < blk_hi) {
      nxt_bid = bt[k + 1];
      nxt_slot = slot_of(nxt_bid);
      if (nxt_slot < 0)
        issue_tile(smem_base + (stage ^ 1) * L::kStage, tile_of(nxt_bid), tid);
    }
    cp_async_commit();

    const uint8_t* st = smem + stage * L::kStage;
    const bool pooled = cur_slot >= 0;
    const int n_tok = min(seq - k * kBlockTokens, kBlockTokens);
    const bool fast = p.window == 0 && (k + 1) * kBlockTokens <= causal0;
    const char* pool_k = nullptr;
    const char* pool_v = nullptr;
    if (pooled) {
      pool_k = p.pool_k + static_cast<int64_t>(cur_slot) * p.pk_s_slot +
               static_cast<int64_t>(hk) * p.pk_s_head;
      pool_v = p.pool_v + static_cast<int64_t>(cur_slot) * p.pv_s_slot +
               static_cast<int64_t>(hk) * p.pv_s_head;
    }

    // 1 QK.
    float s[NT][4];
    float fa, fb;
    if (pooled) {
      qk<NT, kMode, true>(s, st, q_smem, pool_k + ta * p.pk_s_tok,
                          pool_k + tb * p.pk_s_tok, ta < n_tok, tb < n_tok,
                          warp, lane);
      fa = fb = p.score_scale_log2;
    } else {
      qk<NT, kMode, false>(s, st, q_smem, nullptr, nullptr, true, true, warp,
                           lane);
      fa = h2f(lds16(st + kKSRow + 2 * ta)) * p.score_scale_log2;
      fb = h2f(lds16(st + kKSRow + 2 * tb)) * p.score_scale_log2;
    }

    // 2 scale, mask (selects: a stale s_row past the end may be NaN), max.
    const int pos_a = k * kBlockTokens + ta, pos_b = pos_a + 4;
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
  #pragma unroll
      for (int i = 0; i < 2; ++i) {
        float va = s[n][i] * fa, vb = s[n][2 + i] * fb;
        if (!fast) {
          const int row = 8 * n + 2 * q4 + i;
          const int lim = lim_s[row], lo = lo_s[row];
          va = (pos_a < lim && pos_a >= lo) ? va : -INFINITY;
          vb = (pos_b < lim && pos_b >= lo) ? vb : -INFINITY;
        }
        s[n][i] = va;
        s[n][2 + i] = vb;
        float mx = fmaxf(va, vb);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
        if (g == 0) xchg[(8 * n + 2 * q4 + i) * kWarps + warp] = mx;
      }
    }
    __syncthreads();  // barrier A

    // 3 online softmax, O rescale, P to shared memory.
    uint8_t* pbuf = smem + L::kP;
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
  #pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int row = 8 * n + 2 * q4 + i;
        const float4 x0 = *reinterpret_cast<const float4*>(xchg + row * kWarps);
        const float4 x1 = *reinterpret_cast<const float4*>(xchg + row * kWarps + 4);
        const float tmax = fmaxf(fmaxf(fmaxf(x0.x, x0.y), fmaxf(x0.z, x0.w)),
                                 fmaxf(fmaxf(x1.x, x1.y), fmaxf(x1.z, x1.w)));
        const float m_new = fmaxf(m[n][i], tmax);
        // A row with no live key yet keeps m = -inf: a finite stand-in keeps
        // alpha and p at exactly 0 instead of NaN.
        const float m_use = m_new == -INFINITY ? 0.f : m_new;
        const float alpha = ex2(m[n][i] - m_use);
        const float pa = ex2(s[n][i] - m_use), pb = ex2(s[n][2 + i] - m_use);
        l[n][i] = l[n][i] * alpha + pa + pb;
        m[n][i] = m_new;
  #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          o[mt][n][i] *= alpha;
          o[mt][n][2 + i] *= alpha;
        }
        *reinterpret_cast<uint16_t*>(pbuf + row * kPPitch + 2 * ta) = f2h(pa);
        *reinterpret_cast<uint16_t*>(pbuf + row * kPPitch + 2 * tb) = f2h(pb);
      }
    }
    __syncthreads();  // barrier B

    // 4 PV.
    if (pooled) {
      pv<NT, kMode, true>(o, st, p_smem, pool_v, p.pv_s_tok, n_tok, warp, lane);
    } else {
      pv<NT, kMode, false>(o, st, p_smem, nullptr, 0, n_tok, warp, lane);
    }
    cur_bid = nxt_bid;
    cur_slot = nxt_slot;
  }

  // Epilogue: l over lanes and warps (every thread passed the last barrier
  // B, so the exchange is free), then the rows' outputs.
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
  #pragma unroll
    for (int i = 0; i < 2; ++i) {
      float v = l[n][i];
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      if (g == 0) xchg[(8 * n + 2 * q4 + i) * kWarps + warp] = v;
    }
  }
  __syncthreads();
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
  #pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int row = 8 * n + 2 * q4 + i;
      if (row >= rows_real) continue;
      const float4 x0 = *reinterpret_cast<const float4*>(xchg + row * kWarps);
      const float4 x1 = *reinterpret_cast<const float4*>(xchg + row * kWarps + 4);
      const float lsum = ((x0.x + x0.y) + (x0.z + x0.w)) + ((x1.x + x1.y) + (x1.z + x1.w));
      const bool live = lsum > 0.f;
      const float inv = live ? 1.f / lsum : 0.f;
      const int64_t tok = static_cast<int64_t>(b) * qlen + row / group;
      const int head = hk * group + row % group;
      if (p.num_splits == 1) {
        char* orow = p.out + tok * p.out_s_tok + static_cast<int64_t>(head) * p.out_s_head;
  #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const int d = 16 * (2 * warp + mt) + g;
          *reinterpret_cast<uint16_t*>(orow + 2 * d) = f2h(o[mt][n][i] * inv);
          *reinterpret_cast<uint16_t*>(orow + 2 * (d + 8)) = f2h(o[mt][n][2 + i] * inv);
        }
      } else {
        if (live) {
          char* wrow = reinterpret_cast<char*>(p.ws_o) + split * p.wo_s_split +
                       tok * p.wo_s_tok + static_cast<int64_t>(head) * p.wo_s_head;
  #pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            const int d = 16 * (2 * warp + mt) + g;
            *reinterpret_cast<float*>(wrow + 4 * d) = o[mt][n][i] * inv;
            *reinterpret_cast<float*>(wrow + 4 * (d + 8)) = o[mt][n][2 + i] * inv;
          }
        }
        if (warp == 0 && g == 0) {
          *reinterpret_cast<float*>(reinterpret_cast<char*>(p.ws_lse) +
                                    split * p.wl_s_split + tok * p.wl_s_tok +
                                    static_cast<int64_t>(head) * p.wl_s_head) =
              live ? m[n][i] + log2f(lsum) : -INFINITY;
        }
      }
    }
  }
}

// One instantiation's launch. The shared-memory opt-in is a function-local
// static per instantiation, as attn_decode.cuh's launchers (setup.py:
// -fno-gnu-unique keeps it per DSO).
template <int NT, int kMode>
cudaError_t launch_split_kernel(const Params& p, cudaStream_t stream) {
  static const cudaError_t attr = cudaFuncSetAttribute(
      kvarn_verify_kernel<NT, kMode>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      Smem<NT>::kBytes);
  if (attr != cudaSuccess) return attr;
  const dim3 grid(static_cast<unsigned>(p.num_reqs),
                  static_cast<unsigned>(p.num_kv_heads),
                  static_cast<unsigned>(p.num_splits));
  kvarn_verify_kernel<NT, kMode><<<grid, kThreads, Smem<NT>::kBytes, stream>>>(p);
  return cudaGetLastError();
}

#endif  // __CUDACC__

}  // namespace kvarn
}  // namespace pth_attn
