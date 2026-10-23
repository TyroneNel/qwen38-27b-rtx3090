// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_launch.h: the host launcher of the attention kernels (attn_launch.cu)
// and of the split combine (attn_combine.cu).
//
// The op (launch.cu) and the C ABI of the test library (attn_capi.cu, driven
// by the kernel test suite through ctypes) both launch through these
// functions, so the op runs exactly the code the tests prove.
//
// A launcher checks its arguments (every check names the field), turns them
// into the kernel parameters, picks the instantiation and enqueues it on
// `stream`. It allocates nothing, reads nothing back from the device and does
// not synchronise. It returns PTH_ATTN_OK, PTH_ATTN_EINVAL (an argument check
// failed) or PTH_ATTN_ECUDA (the launch failed); the message of a failure is
// last_launch_error() on the calling thread.
//
// Requirements beyond the semantics of attn_args.h:
// - wm is derived: the smallest wm in {1, 2, 4} with bq * G <= 16 wm
//   (= the planner's wm for the planner's bq); bq * G > 64 is
//   EINVAL.
// - Alignment: q, out and the workspace need 16-B aligned base pointers and
//   outer byte strides (16-B vector loads/stores); the four cache streams
//   need 4-B alignment (the narrowest cp.async). EINVAL otherwise.
// - window > 0 or causal == 0 (the windowed kernels) needs
//   head_size 128 and the separate-stream loader; EINVAL otherwise.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "attn_args.h"

namespace pth_attn {

// KV loader of launch_attention: the shipped choice (the whole-cell loader
// wherever it applies), or one forced by a test: separate streams, or the
// whole cell (D = 256 with the inline cell layout only; EINVAL otherwise).
constexpr int32_t kLoaderDefault = 0;
constexpr int32_t kLoaderSeparate = 1;
constexpr int32_t kLoaderCell = 2;

int32_t launch_attention(const PthAttnArgs& args, int32_t loader,
                         cudaStream_t stream);
int32_t launch_combine(const PthAttnCombineArgs& args, cudaStream_t stream);

// The calling thread's message of the last failed launch ("" after a
// successful one). Valid until the next launch on the same thread.
const char* last_launch_error();
// Records `msg` as the calling thread's last error and returns `code`.
int32_t launch_fail(int32_t code, const std::string& msg);
// Clears the last error; every launcher and C entry point starts with it.
void clear_launch_error();

// Head sizes with a kernel instantiation (instantiate_d{64,128,256}.cu).
inline bool is_head_size(int32_t d) { return d == 64 || d == 128 || d == 256; }

// attn-1 experiments (docs/research/deepinfer/attention-kv.md, attn-1): a
// process-wide bit set read by every launch_attention, 0 (the default) = the
// shipped kernels. Set by the op library's C entry point
// pth_flash_attn_set_attn1_mode (launch.cu), which pth_flash_attn.ops calls
// from its environment knobs. Bits:
//   kAttn1Tile48  D = 256: row tiles of 33..48 rows on the decode kernel
//                 (NT = 5, 6) instead of the padded 64-row tile (stage 1).
//   kAttn1Mma*    bits 1-2, the decode kernel's MMA mode (attn_decode.cuh
//                 DecMma): 0 fp32 accumulate (shipped), 1 PV with fp16
//                 accumulate (stage 2), 2 = 1 plus QK on int8 (stage 3).
//                 Only launches on the decode kernel (D = 256, <= 32 rows,
//                 or <= 48 with kAttn1Tile48) use it.
constexpr int32_t kAttn1Tile48 = 1;
constexpr int32_t kAttn1MmaShift = 1;
constexpr int32_t kAttn1MmaMask = 3 << kAttn1MmaShift;
constexpr int32_t kAttn1MmaModes = 3;  // DecMma values 0 .. kAttn1MmaModes - 1
//   kAttn3PrefillS8  bit 3 (attn-3, lossy): QK on int8 in the 64-row-tile
//                 kernel at D = 256, causal, no window (the prefill launches
//                 of the target's full-attention layers): each Q row
//                 quantised by its absmax, the cache's int8 K fed straight
//                 into m16n8k32.s8 (attn_prefill_s8.cuh). PV is unchanged.
constexpr int32_t kAttn3PrefillS8 = 1 << 3;
constexpr int32_t kAttn1KnownBits = kAttn1Tile48 | kAttn1MmaMask | kAttn3PrefillS8;
// Returns the previous mode; a mode with unknown bits or an MMA mode the
// library does not have is refused (-1, mode unchanged).
int32_t set_attn1_mode(int32_t mode);
int32_t attn1_mode();

}  // namespace pth_attn

// Returns PTH_ATTN_EINVAL with `msg` from the enclosing launcher unless
// `cond` holds.
#define PTH_ATTN_REQUIRE(cond, msg)                                    \
  do {                                                                 \
    if (!(cond)) return pth_attn::launch_fail(PTH_ATTN_EINVAL, (msg)); \
  } while (0)
