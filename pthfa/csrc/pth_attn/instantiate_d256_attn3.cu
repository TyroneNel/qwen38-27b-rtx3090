// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Instantiations of attn-3's int8-QK 64-row-tile kernel at D = 256
// (docs/research/deepinfer/attention-kv.md, attn-3; attn_prefill_s8.cuh):
// WM = 4, both q dtypes and both loaders, causal. Reached only when the
// launch's AttnLaunch::prefill_s8 is set (attn_launch.h kAttn3PrefillS8); the
// shipped kernels are instantiate_d256.cu's. A separate translation unit so
// that the shipped one compiles as before and the two compile in parallel.

#include "attn_prefill_s8.cuh"

namespace pth_attn {

cudaError_t launch_attn_d256_attn3(const AttnLaunch& L, cudaStream_t stream) {
  if (L.wm == 4) return launch_attn_s8_wm<256, 4>(L, stream);
  return cudaErrorInvalidValue;
}

}  // namespace pth_attn
