// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// Instantiations of the decode kernel's attn-1 MMA modes at D = 256
// (docs/research/deepinfer/attention-kv.md, attn-1; attn_decode.cuh DecMma),
// for every n-tile count 1..6, both q dtypes and both loaders. Reached only
// when the launch's AttnLaunch::mma is non-zero (attn_launch.h
// set_attn1_mode); the shipped kernels are instantiate_d256.cu's. A separate
// translation unit so that the shipped one compiles as before and the two
// compile in parallel.

#include "attn_decode.cuh"

namespace pth_attn {

cudaError_t launch_attn_d256_attn1(const AttnLaunch& L, cudaStream_t stream) {
  switch (L.mma) {
    case kDecMmaPv16:
      return launch_dec<256, kDecMmaPv16>(L, stream);
    case kDecMmaS8Pv16:
      return launch_dec<256, kDecMmaS8Pv16>(L, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace pth_attn
