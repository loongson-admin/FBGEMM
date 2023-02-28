/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <array>
#include <cmath>
#include <utility>

#include "./FbgemmFP16UKernelsLasx.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmFPCommon.h"

namespace fbgemm {

namespace {
// optimized kernels to cover all cases
// 2 in ?x2 should be the same as kernel_ncol_blocks.
// Here with kernel_ncol_blocks = 2, we can provide up to 6x2 kernels, due to
// the restrictions of ymm register numbers (16).

constexpr kernel_array_t<float16> kernel_fp16_lasx = {
    nullptr,
    gemmkernel_1x2_Lasx_fp16_fA0fB0fC0,
    gemmkernel_2x2_Lasx_fp16_fA0fB0fC0,
    gemmkernel_3x2_Lasx_fp16_fA0fB0fC0,
    gemmkernel_4x2_Lasx_fp16_fA0fB0fC0,
    gemmkernel_5x2_Lasx_fp16_fA0fB0fC0,
    gemmkernel_6x2_Lasx_fp16_fA0fB0fC0 };

} // namespace

template <>
const isa_descriptor<float16>& getIsaHandlers(inst_set_t isa, float16) {
  static isa_descriptor<float16> lasx_descriptor =
      std::make_tuple(kernel_fp16_lasx, partition_lasx);

  switch (isa) {
    case inst_set_t::anyarch:
    case inst_set_t::lasx:
      return lasx_descriptor;
    default:
      break;
  }

  throw std::runtime_error("Unsupported uArch");
}

#ifdef FBGEMM_FP16_FALLBACK_TO_REF_KERNEL
template <>
FBGEMM_API void ref_kernel<float16>(
    int kernel_nrows,
    GemmParams<float16>* gp,
    const float* C_base,
    int m_total,
    int n_total,
    int simd_len) {
  int kernel_ncol_blocks = 2;
  int block_col_size = simd_len * kernel_ncol_blocks;
  for (int jb = 0; jb < gp->b_block_cols; ++jb) {
    for (int k = 0; k < gp->k; ++k) {
      for (int i = 0; i < kernel_nrows; ++i) {
        float a = gp->A[i + k * kernel_nrows];
        for (int j = 0; j < block_col_size; ++j) {
          float* C_ptr =
              gp->C + i * (gp->ldc / sizeof(float)) + jb * block_col_size + j;
          assert(C_ptr < C_base + m_total * n_total);
          float b =
              cpu_half2float(gp->B[(jb * gp->k + k) * block_col_size + j]);
          if (k == 0) {
            if (gp->beta) {
              *C_ptr = std::fma(a, b, (gp->beta) * (*C_ptr));
            } else {
              *C_ptr = a * b;
            }
          } else {
            *C_ptr = std::fma(a, b, *C_ptr);
          }
        }
      }
    }
  }
}
#endif // FBGEMM_FP16_FALLBACK_TO_REF_KERNEL

template FBGEMM_API void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixB<float16>& Bp,
    const float beta,
    float* C,
    int thread_id,
    int num_threads);

} // namespace fbgemm
