/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "x86ToLoongArchIntrin.h"
#include "fbgemm/FbgemmConvert.h"

namespace fbgemm {

namespace {

inline void Float16ToFloatKernelLasx(const float16* src, float* dst) {
  __m128i half_vector = _mm_loadu_si128((__m128i*)src);
  __m256 float_vector = _mm256_cvtph_ps(half_vector);
  _mm256_storeu_ps(dst, float_vector);
}

} // namespace

void Float16ToFloat_lasx(const float16* src, float* dst, size_t size) {
  size_t i = 0;
  for (i = 0; i + 8 <= size; i += 8) {
    Float16ToFloatKernelLasx(src + i, dst + i);
  }
  Float16ToFloat_ref(src + i, dst + i, size - i);
}

} // namespace fbgemm
