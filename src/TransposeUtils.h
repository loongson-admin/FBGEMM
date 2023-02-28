/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "fbgemm/FbgemmBuild.h"

#include <cstdint>

namespace fbgemm {

/**
 * @brief Reference implementation of matrix transposition: B = A^T.
 * @param M The height of the matrix.
 * @param N The width of the matrix.
 * @param src The memory buffer of the source matrix A.
 * @param ld_src The leading dimension of the source matrix A.
 * @param dst The memory buffer of the destination matrix B.
 * @param ld_dst The leading dimension of the destination matrix B.
 */
template <typename T>
FBGEMM_API void transpose_ref(
    unsigned M,
    unsigned N,
    const T* src,
    unsigned ld_src,
    T* dst,
    unsigned ld_dst);

namespace internal {

/**
 * @brief Transpose a matrix using LASX.
 *
 * This is called if the code is running on a CPU with LASX support.
 */
template <typename T>
void transpose_lasx(
    unsigned M,
    unsigned N,
    const T* src,
    unsigned ld_src,
    T* dst,
    unsigned ld_dst);

} // namespace internal

} // namespace fbgemm
