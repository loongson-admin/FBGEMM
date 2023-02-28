/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmI64.h"

#include "x86ToLoongArchIntrin.h"
#include <cmath>
#include <iostream>
#include <vector>

#include "./GenerateKernel.h"
#include "./RefImplementations.h"
#include "fbgemm/PackingTraits-inl.h"

using namespace std;

namespace fbgemm {

// Expected to have overflows
NO_SANITIZE("undefined")
void cblas_gemm_i64_i64acc(
    matrix_op_t transa,
    matrix_op_t transb,
    int M,
    int N,
    int K,
    const int64_t* A,
    int lda,
    const int64_t* B,
    int ldb,
    bool accumulate,
    int64_t* C,
    int ldc) {
  cpuinfo_initialize();
  {  //Go ref when no 512 support
    cblas_gemm_i64_i64acc_ref(
        transa, transb, M, N, K, A, lda, B, ldb, accumulate, C, ldc);
    return;
  }
}

} // namespace fbgemm
