/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "./GenerateKernel.h"

namespace fbgemm {

namespace la64 = asmjit::la64;

/**
 * Generate instructions for initializing the C registers to 0 in 32-bit
 * Accumulation kernel.
 */

void initCRegs(la64::Emitter* a, int rowRegs, int colRegs) {
  using CRegs = la64::VecX;
  // Take advantage of implicit zeroing out
  // i.e., zero out xmm and ymm will be zeroed out too
  for (int i = 0; i < rowRegs; ++i) {
    for (int j = 0; j < colRegs; ++j) {
      a->xvxor_v(
          CRegs(i * colRegs + j),
          CRegs(i * colRegs + j),
          CRegs(i * colRegs + j));
    }
  }
}

} // namespace fbgemm
