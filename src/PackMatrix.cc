/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS

#include <cpuinfo.h>
#include <iomanip>
#include <stdexcept>
#include <type_traits>
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename PT, typename inpType, typename accType>
PackMatrix<PT, inpType, accType>::PackMatrix(
    int32_t rows,
    int32_t cols,
    inpType* buf,
    int groups,
    const BlockingFactors* params)
    : buf_(buf), nrows_(rows), ncols_(cols), G_(groups) {
  bufAllocatedHere_ = false;
  blocking_params = params;
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

template <typename PT, typename inpType, typename accType>
int PackMatrix<PT, inpType, accType>::packedBufferSize(
    int rows,
    int cols,
    const BlockingFactors* params) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if (!fbgemmHasLasxSupport()) {
    assert(0 && "unknown architecure");
  }

  int MCB, KCB, NCB;
  if (params) {
    MCB = params->MCB;
    NCB = params->NCB;
    KCB = params->KCB;
  } else {
    const inst_set_t isa = fbgemmInstructionSet();
    switch (isa) {
      case inst_set_t::lasx:
        MCB = PackingTraits<inpType, accType, inst_set_t::lasx>::MCB;
        NCB = PackingTraits<inpType, accType, inst_set_t::lasx>::NCB;
        KCB = PackingTraits<inpType, accType, inst_set_t::lasx>::KCB;
        break;

      default:
        assert(0 && "unknown architecure");
        throw std::runtime_error("unknown architecure");
    }
  }

  if (isA()) {
    return MCB * KCB;
  } else {
    int rowBlock = KCB;
    int colBlock = NCB;
    return (((rows + rowBlock - 1) / rowBlock) * rowBlock) *
        (((cols + colBlock - 1) / colBlock) * colBlock);
  }

  return -1;
}

// int32 accumulation
template class PackMatrix<PackAMatrix<uint8_t, int32_t>, uint8_t, int32_t>;

template class PackMatrix<
    PackAWithRowOffset<uint8_t, int32_t>,
    uint8_t,
    int32_t>;

template class PackMatrix<
    PackAWithIm2Col<uint8_t, int32_t, 1>,
    uint8_t,
    int32_t>;
template class PackMatrix<PackAWithIm2Col<uint8_t, int32_t>, uint8_t, int32_t>;
template class PackMatrix<
    PackAWithIm2Col<uint8_t, int32_t, 3>,
    uint8_t,
    int32_t>;

template class PackMatrix<
    PackAWithQuantRowOffset<uint8_t, int32_t>,
    uint8_t,
    int32_t>;

template class PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>;

// int16 accumulation
template class PackMatrix<
    PackAWithIm2Col<uint8_t, int16_t, 1>,
    uint8_t,
    int16_t>;
template class PackMatrix<PackAWithIm2Col<uint8_t, int16_t>, uint8_t, int16_t>;
template class PackMatrix<
    PackAWithIm2Col<uint8_t, int16_t, 3>,
    uint8_t,
    int16_t>;

template class PackMatrix<
    PackAWithRowOffset<uint8_t, int16_t>,
    uint8_t,
    int16_t>;

template class PackMatrix<PackAMatrix<uint8_t, int16_t>, uint8_t, int16_t>;

template class PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>;
} // namespace fbgemm
