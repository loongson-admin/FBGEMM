/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/FbgemmEmbedding.h"

#include <cassert>
#include <cmath>

#include "fbgemm/Types.h"

namespace fbgemm {
namespace internal {

template <typename InType, typename IndexType, typename OffsetType>
bool EmbeddingSpMDMBlockSize1_(
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size, // the number of rows in input
    const InType* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional,
    bool use_offsets) {
  int64_t current = 0;
  for (int m = 0; m < output_size; ++m) {
    out[m] = 0;
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }
    int i = 0;

    for (; i < len; ++i) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      float w = 1.f;
      if (weights) {
        w = weights[is_weight_positional ? i : current];
      }

      const InType* inptr = input + indices[current];
      out[m] = std::fma(
          w,
          std::is_same<InType, float16>::value ? cpu_half2float(*inptr)
                                               : *inptr,
          out[m]);

      ++current;
    }
    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
      out[m] *= scale;
    }
  }
  return current == index_size;
}

#define INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE) \
  template bool EmbeddingSpMDMBlockSize1_(                       \
      const std::int64_t output_size,                            \
      const std::int64_t index_size,                             \
      const std::int64_t data_size,                              \
      const IN_TYPE* input,                                      \
      const INDEX_TYPE* indices,                                 \
      const OFFSET_TYPE* offsets_or_lengths,                     \
      const float* weights,                                      \
      bool normalize_by_lengths,                                 \
      float* out,                                                \
      bool is_weight_positional,                                 \
      bool use_offsets);

#define INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, INDEX_TYPE)     \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, std::int64_t)

#define INSTANTIATE_SPMDM_INDEX_T(IN_TYPE)          \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int64_t)

INSTANTIATE_SPMDM_INDEX_T(float)
INSTANTIATE_SPMDM_INDEX_T(float16)
INSTANTIATE_SPMDM_INDEX_T(std::uint8_t)

#undef INSTANTIATE_SPMDM_INDEX_T
#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_BASE

} // namespace internal
} // namespace fbgemm
