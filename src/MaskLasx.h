/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>

namespace fbgemm {

namespace internal {

// clang-format off
alignas(64) static const int lasx_ps_or_epi32_masks[9][8] = {
  // NOTE: clang-format wants to use a different formatting but the current
  // formatting should be easier to read.
  {  0,  0,  0,  0,  0,  0,  0,  0,  },
  { -1,  0,  0,  0,  0,  0,  0,  0,  },
  { -1, -1,  0,  0,  0,  0,  0,  0,  },
  { -1, -1, -1,  0,  0,  0,  0,  0,  },
  { -1, -1, -1, -1,  0,  0,  0,  0,  },
  { -1, -1, -1, -1, -1,  0,  0,  0,  },
  { -1, -1, -1, -1, -1, -1,  0,  0,  },
  { -1, -1, -1, -1, -1, -1, -1,  0,  },
  { -1, -1, -1, -1, -1, -1, -1, -1,  },
};

static const int lasx_ps_or_epi32_combined_mask[16] = {
  -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
};

// clang-format on

} // namespace internal

} // namespace fbgemm
