/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include "./CodeCache.h"
#include "fbgemm/Fbgemm.h"
//#define FBGEMM_LOG_CODE 1

namespace fbgemm {

namespace la64 = asmjit::la64;

/**
 * @brief Generate instructions for initializing the C registers to 0.
 */
void initCRegs(la64::Emitter* a, int rowRegs, int colRegs);

/**
 * @tparam TA Type of matrix A.
 * @tparam TB Type of matrix B.
 * @tparam TC Type of matrix C.
 * @tparam accT Accumulation type, currently we support 16-bit (std::int16_t) or
 * 32-bit (std::int32_t) accumulation.
 */
template <typename TA, typename TB, typename TC, typename accT>
class CodeGenBase {
 public:
  using jit_micro_kernel_fp = void (*)(
      const TA* bufferA,
      const TB* bufferB,
      const TB* b_pf,
      TC* bufferC,
      int kc,
      int ldc);

  /**
   * @brief Constructor for initializing AVX2/AVX512 registers.
   */
  CodeGenBase(const BlockingFactors* params = nullptr)
      : blocking_params(params) {}

  /**
   * @brief Get or Create the instructions for macro-kernel.
   *
   * If the problem size (mc, nc) and accumulation flag (accum) can be found in
   * the code cache (a hash map), then get the macro-kernel instructions
   * directly from it. Otherwise, create the instructions for macro-kernel, and
   * store that into the code cache.
   */
  template <inst_set_t instSet>
  jit_micro_kernel_fp
  getOrCreate(bool accum, int32_t mc, int32_t nc, int32_t kc);

  /**
   * @brief Generate instructions for computing block in the rank-k update.
   */
  template <inst_set_t instSet>
  void genComputeBlock(
      la64::Emitter* a,
      la64::Gp buffer_A,
      la64::Gp buffer_B,
      la64::Gp B_pf,
      int rowRegs,
      int colRegs,
      int lda);

  /**
   * @brief Generate instructions for storing the C registers back to the
   * memory.
   */
  template <inst_set_t instSet>
  void storeCRegs(
    la64::Emitter* a,
    int rowRegs,
    int colRegs,
    la64::Gp C_Offset,
    la64::Gp ldcReg,
    bool accum);

  const BlockingFactors* blocking_params;
  /**
   * @brief Generate filename to dump generated code
   * (debug-only)
   */
  template <inst_set_t instSet>
  static std::string getCodeLoggingFile(
      bool accum,
      int mc,
      int nc,
      int NCB,
      int KCB,
      int MR,
      int NR) {
    std::ostringstream oss;
    oss << "gemm_";
    if (std::is_same<accT, std::int16_t>::value) {
      oss << "acc16_";
    } else if (std::is_same<accT, std::int32_t>::value) {
      oss << "acc32_";
    } else {
      oss << "unknown_";
    }
    oss << "accum-" + std::to_string(accum) << "_MC-" + std::to_string(mc)
        << "_NC-" + std::to_string(nc) << "_NCB-" + std::to_string(NCB)
        << "_KCB-" + std::to_string(KCB) << "_MR-" + std::to_string(MR)
        << "_NR-" + std::to_string(NR);
    if (instSet == inst_set_t::lasx) {
      oss << "_lasx";
    }
    oss << ".txt";
    return oss.str();
  }

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                  // depents on other static
                                  // variables.  Required to prevent
                                  // initialization order fiasco
    return rt;
  }

  static std::mutex rtMutex_; ///< Controll access to runtime;

  // The hash depends on accumulate, mc, nc, ncb, kcb, nr, mr
  static CodeCache<
      std::tuple<bool, int, int, int, int, int, int>,
      jit_micro_kernel_fp>
      codeCache_; ///< JIT Code Cache for reuse.
};

template <typename TA, typename TB, typename TC, typename accT>
std::mutex CodeGenBase<TA, TB, TC, accT>::rtMutex_;

template <typename TA, typename TB, typename TC, typename accT>
CodeCache<
    std::tuple<bool, int, int, int, int, int, int>,
    typename CodeGenBase<TA, TB, TC, accT>::jit_micro_kernel_fp>
    CodeGenBase<TA, TB, TC, accT>::codeCache_;

} // namespace fbgemm
