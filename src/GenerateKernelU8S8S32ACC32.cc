/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include "./CodeGenHelpers.h"
#include "./GenerateKernel.h"

namespace fbgemm {

namespace la64 = asmjit::la64;

/**
 * Generate LASX instructions for computing block in the rank-k update of
 * 32-bit Accumulation kernel.
 */
template <>
template <inst_set_t instSet>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::genComputeBlock(
    la64::Emitter* a,
    la64::Gp buffer_A,
    la64::Gp buffer_B,
    la64::Gp B_pf,
    int rowRegs,
    int colRegs,
    int lda) {
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;

  // used for matrix A
  VecRegT AReg(numRegs - 1);

  // used for matrix B
  VecRegT BReg(numRegs - 2);

  // Contains 16-bit 1s
  VecRegT oneReg(numRegs - 3);

  // temporary register
  VecRegT res1(numRegs - 4);

  VecRegT tmpReg1(numRegs - 5);

  la64::Gp tempGPX = la64::s4;
  la64::Gp temp_buffer_A = la64::s5;

  for (int j = 0; j < colRegs; ++j) {
    // load B
    emitLoadDWord<instSet, VecRegT>(
        // a, BReg, x86::dword_ptr(buffer_B, j * vectorLen * sizeof(int8_t)));
        a, BReg, ptr(buffer_B, j * vectorLen * sizeof(int8_t)));
    // load A, broadcast and fmas
    for (int i = 0; i < rowRegs; ++i) {

      mov_imm(a, temp_buffer_A, (i * lda) * sizeof(uint8_t));
      a->add_d(temp_buffer_A, buffer_A, temp_buffer_A);
      a->ld_d(tempGPX, ptr(temp_buffer_A));
      a->xvreplgr2vr_w(AReg, tempGPX);
      a->xvmulwev_h_bu_b (res1, AReg, BReg);
      a->xvmaddwod_h_bu_b(res1, AReg, BReg);
      a->xvmulwev_w_h (tmpReg1, oneReg, res1);
      a->xvmaddwod_w_h(tmpReg1, oneReg, res1);
      a->xvadd_w(VecRegT(i * colRegs + j), tmpReg1, VecRegT(i * colRegs + j));
    }
    a->preld(0, B_pf, j * vectorLen * sizeof(int8_t));
  }
}

/**
 * Generate LASX instructions for storing the C registers back to the memory
 * in 32-bit Accumulation kernel.
 */
template <>
template <inst_set_t instSet>
void CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::storeCRegs(
    la64::Emitter* a,
    int rowRegs,
    int colRegs,
    la64::Gp C_Offset,
    la64::Gp ldcReg,
    bool accum) {
  using VecT = typename simd_info<instSet>::vec_reg_t;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;
  la64::VecX tmpReg31 = la64::xr31;
  la64::Gp tempGPX = la64::t7;

  for (int i = 0; i < rowRegs; ++i) {
    if (i != 0) {
      a->add_d(C_Offset, C_Offset, ldcReg);
    } else {
      a->xor_(C_Offset, C_Offset, C_Offset);
    }
    for (int j = 0; j < colRegs; ++j) {
      a->addi_d(tempGPX, C_Offset, j * vectorLen * sizeof(int8_t));
      a->add_d(tempGPX, tempGPX, la64::a3);
      if (accum) {
        a->xvld(tmpReg31, ptr(tempGPX));
        a->xvadd_w(VecT(i * colRegs + j), VecT(i * colRegs + j), tmpReg31);
      }
      a->xvst(VecT(i * colRegs + j), ptr(tempGPX));
    }
  }
}

/**
 * Get or Create the LASX instructions for 32-bit Accumulation macro-kernel.
 *
 */
template <>
template <inst_set_t instSet>
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::getOrCreate(
    bool accum,
    int32_t mc,
    int32_t nc,
    int32_t kc) {
  (void)kc; // Suppress unused variable warning
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  std::tuple<bool, int, int, int, int, int, int> kernelSig;
  int kBlock;
  int nBlock;
  int mRegBlockSize;
  int nRegBlockSize;
  int nRegBlockSizeMin;
  int row_interleave;

  if (blocking_params) {
    kBlock = blocking_params->KCB;
    nBlock = blocking_params->NCB;
    mRegBlockSize = blocking_params->MR;
    nRegBlockSize = blocking_params->NR;
    nRegBlockSizeMin = blocking_params->NR_MIN;
    row_interleave = blocking_params->ROW_INTERLEAVE;
  } else {
    kBlock = PackingTraits<uint8_t, int32_t, instSet>::KCB;
    nBlock = PackingTraits<uint8_t, int32_t, instSet>::NCB;
    mRegBlockSize = PackingTraits<uint8_t, int32_t, instSet>::MR;
    nRegBlockSize = PackingTraits<uint8_t, int32_t, instSet>::NR;
    nRegBlockSizeMin = PackingTraits<uint8_t, int32_t, instSet>::NR_MIN;
    row_interleave = PackingTraits<uint8_t, int32_t, instSet>::ROW_INTERLEAVE;
  }
  (void)nRegBlockSizeMin; // Suppress unused variable warning

  kernelSig = std::make_tuple(
      accum, mc, nc, nBlock, kBlock, mRegBlockSize, nRegBlockSize);

  return codeCache_.getOrCreate(kernelSig, [&]() -> jit_micro_kernel_fp {
    asmjit::CodeHolder code;
    code.init(runtime().environment());
    la64::Assembler assembler(&code);
    la64::Emitter* a = assembler.as<la64::Emitter>();
#if defined(FBGEMM_LOG_CODE)
    // generated code logging
    FILE* codeLogfile = fopen(
        getCodeLoggingFile<instSet>(
            accum, mc, nc, nBlock, kBlock, mRegBlockSize, nRegBlockSize)
            .c_str(),
        "w");
    asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
    if (codeLogger) {
      code.setLogger(codeLogger);
    }
#endif

    assert(
        kc % row_interleave == 0 && "kc must be a multiple of row_interleave");
    assert(nc % nRegBlockSizeMin == 0 && "nc must be a multiple of NR_MIN");
    const int maxMRegs = mRegBlockSize;
    (void)maxMRegs; // Suppress unused variable warning
    const int maxNRegs = nRegBlockSize * row_interleave / vectorLen;
    assert(
        maxMRegs * maxNRegs <= numRegs - 4 &&
        "MRegs x NRegs is above available registers (MAX_REGS - 4)");

    int mRegBlocks = mc / mRegBlockSize;
    int mRegBlocksRem = mc % mRegBlockSize;

    la64::Gp buffer_A = la64::a0;
    la64::Gp buffer_B = la64::a1;
    la64::Gp B_pf = la64::a2;
    la64::Gp CBase = la64::a3;
    la64::Gp kSize = la64::a4;
    la64::Gp ldcReg = la64::a5;

    asmjit::FuncDetail func;
    func.init(
        asmjit::FuncSignatureT<
            void,
            uint8_t*,
            int8_t*,
            int8_t*,
            int32_t*,
            int,
            int>(asmjit::CallConv::kIdHost),
        a->environment());

    asmjit::FuncFrame frame;
    frame.init(func);

    auto dirtyVecRegs = asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
        asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15);
    if (numRegs >= 16) {
      dirtyVecRegs |= asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
          asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31);
    }

    frame.setDirtyRegs(la64::Reg::kGroupVec, dirtyVecRegs);
    frame.setDirtyRegs(
        la64::Reg::kGroupGp,
        asmjit::Support::bitMask(23, 24, 25, 26, 27, 28, 29, 30) | asmjit::Support::bitMask(31));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(buffer_A, buffer_B, B_pf, CBase, kSize, ldcReg);

    args.updateFuncFrame(frame);
    frame.finalize();

    a->emitProlog(frame);
    a->emitArgsAssignment(frame, args);

    asmjit::Label LoopMBlocks = a->newLabel();
    asmjit::Label LoopNBlocks = a->newLabel();

    la64::Gp buffer_B_saved = la64::a6;
    la64::Gp C_Offset = la64::a7;
    la64::Gp B_pf_saved = la64::s0;
    la64::Gp iIdx = la64::s1;
    la64::Gp jIdx = la64::s2;
    la64::Gp kIdx = la64::s3;
    la64::Gp tmpReg1 = la64::s4;

    VecRegT oneReg(numRegs - 3);

    gen16BitVectorOne<instSet, VecRegT>(a, oneReg);
    mov_imm(a, tmpReg1, sizeof(int32_t));
    a->mul_d(ldcReg, ldcReg, tmpReg1);

    // save B_buffer address
    a->add_d(buffer_B_saved, buffer_B, la64::zero);
    a->add_d(B_pf_saved, B_pf, la64::zero);

    int currColRegs = nc * row_interleave / vectorLen;
    int colRegs = std::min(currColRegs, maxNRegs);

    auto issueLoopOverK = [&](int rowRegs) {
      asmjit::Label LoopKLabel = a->newLabel();

      // Init C (result) vector registers
      initCRegs(a, rowRegs, colRegs);

      // Loops over K
      a->xor_(kIdx, kIdx, kIdx);
      a->bind(LoopKLabel);

      // k is incremented by row_interleave
      a->addi_d(kIdx, kIdx, row_interleave);

      genComputeBlock<instSet>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, kBlock);

      // update buffer_A address for next k iteration
      a->addi_d(
          buffer_A, buffer_A, static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

      // update buffer_B address for next k iteration
      mov_imm(a, tmpReg1, nBlock * row_interleave * sizeof(int8_t));
      a->add_d(buffer_B, buffer_B, tmpReg1);
      a->add_d(B_pf, B_pf, tmpReg1);

      a->blt(kIdx, kSize, LoopKLabel);

      // store C matrix
      storeCRegs<instSet>(a, rowRegs, colRegs, C_Offset, ldcReg, accum);
    };

    if (mRegBlocks > 0) {
      // move 0 to iteration variables
      a->xor_(iIdx, iIdx, iIdx);

      a->bind(LoopMBlocks);
      a->addi_d(iIdx, iIdx, 1);

      a->xor_(jIdx, jIdx, jIdx);

      a->bind(LoopNBlocks);
      a->addi_d(jIdx, jIdx, 1);

      issueLoopOverK(mRegBlockSize);

      int rowRegs = mRegBlockSize;

      // reset A
      a->sub_d(buffer_A, buffer_A, kSize);

      // B for next block
      a->add_d(buffer_B, buffer_B_saved, la64::zero);
      // using C_Offset as temp reg
      mov_imm(a, tmpReg1, nRegBlockSize * row_interleave * sizeof(int8_t));
      a->mul_d(C_Offset, jIdx, tmpReg1);
      a->add_d(buffer_B, buffer_B, C_Offset);
      a->add_d(B_pf, B_pf_saved, la64::zero);
      a->add_d(B_pf, B_pf, C_Offset);

      // increment C for next B block
      a->addi_d(CBase, CBase, static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int32_t)));

      int jLoopTrips = currColRegs / maxNRegs;
      // jLoopTrips should be at least 1
      jLoopTrips = jLoopTrips ? jLoopTrips : 1;
      mov_imm(a, tmpReg1, jLoopTrips);
      a->blt(jIdx, tmpReg1, LoopNBlocks);

      // increment A for next block
      mov_imm(a, tmpReg1, (rowRegs)*kBlock * sizeof(uint8_t));
      a->add_d(
          buffer_A, buffer_A,tmpReg1);

      // increment C for next A block
      a->addi_d(
          CBase, CBase,
          -1 * jLoopTrips * nRegBlockSize * sizeof(int32_t));
      mov_imm(a, tmpReg1, rowRegs);
      a->mul_d(C_Offset, ldcReg, tmpReg1);
      a->add_d(CBase, CBase, C_Offset);

      // reset B
      a->add_d(buffer_B, buffer_B_saved, la64::zero);
      a->add_d(B_pf, B_pf_saved, la64::zero);
      mov_imm(a, tmpReg1, mRegBlocks);
      a->blt(iIdx, tmpReg1, LoopMBlocks);
    }
    // generate code for remainder
    if (mRegBlocksRem > 0) {
      asmjit::Label LoopNRem = a->newLabel();

      a->xor_(jIdx, jIdx, jIdx);
      a->bind(LoopNRem);
      a->addi_d(jIdx, jIdx, 1);

      issueLoopOverK(mRegBlocksRem);

      // increment C for next B block
      a->addi_d(CBase, CBase, static_cast<asmjit::Imm>(nRegBlockSize * sizeof(int32_t)));

      int jLoopTrips = currColRegs / maxNRegs;
      // jLoopTrips should be at least 1
      jLoopTrips = jLoopTrips ? jLoopTrips : 1;
      mov_imm(a, tmpReg1, jLoopTrips);
      a->blt(jIdx, tmpReg1, LoopNRem);
    }

    a->emitEpilog(frame);

    jit_micro_kernel_fp fn;
    asmjit::Error err;
    {
      std::unique_lock<std::mutex> lock(rtMutex_);
      err = runtime().add(&fn, &code);
    }
    if (err) {
      std::cout << "Error: in fn add" << std::endl;
      return nullptr;
    }

#if defined(FBGEMM_LOG_CODE)
    fclose(codeLogfile);
    delete codeLogger;
#endif

    return fn;
  });
}

template CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::getOrCreate<inst_set_t::lasx>(
    bool accum,
    int32_t mc,
    int32_t nc,
    int32_t kc);

template void
CodeGenBase<uint8_t, int8_t, int32_t, int32_t>::storeCRegs<inst_set_t::lasx>(
    la64::Emitter* a,
    int rowRegs,
    int colRegs,
    la64::Gp C_Offset,
    la64::Gp ldcReg,
    bool accum);

} // namespace fbgemm
