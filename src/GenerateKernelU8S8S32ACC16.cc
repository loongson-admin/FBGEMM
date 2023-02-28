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

template <>
template <>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::genComputeBlock<
    inst_set_t::lasx>(
    la64::Emitter* a,
    la64::Gp buffer_A,
    la64::Gp buffer_B,
    la64::Gp /* unused (reserved for prefetching)*/,
    int rowRegs,
    int colRegs,
    int lda) {
  using CRegs = la64::VecX;
  static constexpr int vectorLen = simd_info<inst_set_t::lasx>::WIDTH_BYTES;

  // used for matrix A
  la64::VecX AReg = la64::xr13;
  la64::VecX tmpReg = la64::xr14;
  la64::VecX tmpReg31 = la64::xr27;
  la64::VecX tmpReg26 = la64::xr26;
  la64::Gp tempGPX = la64::s5;
  la64::Gp temp_buffer = la64::s6;

  for (int i = 0; i < rowRegs; ++i) {
    // broadcast A
    // printf("i:%d, lda:%d, sizeof(uint8_t):%d,  all:%d\n",i,lda,sizeof(uint8_t),(i * lda) * sizeof(uint8_t));
    a->addi_d(temp_buffer, buffer_A, (i * lda) * sizeof(uint8_t));
    a->ld_d(tempGPX, ptr(temp_buffer));
    a->xvreplgr2vr_h(AReg, tempGPX);
    for (int j = 0; j < colRegs; ++j) {

      a->addi_d(temp_buffer, buffer_B, j * vectorLen * sizeof(int8_t));
      a->xvld(tmpReg31, ptr(temp_buffer));
      a->xvmulwev_h_bu_b (tmpReg, AReg, tmpReg31);
      a->xvmulwod_h_bu_b(tmpReg26, AReg, tmpReg31);
      a->xvsadd_h(tmpReg, tmpReg, tmpReg26);
      a->xvsadd_h(CRegs(i * colRegs + j), tmpReg, CRegs(i * colRegs + j));
      // Prefetching is hurting performance in some cases
      // because prefetch instructions itself consumes a slot
      // in pipeline issue thus slowing down the kernel.
      // if((i == rowRegs - 1) && j % 2 == 0){
      // a->preldx(0, B_pf, j*VLEN_*sizeof(int8_t));
      // }
    }
  }
}

 /**
  * Generate instructions for storing the C registers back to the memory
  * in 16-bit Accumulation kernel.
  */
template <>
template <inst_set_t instSet>
void CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::storeCRegs(
    la64::Emitter* a,
    int rowRegs,
    int colRegs,
    la64::Gp C_Offset,
    la64::Gp ldcReg,
    bool accum) {
  using VecT = typename simd_info<instSet>::vec_reg_t;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  // VecT extractDestFull(simd_info<instSet>::NUM_VEC_REGS - 1);
  VecT extractDestFull(16 - 1);
  auto extractDestHalf = extractDestFull.half();

  la64::Gp tmpReg1 = la64::s5;
  la64::Gp tmpReg2 = la64::s6;
  la64::VecX tmpReg31 = la64::xr28;

  for (int i = 0; i < rowRegs; ++i) {
    mov_imm(a, tmpReg1, (i * sizeof(int32_t)));
    a->mul_d(C_Offset, ldcReg, tmpReg1);

    for (int j = 0; j < colRegs; ++j) {
      for (int idx = 0; idx < 2; ++idx) {
        emitExtractHalfVector<instSet, VecT>(
            a, extractDestHalf, VecT(i * colRegs + j), idx);
        a->xvpermi_d(tmpReg31, extractDestHalf, 0x40);
        a->xvexth_w_h(extractDestFull, tmpReg31);
        a->addi_d(tmpReg2, C_Offset, (j * 2 + idx) * vectorLen);
        a->add_d(tmpReg2, tmpReg2, la64::a3);
        la64::Mem destAddr = ptr(tmpReg2);
        if (accum) {
          a->xvld(tmpReg31, destAddr);
          a->xvadd_w(extractDestFull, extractDestFull, tmpReg31);
        }
        a->xvst(extractDestFull, destAddr);
      }
    }
  }
}

 /**
  * Get or Create the LASX instructions for 16-bit Accumulation macro-kernel.
  *
  */
template <>
template <>
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::jit_micro_kernel_fp
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::getOrCreate<inst_set_t::lasx>(
    bool accum,
    int32_t mc,
    int32_t nc,
    int32_t kc) {
  (void)kc; // Suppress unused variable warning
  constexpr int vectorLen = simd_info<inst_set_t::lasx>::WIDTH_BYTES;
  constexpr int numRegs = simd_info<inst_set_t::lasx>::NUM_VEC_REGS;
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
    kBlock = PackingTraits<uint8_t, int16_t, inst_set_t::lasx>::KCB;
    nBlock = PackingTraits<uint8_t, int16_t, inst_set_t::lasx>::NCB;
    mRegBlockSize = PackingTraits<uint8_t, int16_t, inst_set_t::lasx>::MR;
    nRegBlockSize = PackingTraits<uint8_t, int16_t, inst_set_t::lasx>::NR;
    nRegBlockSizeMin =
        PackingTraits<uint8_t, int16_t, inst_set_t::lasx>::NR_MIN;
    row_interleave =
        PackingTraits<uint8_t, int16_t, inst_set_t::lasx>::ROW_INTERLEAVE;
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
        getCodeLoggingFile<inst_set_t::lasx>(
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
    const int maxNRegs = nRegBlockSize * row_interleave / vectorLen;
    (void)maxMRegs; // Suppress unused variable warning
    (void)maxNRegs; // Suppress unused variable warning
    assert(
        maxMRegs * maxNRegs <= 13 &&
        "MR*(NR*ROW_INTERLEAVE*8/256"
        "must be <= 13(available registers constraint)");

    int mRegBlocks = mc / mRegBlockSize;
    int mRegBlocksRem = mc % mRegBlockSize;

    // assert((nc == nRegBlockSize) &&
    //"nc must be equal to the number of register blocks");

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
        la64::Reg::kGroupGp, asmjit::Support::bitMask(23, 24, 25, 26, 27, 28, 29, 30) | asmjit::Support::bitMask(31));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(buffer_A, buffer_B, B_pf, CBase, kSize, ldcReg);

    args.updateFuncFrame(frame);
    frame.finalize();

    a->emitProlog(frame);
    a->emitArgsAssignment(frame, args);

    asmjit::Label Loopk = a->newLabel();
    asmjit::Label LoopMBlocks = a->newLabel();

    la64::Gp buffer_B_saved = la64::a6;
    la64::Gp C_Offset = la64::a7;
    la64::Gp iIdx = la64::s1;
    la64::Gp kIdx = la64::s2;
    la64::Gp tmpReg1 = la64::s4;

    int colRegs = nc * row_interleave / vectorLen;
    if (mRegBlocks > 0) {
      // move 0 to iteration variables
      a->xor_(iIdx, iIdx, iIdx);

      // save B_buffer address
      a->add_d(buffer_B_saved, buffer_B, la64::zero);

      a->bind(LoopMBlocks);
      a->addi_d(iIdx, iIdx, 1);

      int rowRegs = mRegBlockSize;

      // init C registers
      initCRegs(a, rowRegs, colRegs);

      // init k loop index
      a->xor_(kIdx, kIdx, kIdx);
      a->bind(Loopk);
      // k is incremented by row_interleave
      a->addi_d(kIdx, kIdx, row_interleave);

      genComputeBlock<inst_set_t::lasx>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, kBlock);

      // update buffer_A address for next k iteration
      a->addi_d(
          buffer_A, buffer_A, row_interleave * sizeof(uint8_t));

      // update buffer_B address for next k iteration
      a->addi_d(
          buffer_B, buffer_B,
          nBlock * row_interleave * sizeof(int8_t));

      a->blt(kIdx, kSize, Loopk);

      // store C matrix
      storeCRegs<inst_set_t::lasx>(
          a, rowRegs, colRegs, C_Offset, ldcReg, accum);

      // increment A for next block
      a->sub_d(buffer_A, buffer_A, kSize);
      a->addi_d(
          buffer_A, buffer_A,
          (rowRegs)*kBlock * sizeof(uint8_t));
      // increment C for next block
      mov_imm(a,
          tmpReg1,
          rowRegs * sizeof(int32_t));
      a->mul_d(C_Offset, ldcReg, tmpReg1);
      a->add_d(CBase, CBase, C_Offset);
      // reset B
      a->add_d(buffer_B, buffer_B_saved, la64::zero);

      mov_imm(a, tmpReg1, mRegBlocks);
      a->blt(iIdx, tmpReg1, LoopMBlocks);
    }
    // generate code for remainder
    if (mRegBlocksRem > 0) {
      asmjit::Label LoopkRem = a->newLabel();
      int rowRegs = mRegBlocksRem;

      // init C registers
      initCRegs(a, rowRegs, colRegs);

      // init k loop index
      a->xor_(kIdx, kIdx, kIdx);
      a->bind(LoopkRem);

      // k is incremented by row_interleave
      a->addi_d(kIdx, kIdx, row_interleave);

      genComputeBlock<inst_set_t::lasx>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, kBlock);

      // update buffer_A address for next k iteration
      a->addi_d(
          buffer_A, buffer_A, row_interleave * sizeof(uint8_t));

      // update buffer_B address for next k iteration
      a->addi_d(
          buffer_B, buffer_B,
          nBlock * row_interleave * sizeof(int8_t));

      a->blt(kIdx, kSize, LoopkRem);

      // store C matrix
      storeCRegs<inst_set_t::lasx>(
          a, rowRegs, colRegs, C_Offset, ldcReg, accum);
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

/**
 * Instantiate the inst_set_t::lasx instructions for store kernel.
 *
 */
template void
CodeGenBase<uint8_t, int8_t, int32_t, int16_t>::storeCRegs<inst_set_t::lasx>(
    la64::Emitter* a,
    int rowRegs,
    int colRegs,
    la64::Gp C_Offset,
    la64::Gp ldcReg,
    bool accum);

} // namespace fbgemm
