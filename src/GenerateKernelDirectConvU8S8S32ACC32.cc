/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include "./CodeGenHelpers.h"
#include "./DirectConv.h"

namespace fbgemm {

namespace la64 = asmjit::la64;

template <>
template <inst_set_t instSet>
void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::storeCRegs(
    la64::Emitter* a,
    int rowRegs,
    int colRegs,
    la64::Gp C_Offset,
    la64::Gp ldcReg,
    bool accum) {
  using VecT = typename simd_info<instSet>::vec_reg_t;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;
  la64::VecX tmpReg31 = la64::xr31;
  la64::Gp tempGpX = la64::t7;

  for (int i = 0; i < rowRegs; ++i) {
    if (i != 0) {
      a->add_d(C_Offset, C_Offset, ldcReg);
    } else {
      a->xor_(C_Offset, C_Offset, C_Offset);
    }
    for (int j = 0; j < colRegs; ++j) {
      a->addi_d(tempGpX, C_Offset, j * vectorLen * sizeof(int8_t));
      a->add_d(tempGpX, tempGpX, la64::a3);
      if (accum) {
        a->xvld(tmpReg31, ptr(tempGpX));
        a->xvadd_w(VecT(i * colRegs + j), VecT(i * colRegs + j), tmpReg31);
      }
      a->xvst(VecT(i * colRegs + j), ptr(tempGpX));
    }
  }
}

template <>
template <inst_set_t instSet>
void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    genComputeBlockDirectConv(
        la64::Emitter* a,
        la64::Gp buffer_A,
        la64::Gp buffer_B,
        la64::Gp /*B_pf*/,
        int rowRegs,
        int colRegs,
        int strideXich) {
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
  VecRegT tmpReg_6(numRegs - 6);
  la64::Gp tempGpX = la64::t7;

  for (int j = 0; j < colRegs; ++j) {
    // load B
    emitLoadDWord<instSet, VecRegT>(
        a, BReg, ptr(buffer_B, j * vectorLen * sizeof(int8_t)));
    // load A, broadcast and fmas
    for (int i = 0; i < rowRegs; ++i) {
      a->ldx_d(tempGpX, ptr(buffer_A, (i * strideXich) * sizeof(uint8_t)));
      a->xvreplgr2vr_w(AReg, tempGpX);

      a->xvmulwev_h_bu_b (res1, AReg, BReg);
      a->xvmulwod_h_bu_b(tmpReg_6, AReg, BReg);
      a->xvsadd_h(res1, res1, tmpReg_6);
      a->xvmulwev_w_h (tmpReg1, oneReg, res1);
      a->xvmaddwod_w_h(tmpReg1, oneReg, res1);
      a->xvadd_w(VecRegT(i * colRegs + j), tmpReg1, VecRegT(i * colRegs + j));
    }
    // a->prefetcht0(x86::dword_ptr(B_pf, j * vectorLen * sizeof(int8_t)));
  }
}

template <>
template <inst_set_t instSet>
DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::getOrCreateDirectConv(
    bool accum,
    int32_t O1,
    int32_t i1Xich,
    int32_t strideXich) {
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  std::tuple<bool, int, int, int, int, int, int> kernelSig;
  // int ichSize = 32;
  int mRegBlockSize = 12;
  int nRegBlockSize = 8;
  // int nRegBlockSizeMin;
  int row_interleave = 4;

  kernelSig = std::make_tuple(
      accum, O1, i1Xich, strideXich, i1Xich, mRegBlockSize, nRegBlockSize);

  return codeCache_.getOrCreate(kernelSig, [&]() -> jit_micro_kernel_fp {
    asmjit::CodeHolder code;
    code.init(runtime().environment());
    la64::Assembler assembler(&code);
    la64::Emitter* a = assembler.as<la64::Emitter>();
#if defined(FBGEMM_LOG_CODE)
    // generated code logging
    FILE* codeLogfile = fopen(
        getCodeLoggingFile<instSet>(
            accum, O1, i1Xich, strideXich, i1Xich, mRegBlockSize, nRegBlockSize)
            .c_str(),
        "w");
    asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
    if (codeLogger) {
      code.setLogger(codeLogger);
    }
#endif

    const int maxMRegs = mRegBlockSize;
    (void)maxMRegs; // Suppress unused variable warning
    const int maxNRegs = nRegBlockSize * row_interleave / vectorLen;
    assert(
        maxMRegs * maxNRegs <= numRegs - 4 &&
        "MRegs x NRegs is above available registers (MAX_REGS - 4)");

    int O1RegBlocks = O1 / mRegBlockSize;
    int O1RegBlocksRem = O1 % mRegBlockSize;

    la64::Gp buffer_A = la64::a0;
    la64::Gp buffer_B = la64::a1;
    la64::Gp B_pf = la64::a2;
    la64::Gp CBase = la64::a3;
    la64::Gp ichXk1 = la64::a4;
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
        asmjit::Support::bitMask(23, 24, 25, 26, 27, 28, 29, 30));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(buffer_A, buffer_B, B_pf, CBase, ichXk1, ldcReg);

    args.updateFuncFrame(frame);
    frame.finalize();

    a->emitProlog(frame);
    a->emitArgsAssignment(frame, args);

    asmjit::Label LoopMBlocks = a->newLabel();
    la64::Gp buffer_B_saved = la64::a6;
    la64::Gp C_Offset = la64::a7;
    la64::Gp iIdx = la64::s1;
    la64::Gp kIdx = la64::s3;
    la64::Gp tmpReg1 = la64::t7;

    VecRegT oneReg(numRegs - 3);

    gen16BitVectorOne<instSet, VecRegT>(a, oneReg);
    mov_imm(a, tmpReg1, sizeof(int32_t));
    a->mul_d(ldcReg, ldcReg, tmpReg1);

    int colRegs = maxNRegs;

    auto issueLoopOverK = [&](int rowRegs) {
      // loopKLabel: corresponds to loop "r" where r = 0
      // loopK0Label: corresponds to loop "r" where r = 1
      asmjit::Label LoopKLabel = a->newLabel();
      asmjit::Label LoopK0Label = a->newLabel();

      // Init C (result) vector registers
      initCRegs(a, rowRegs, colRegs);

      // Loops over K: input channel
      // a.k.a this issueLoopOverK code block generates code
      // corresponding to the "ich" loop of the psedo-code
      a->xor_(kIdx, kIdx, kIdx);
      a->bind(LoopKLabel);

      // k is incremented by row_interleave
      a->addi_d(kIdx, kIdx, row_interleave);

      // this ComputeBlock generates code correspondent to
      // the above psedu-code since the kernel_height loop (loop "r").
      // And because K[0] == 2 and IN_DIM[2] (requirement #2),
      // we can unroll loop "r" here. Thus this following
      // genComputeBlockDirectConv generates code for loop "r" = 0
      genComputeBlockDirectConv<instSet>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, strideXich);

      // update buffer_A address for next k iteration
      a->addi_d(
          buffer_A, buffer_A, row_interleave * sizeof(uint8_t));

      // update buffer_B address for next k iteration
      a->addi_d(buffer_B, buffer_B, 8 * sizeof(int32_t));
      a->addi_d(B_pf, B_pf, 8 * sizeof(int32_t));

      a->blt(kIdx, ichXk1, LoopKLabel);

      a->sub_d(buffer_A, buffer_A, ichXk1);

      a->addi_d(buffer_A, buffer_A, i1Xich);

      a->xor_(kIdx, kIdx, kIdx);
      a->bind(LoopK0Label);

      // k is incremented by row_interleave
      a->addi_d(kIdx, kIdx, row_interleave);

      // this ComputeBlock generates code that corresponds
      // to the kernel_height loop (loop "r") in the psedu-code above.
      // And the following genComputeBlockDirectConv
      // generates code for loop "r" where "r" = 1
      genComputeBlockDirectConv<instSet>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, strideXich);

      // update buffer_A address for next k iteration
      a->addi_d(
          buffer_A, buffer_A, row_interleave * sizeof(uint8_t));

      // update buffer_B address for next k iteration
      a->addi_d(buffer_B, buffer_B, 8 * sizeof(int32_t));
      a->addi_d(B_pf, B_pf, 8 * sizeof(int32_t));

      a->blt(kIdx, ichXk1, LoopK0Label);

      a->sub_d(buffer_A, buffer_A, ichXk1);

      // store C matrix
      storeCRegs<instSet>(a, rowRegs, colRegs, C_Offset, ldcReg, accum);
    };

    if (O1RegBlocks > 0) {
      // move 0 to iteration variables
      a->xor_(iIdx, iIdx, iIdx);

      // iIdex loop corresponds to kernel_width loop (loop "s")
      // in the direct conv loops
      a->bind(LoopMBlocks);
      a->addi_d(iIdx, iIdx, 1);

      // save B_buffer address
      a->add_d(buffer_B_saved, buffer_B, la64::zero);

      issueLoopOverK(mRegBlockSize);

      int rowRegs = mRegBlockSize;

      // reset A
      a->addi_d(buffer_A, buffer_A, -1 * i1Xich);

      // increment A for next block
      a->addi_d(
          buffer_A, buffer_A,
          rowRegs * strideXich * sizeof(uint8_t));

      // B for next block
      a->add_d(buffer_B, buffer_B_saved,  la64::zero);

      // increment C for next B block
      // ldcReg already multiplied with 4 (sizeof(int32_t))
      mov_imm(a, tmpReg1, rowRegs * sizeof(int8_t));
      a->mul_d(C_Offset, ldcReg, tmpReg1);
      a->add_d(CBase, CBase, C_Offset);

      mov_imm(a, tmpReg1, O1RegBlocks);
      a->blt(iIdx, tmpReg1, LoopMBlocks);
    }

    // generate code for remainder
    if (O1RegBlocksRem > 0) {
      issueLoopOverK(O1RegBlocksRem);
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

template <>
template <inst_set_t instSet>
void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::storeCRegsTrans(
    la64::Emitter* a,
    int rowRegs,
    int colRegs,
    la64::Gp C_offset,
    la64::Gp o1XocReg,
    la64::Gp ldcReg,
    bool accum) {
  using VecT = typename simd_info<instSet>::vec_reg_t;
  // static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;
  VecT tmpReg1(27);
  la64::Gp tempGPX = la64::s5;
  a->xor_(C_offset, C_offset, C_offset);


  for (int i = 0; i < rowRegs; ++i) {
    for (int j = 0; j < colRegs; ++j) {
      a->add_d(tempGPX, la64::a3, C_offset);
      if (accum) {
        a->xvld(tmpReg1, ptr(tempGPX));
        a->xvadd_w(VecT(i * colRegs + j), VecT(i * colRegs + j), tmpReg1);
      }
      a->xvst(VecT(i * colRegs + j),ptr(tempGPX));
      a->add_d(C_offset, C_offset, ldcReg);
    }
    a->add_d(C_offset, C_offset, o1XocReg);
  }
}

template <>
template <inst_set_t instSet>
void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    genComputeBlockDirectConvTrans(
        la64::Emitter* a,
        la64::Gp buffer_A,
        la64::Gp buffer_B,
        la64::Gp icReg,
        la64::Gp C_offset,
        int rowRegs,
        int colRegs) {
  // static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;
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

  // temporary register 2
  VecRegT tmpReg1(numRegs - 5);
  VecRegT tmpReg_6(numRegs - 6);

  la64::Gp tempGpX = la64::s6;

  // load A
  a->xvldrepl_w(AReg, ptr(buffer_A));

  a->xor_(C_offset, C_offset, C_offset);
  for (int i = 0; i < rowRegs; ++i) {
    for (int j = 0; j < colRegs; ++j) {
      // load B, broadcast and fmas
      a->slli_d(tempGpX, C_offset, 3);
      a->add_d(tempGpX,buffer_B, tempGpX);
      emitLoadDWord<instSet, VecRegT>(
          a, BReg, ptr(tempGpX));
      a->xvmulwev_h_bu_b (res1, AReg, BReg);
      a->xvmulwod_h_bu_b(tmpReg_6, AReg, BReg);
      a->xvsadd_h(res1, res1, tmpReg_6);
      a->xvmulwev_w_h (tmpReg1, oneReg, res1);
      a->xvmaddwod_w_h(tmpReg1, oneReg, res1);
      a->xvadd_w(VecRegT(i * colRegs + j), tmpReg1, VecRegT(i * colRegs + j));
      a->add_d(C_offset, C_offset, icReg);
    }
    // a->prefetcht0(x86::dword_ptr(B_pf, j * vectorLen * sizeof(int8_t)));
  }
}

template <>
template <inst_set_t instSet>
DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    jit_micro_kernel_fp_convT
    DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
        getOrCreateDirectConvTrans(bool accum, int32_t stride) {
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;
  constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  std::tuple<bool, int, int, int> kernelSig;
  constexpr int mRowRegBlockSize = 2;
  constexpr int mColRegBlockSize = 6;
  constexpr int mRegBlockSize = mRowRegBlockSize * mColRegBlockSize;
  constexpr int nRegBlockSize = 8;
  constexpr int row_interleave = 4;

  kernelSig = std::make_tuple(accum, stride, mRegBlockSize, nRegBlockSize);

  return codeCacheT_.getOrCreate(kernelSig, [&]() -> jit_micro_kernel_fp_convT {
    asmjit::CodeHolder code;
    code.init(runtime().environment());
    la64::Assembler assembler(&code);
    la64::Emitter* a = assembler.as<la64::Emitter>();
#if defined(FBGEMM_LOG_CODE)
    // generated code logging
    FILE* codeLogfile = fopen(
        getCodeLoggingFile<instSet>(
            accum, stride, 0, 0, 0, mRegBlockSize, nRegBlockSize)
            .c_str(),
        "w");
    asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
    if (codeLogger) {
      code.setLogger(codeLogger);
    }
#endif

    const int maxMRegs = mRegBlockSize;
    (void)maxMRegs; // Suppress unused variable warning
    const int maxNRegs = nRegBlockSize * row_interleave / vectorLen;
    assert(
        maxMRegs * maxNRegs <= numRegs - 4 &&
        "MRegs x NRegs is above available registers (MAX_REGS - 4)");

    la64::Gp buffer_A = la64::a0;
    la64::Gp buffer_B = la64::a1;
    la64::Gp CBase = la64::a3;
    la64::Gp ic = la64::a4;
    la64::Gp ldcReg = la64::a5;
    la64::Gp o1Xoc = la64::a6;
    la64::Gp i1 = la64::a7;

    asmjit::FuncDetail func;
    func.init(
        asmjit::FuncSignatureT<
            void,
            uint8_t*,
            int8_t*,
            int32_t*,
            int,
            int,
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
        asmjit::Support::bitMask(23, 24, 25, 26, 27, 28, 29, 30));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(buffer_A, buffer_B, CBase, ic, ldcReg, o1Xoc, i1);

    args.updateFuncFrame(frame);
    frame.finalize();

    a->emitProlog(frame);
    a->emitArgsAssignment(frame, args);

    asmjit::Label LoopMBlocks = a->newLabel();

    la64::Gp C_offset = la64::s0;
    la64::Gp buffer_B_saved = la64::s1;
    la64::Gp iIdx = la64::s2;
    la64::Gp kIdx = la64::s3;
    la64::Gp tmpReg1 = la64::t7;

    VecRegT oneReg(numRegs - 3);

    gen16BitVectorOne<instSet, VecRegT>(a, oneReg);
    mov_imm(a, tmpReg1, sizeof(int32_t));
    a->mul_d(ldcReg, ldcReg, tmpReg1);

    int colRegs = maxNRegs;

    auto issueLoopOverK = [&](int rowRegs) {
      asmjit::Label LoopKLabel = a->newLabel();

      // Init C (result) vector registers
      initCRegs(a, rowRegs, colRegs);

      // Loops over K: input channel
      // corresponds to the "icb" loop in the pseudo code
      a->xor_(kIdx, kIdx, kIdx);
      a->bind(LoopKLabel);

      // k is incremented by row_interleave
      a->addi_d(kIdx, kIdx, 4);
      genComputeBlockDirectConvTrans<instSet>(
          a,
          buffer_A,
          buffer_B,
          ic,
          C_offset,
          mRowRegBlockSize,
          mColRegBlockSize);

      // update buffer_A address for next k iteration
       a->addi_d(
          buffer_A, buffer_A, static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

      // update buffer_B address for next k iteration
      a->addi_d(buffer_B, buffer_B, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

      a->blt(kIdx, ic, LoopKLabel);

      // store C matrix
      storeCRegsTrans<instSet>(
          a,
          mRowRegBlockSize,
          mColRegBlockSize,
          C_offset,
          o1Xoc,
          ldcReg,
          accum);
    };

    {
      // move 0 to iteration variables
      a->xor_(iIdx, iIdx, iIdx);

      a->bind(LoopMBlocks);
      a->addi_d(iIdx, iIdx, 1);

      // save B_buffer address
      a->add_d(buffer_B_saved, buffer_B, la64::zero);

      issueLoopOverK(mRegBlockSize);

      // B for next block
      a->add_d(buffer_B, buffer_B_saved, la64::zero);
      // increment C for next B block
      // ldcReg already multiplied by 4 (sizeof(int32_t))
      mov_imm(a, tmpReg1, stride);
      a->mul_d(C_offset, ldcReg, tmpReg1);
      a->add_d(CBase, CBase, C_offset);

      a->blt(iIdx, i1, LoopMBlocks);
    }

    a->emitEpilog(frame);

    jit_micro_kernel_fp_convT fn;
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

template void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    storeCRegs<inst_set_t::lasx>(
        la64::Emitter* a,
        int rowRegs,
        int colRegs,
        la64::Gp C_Offset,
        la64::Gp ldcReg,
        bool accum);

/**
 * Instantiate the inst_set_t::lasx instructions for store kernel.
 *
 */
template void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    storeCRegsTrans<inst_set_t::lasx>(
        la64::Emitter* a,
        int rowRegs,
        int colRegs,
        la64::Gp C_offset,
        la64::Gp o1XocReg,
        la64::Gp ldcReg,
        bool accum);

/**
 * Instantiate the LASX instructions for 32-bit Accumulation macro-kernel.
 *
 */
template DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    jit_micro_kernel_fp
    DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
        getOrCreateDirectConv<inst_set_t::lasx>(
            bool accum,
            int32_t O1,
            int32_t i1Xich,
            int32_t strideXich);

/**
 * Instantiate the LASX instructions for 32-bit Accumulation macro-kernel.
 *
 */
template DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    jit_micro_kernel_fp_convT
    DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
        getOrCreateDirectConvTrans<inst_set_t::lasx>(
            bool accum,
            int32_t stride);

} // namespace fbgemm
