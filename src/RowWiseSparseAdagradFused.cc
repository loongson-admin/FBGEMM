/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmEmbedding.h"

#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <cassert>
#include <iostream>
#include <mutex>
#include "./CodeCache.h"
#include "./CodeGenHelpers.h"
#include "./MaskLasx.h"
#include "./RefImplementations.h"
#include "fbgemm/Utils.h"

using namespace std;

namespace fbgemm {
namespace {
namespace la64 = asmjit::la64;

template <typename indxType, typename offsetType, typename dataType>
class ReturnFunctionSignature {
 public:
  using jit_sparse_adagrad_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t data_size, // number of rows in w
      dataType* w, // input/output parameters
      const float* g, // input gradients
      float* h, // input/output momentums
      const indxType* indices, // indices of each row
      const offsetType* offsets_or_lengths,
      float epsilon,
      float lr,
      uint32_t* rand_buffer);
};

template <
    typename indxType,
    typename offsetType,
    typename dataType,
    inst_set_t instSet = inst_set_t::lasx>
class GenRowWiseSparseAdagradFused {
 public:
  GenRowWiseSparseAdagradFused() {}

  typename ReturnFunctionSignature<indxType, offsetType, dataType>::
      jit_sparse_adagrad_kernel
      getOrCreate(
          const int* mask_lasx,
          int block_size,
          int prefetch,
          bool use_offsets,
          bool use_stochastic_rounding,
          int grad_stride);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; // JIT Runtime for asmjit
    return rt;
  }

  static mutex rtMutex_; /// Controll access to runtime;

  // The hash depends on:
  // mask array, embedding dimension (block size), prefetch distance,
  // use_offsets and use_stochastic_rouding switch
  static CodeCache<
      tuple<const int*, int, int, bool, bool, int>,
      typename ReturnFunctionSignature<indxType, offsetType, dataType>::
          jit_sparse_adagrad_kernel>
      codeCache_; ///< JIT Code Cache for reuse.
}; // class GenRowWiseSparseAdagradFused

template <
    typename indxType,
    typename offsetType,
    typename dataType,
    inst_set_t instSet>
mutex GenRowWiseSparseAdagradFused<indxType, offsetType, dataType, instSet>::
    rtMutex_;

template <
    typename indxType,
    typename offsetType,
    typename dataType,
    inst_set_t instSet>
CodeCache<
    tuple<const int*, int, int, bool, bool, int>,
    typename ReturnFunctionSignature<indxType, offsetType, dataType>::
        jit_sparse_adagrad_kernel>
    GenRowWiseSparseAdagradFused<indxType, offsetType, dataType, instSet>::
        codeCache_;

template <
    typename indxType,
    typename offsetType,
    typename dataType,
    inst_set_t instSet>
typename ReturnFunctionSignature<indxType, offsetType, dataType>::
    jit_sparse_adagrad_kernel
    GenRowWiseSparseAdagradFused<indxType, offsetType, dataType, instSet>::
        getOrCreate(
            const int* mask_lasx, // runtime constant
            int block_size,
            int prefetch,
            bool use_offsets,
            bool use_stochastic_rounding,
            int grad_stride) {
  tuple<const int*, int, int, bool, bool, int> kernelSig = make_tuple(
      mask_lasx,
      block_size,
      prefetch,
      use_offsets,
      use_stochastic_rounding,
      grad_stride);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() -> typename ReturnFunctionSignature<
                indxType,
                offsetType,
                dataType>::jit_sparse_adagrad_kernel {
        asmjit::CodeHolder code;
        code.init(runtime().environment());
        la64::Assembler assembler(&code);
        la64::Emitter* a = assembler.as<la64::Emitter>();
        bool areIndices64b = is_same<indxType, int64_t>::value;
        bool areWeightsFp16 = is_same<dataType, float16>::value;
#if defined(FBGEMM_LOG_CODE)
        string filename = "RowWiseSparseAdagradFused";
        filename += "_emd_dim_" + to_string(block_size);
        filename += "_wei_float";
        filename += areWeightsFp16 ? "16" : "32";
        filename += areIndices64b ? "_64bit" : "_32bit";
        filename += "_lasx";
        if (prefetch) {
          filename += "_prefetch";
        }
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif

        la64::Gp rand_buffer = la64::s0;
        la64::Gp output_size = la64::a0;
        la64::Gp index_size = la64::a1;
        la64::Gp data_size = la64::a2;
        la64::Gp w = la64::a3;
        la64::Gp g = la64::a4;
        la64::Gp h = la64::a5;
        la64::Gp indices = la64::a6;
        la64::Gp lengths = la64::a7;
        la64::VecV epsilon = la64::VecV(0);
        la64::VecV lr = la64::VecV(1);
        la64::Gp lengths_R = la64::s1;
        la64::Gp scratchReg1 = la64::s2;
        la64::Gp scratchReg2 = la64::s3; // for prefetching
        la64::Gp tempGp = la64::s6;
        la64::Gp tempGp2 = la64::s5;
        la64::Gp temp_wptr = la64::s4;

        asmjit::FuncDetail func;
        func.init(
            asmjit::FuncSignatureT<
                bool, // return type
                int64_t, // output_size
                int64_t, // index_size
                int64_t, // data_size
                dataType*, // w
                const float*, // g
                float*, // h
                const indxType*, // indices
                const int*, // lengths
                float, // epsilon
                float, // lr then rand_buffer
                uint32_t*>(asmjit::CallConv::kIdHost),
            a->environment());

        asmjit::FuncFrame frame;
        frame.init(func);

        if (instSet == inst_set_t::lasx) {
          frame.setDirtyRegs(
              la64::Reg::kGroupVec,
              asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                  asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15, 16, 17));
        }
        frame.setDirtyRegs(
            la64::Reg::kGroupGp,
            asmjit::Support::bitMask(23, 24, 25, 26, 27, 28, 29, 30, 31));

        asmjit::FuncArgsAssignment args(&func);
        args.assignAll(
            output_size,
            index_size,
            data_size,
            w,
            g,
            h,
            indices,
            lengths,
            epsilon,
            lr,
            rand_buffer);

        args.updateFuncFrame(frame);
        frame.finalize();
        a->emitProlog(frame);
        a->emitArgsAssignment(frame, args);

        constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
        // constexpr int NUM_VEC_REG = simd_info<instSet>::NUM_VEC_REGS;
        constexpr int NUM_VEC_REG = 16;

        typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

        int num_vec_regs_per_block = (block_size + vlen - 1) / vlen;
        int remainder = block_size % vlen;

        vec_reg_t src_vreg; // for holding embedding value temporarily
        la64::VecX mask_vreg;

        // Reserve registers with small ids first because some of them need to
        // be used with an instruction not supported in avx512 for which a big
        // register id won't work.
        int first_available_vec_reg_id = 0;
        la64::VecX partial_sum_vreg = la64::VecX(first_available_vec_reg_id);
        ++first_available_vec_reg_id;
        vec_reg_t float_step_vreg = vec_reg_t(first_available_vec_reg_id);
        ++first_available_vec_reg_id;
        vec_reg_t epsilon_vreg = vec_reg_t(first_available_vec_reg_id);  //xr2
        ++first_available_vec_reg_id;
        vec_reg_t lr_vreg = vec_reg_t(first_available_vec_reg_id);   //xr3
        ++first_available_vec_reg_id;

        a->xvreplve0_w(epsilon_vreg, epsilon);
        a->xvreplve0_w(lr_vreg, lr);

        // Reserve vector registers for random buffer generating
        // S0...S3: global random buffer state
        // R: generated random number in uint32_t
        // r0: extracted random byte (uint8_t) shifted to bits[5...13]
        // r1: temp
        vec_reg_t R_vreg, S0_vreg, S1_vreg, S2_vreg, S3_vreg, r0_vreg, r1_vreg;
        if (areWeightsFp16 && use_stochastic_rounding) {
          R_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          S0_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          S1_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          S2_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          S3_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          r0_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;
          r1_vreg = vec_reg_t(first_available_vec_reg_id);
          first_available_vec_reg_id++;

          // Load random buffer for FP16 stochastic rounding
          if (instSet == inst_set_t::lasx) {
            a->xvld(S0_vreg,ptr(rand_buffer));
            a->xvld(S1_vreg,ptr(rand_buffer, 1 * vlen * sizeof(uint32_t)));
            a->xvld(S2_vreg,ptr(rand_buffer, 2 * vlen * sizeof(uint32_t)));
            a->xvld(S3_vreg,ptr(rand_buffer, 3 * vlen * sizeof(uint32_t)));
          }
        }

        if (remainder) {
          if (instSet == inst_set_t::lasx) {
            src_vreg = vec_reg_t(first_available_vec_reg_id);
            ++first_available_vec_reg_id;

            mask_vreg = la64::VecX(first_available_vec_reg_id);
            ++first_available_vec_reg_id;
            // Use scratchReg1 as temp
            mov_imm(a, scratchReg1, int64_t(mask_lasx));
            a->xvld(mask_vreg, ptr(scratchReg1, (vlen - remainder) % vlen * sizeof(int32_t)));
          }
        }
        // Need an extra mask for computing sum of gradients
        la64::VecX tmpReg = la64::VecX(16);
        int unroll_factor = NUM_VEC_REG - first_available_vec_reg_id;

        // Compute the end address of indices
        mov_imm(a, tempGp, sizeof(indxType));
        a->mul_d(scratchReg1, index_size, tempGp);
        a->add_d(scratchReg1, scratchReg1, indices);
        a->add_d(index_size, scratchReg1, la64::zero);

        asmjit::Label exit = a->newLabel();
        asmjit::Label error = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        // rangeIndex loop begin (iterate output_size times)
        a->bind(LoopRangeIndexBegin);
        a->addi_d(output_size, output_size, -1);
        a->blt(output_size, la64::zero, LoopRangeIndexEnd);

        // Compute sq avg of gradients
        constexpr int vlen_lasx =
            simd_info<inst_set_t::lasx>::WIDTH_32BIT_ELEMS;
        int num_vec_regs_per_block_lasx =
            (block_size + vlen_lasx - 1) / vlen_lasx;

        a->xvxor_v(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);

        // TODO: need to do a tree-reduction to fully take advantage of
        // unrolling
        for (int vec_idx = 0; vec_idx < num_vec_regs_per_block_lasx;
             vec_idx += unroll_factor) {
          int cur_unroll_factor =
              std::min(unroll_factor, num_vec_regs_per_block_lasx - vec_idx);
          for (int v = 0; v < cur_unroll_factor; ++v) {
            la64::VecX out_vreg = la64::VecX(v + first_available_vec_reg_id);

            auto g_ptr =
                la64::ptr(g, (vec_idx + v) * vlen_lasx * sizeof(float));
            if (block_size % simd_info<inst_set_t::lasx>::WIDTH_32BIT_ELEMS &&
                vec_idx + v == num_vec_regs_per_block_lasx - 1) {
              if (instSet == inst_set_t::lasx) {
                a->xvld(out_vreg, g_ptr);
                a->xvand_v(out_vreg, out_vreg, mask_vreg);
              }
            } else {
              a->xvld(out_vreg, g_ptr);
            }
            a->xvfmul_s(out_vreg, out_vreg, out_vreg);
            a->xvfadd_s(partial_sum_vreg, partial_sum_vreg, out_vreg);
          }
        }
        // Reduce sum to 1 value
        // partial_sum[31:0] := w0+w2+...+w7
        a->xvbsrl_v(tmpReg, partial_sum_vreg, 4);
        a->xvfadd_s(partial_sum_vreg, partial_sum_vreg, tmpReg);
        a->xvbsrl_v(tmpReg, partial_sum_vreg, 8);
        a->xvfadd_s(partial_sum_vreg, partial_sum_vreg, tmpReg);
        a->xvpermi_d(float_step_vreg, partial_sum_vreg, 0x2);
        a->fadd_s(partial_sum_vreg, partial_sum_vreg, float_step_vreg);

        // This fragment moves block size (N) to stack and bcasts it to xmm reg
        mov_imm(a, tempGp, block_size);
        a->xvreplgr2vr_w(float_step_vreg, tempGp);
        a->ffint_s_w(float_step_vreg, float_step_vreg);   //int32 -> float

        // final_sum /= N
        a->fdiv_s(partial_sum_vreg, partial_sum_vreg, float_step_vreg);

        if (use_offsets) {
          a->ld_w(lengths_R, ptr(lengths, sizeof(offsetType)));
          a->ld_w(tempGp, ptr(lengths));
          a->sub_d(lengths_R, lengths_R, tempGp);
        } else {
          a->ld_w(lengths_R, ptr(lengths));
        }

        // Array out of bound check
        mov_imm(a, tempGp, sizeof(indxType));
        a->mul_d(scratchReg1, lengths_R, tempGp);

        a->add_d(scratchReg1, scratchReg1, indices);
        a->blt(index_size, scratchReg1, error);

        asmjit::Label LoopDataIndexBegin = a->newLabel();
        asmjit::Label LoopDataIndexEnd = a->newLabel();

        // dataIndex loop begins (iterate lengths_R_ times)
        a->bind(LoopDataIndexBegin);
        a->addi_d(lengths_R, lengths_R, -1);
        a->blt(lengths_R, la64::zero, LoopDataIndexEnd);

        // Array out of bound check
        if (areIndices64b) {
          a->ld_d(scratchReg1, ptr(indices));
        } else {
          a->ld_w(scratchReg1, ptr(indices));
        }

        // A trick to check x >= data_size or x < 0 in one shot by treating
        // scratchReg1_ as if it has unsigned value
        // (https://stackoverflow.com/a/34072155).
        a->bge(scratchReg1, data_size, error);
        a->blt(scratchReg1, la64::zero, error);

        if (prefetch) {
          asmjit::Label pref_dist_reset_start = a->newLabel();
          asmjit::Label pref_dist_reset_end = a->newLabel();
          // out of bound handling for prefetch
          a->add_d(scratchReg2, indices, la64::zero);
          mov_imm(a, tempGp, prefetch * sizeof(indxType));
          a->add_d(scratchReg2, scratchReg2, tempGp);
          a->bge(scratchReg2, index_size, pref_dist_reset_start);
          a->add_d(tempGp, indices, tempGp);
          if (areIndices64b) {
            a->ld_d(scratchReg2, la64::ptr(tempGp));
          } else {
            a->ld_w(scratchReg2, la64::ptr(tempGp));
          }

          a->b(pref_dist_reset_end);

          a->bind(pref_dist_reset_start);
          // things are not okay just get the current row
          // this can be improved to getting the max dist row.
          if (areIndices64b) {
            a->ld_d(scratchReg2, la64::ptr(indices));
          } else {
            a->ld_w(scratchReg2, la64::ptr(indices));
          }

          a->bind(pref_dist_reset_end);
        }

        a->addi_d(indices, indices, static_cast<asmjit::Imm>(sizeof(indxType)));

        if (prefetch) {
          a->slli_d(tempGp, scratchReg2, 2);
          a->add_d(tempGp, tempGp, h);
          a->preld(8, tempGp, 0);
        }
        // load h
        a->slli_d(tempGp, scratchReg1, 2);
        a->add_d(tempGp, tempGp, h);
        a->xvld(float_step_vreg, ptr(tempGp));
        // *h + final_sum
        a->fadd_s(float_step_vreg, float_step_vreg, partial_sum_vreg);
        // store h
        a->xvstelm_w(float_step_vreg, tempGp, 0, 0);
        // sqrt(hi)
        a->fsqrt_s(float_step_vreg, float_step_vreg);
        // bcast partial to all of ymm/zmm reg
        a->xvreplve0_w(float_step_vreg, float_step_vreg);
        // lr / sqrt(hi) + epsilon
        a->xvfadd_s(float_step_vreg, float_step_vreg, epsilon_vreg);
        a->xvfdiv_s(float_step_vreg, lr_vreg, float_step_vreg);

        mov_imm(a, tempGp, block_size);
        a->mul_d(scratchReg1, scratchReg1, tempGp);
        if (prefetch) {
          a->mul_d(scratchReg2, scratchReg2, tempGp);
        }

        for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
             vec_idx += unroll_factor) {
          int cur_unroll_factor =
              std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

          // The main computation
          for (int v = 0; v < cur_unroll_factor; ++v) {
            vec_reg_t out_vreg = vec_reg_t(v + first_available_vec_reg_id);

            auto g_ptr =
                la64::ptr(g, (vec_idx + v) * vlen * sizeof(float));
            if (!areWeightsFp16) { // float weights
              a->slli_d(temp_wptr, scratchReg1, 2);
              a->addi_d(temp_wptr, temp_wptr, (vec_idx + v) * vlen * sizeof(dataType));
              a->add_d(temp_wptr, temp_wptr, w);
              auto w_ptr = la64::ptr(temp_wptr);

              if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                if (instSet == inst_set_t::lasx) {
                  a->xvld(src_vreg, g_ptr);
                  a->xvand_v(src_vreg, src_vreg, mask_vreg);
                  a->xvfmul_s(src_vreg, float_step_vreg, src_vreg);

                  a->xvld(out_vreg, w_ptr);
                  a->xvand_v(out_vreg, out_vreg, mask_vreg);
                  a->xvfadd_s(out_vreg, src_vreg, out_vreg);

                  for(int st_idx = 0; st_idx < remainder; st_idx++){
                    a->xvstelm_w(out_vreg, temp_wptr, 0, st_idx);
                    a->addi_d(temp_wptr, temp_wptr, sizeof(float));
                  }
                }
              } else {
                a->xvld(tmpReg, g_ptr);
                a->xvfmul_s(out_vreg, float_step_vreg, tmpReg);
                a->xvld(tmpReg, w_ptr);
                a->xvfadd_s(out_vreg, out_vreg, tmpReg);
                a->xvst(out_vreg, w_ptr);
              }
            } else { // float16 weights
              a->slli_d(temp_wptr, scratchReg1, 1);
              a->addi_d(temp_wptr, temp_wptr, (vec_idx + v) * vlen * sizeof(dataType));
              a->add_d(temp_wptr, temp_wptr, w);
              auto w_ptr = la64::ptr(temp_wptr);

              if (use_stochastic_rounding) {
                // Index [0..3] for extracted bytes
                // Each int32 has 4 8-bit rand byte
                int sr_idx = (vec_idx + v) % 4;

                if (sr_idx == 0) {
                  // Generate R buffer every 4 steps of num_vec_regs_per_block
                  // loop. Each 8-bit in R (uint32_t) will be used once. It is
                  // shifted to the bits [5-13] then added to FP32 weights
                  // before FP16 conversion.
                  //
                  // The shifted 8 bit region
                  // +-------+--------+--------+--------+
                  // |       |        |   xxxxx|xxx     |
                  //  31      23       15       7      0
                  //
                  // Half float has 10 bits of mantissa, and float has 23, we
                  // are shifting the bits to cover the region where half
                  // floats can't represent data. This is bits[13..23] of the
                  // mantissa of FP32. This will be effectively adding a random
                  // variable of [0,1]

                  // Random generator using xoshiro128++
                  // Ref: http://prng.di.unimi.it/xoshiro128plusplus.c
                  a->xvadd_w(r0_vreg, S0_vreg, S3_vreg);
                  a->xvslli_w(r1_vreg, r0_vreg, 7);
                  a->xvsrli_w(r0_vreg, r0_vreg, 25);
                  if (instSet == inst_set_t::lasx) {
                    a->xvor_v(R_vreg, r0_vreg, r1_vreg);
                  }
                  a->xvadd_w(R_vreg, R_vreg, S0_vreg);

                  a->xvslli_w(r0_vreg, S1_vreg, 9);

                  if (instSet == inst_set_t::lasx) {

                    a->xvxor_v(S2_vreg, S2_vreg, S0_vreg);
                    a->xvxor_v(S3_vreg, S3_vreg, S1_vreg);
                    a->xvxor_v(S1_vreg, S1_vreg, S2_vreg);
                    a->xvxor_v(S0_vreg, S0_vreg, S3_vreg);

                    a->xvxor_v(S2_vreg, S2_vreg, r0_vreg);
                  }
                  a->xvslli_w(r0_vreg, S3_vreg, 11);
                  a->xvsrli_w(r1_vreg, S3_vreg, 21);
                  if (instSet == inst_set_t::lasx) {
                    a->xvor_v(S3_vreg, r0_vreg, r1_vreg);
                  }

                  // Extract byte 0 and shift to bits[5..13]
                  a->xvslli_w(r0_vreg, R_vreg, 24);
                  a->xvsrli_w(r0_vreg, r0_vreg, 19);
                } else if (sr_idx == 1) {
                  // Extract byte 1 and shift to bits[[5..13]
                  a->xvsrli_w(r0_vreg, R_vreg, 8);
                  a->xvslli_w(r0_vreg, r0_vreg, 24);
                  a->xvsrli_w(r0_vreg, r0_vreg, 19);
                } else if (sr_idx == 2) {
                  // Extract byte 2 and shift to bits[5..13]
                  a->xvslli_w(r0_vreg, R_vreg, 8);
                  a->xvsrli_w(r0_vreg, r0_vreg, 24);
                  a->xvslli_w(r0_vreg, r0_vreg, 5);
                } else { // sr_idx == 3
                  // Extract byte 3 and shift to bits[5..13]
                  a->xvsrli_w(r0_vreg, R_vreg, 24);
                  a->xvslli_w(r0_vreg, r0_vreg, 5);
                }
              }

              if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                if (instSet == inst_set_t::lasx) {
                  a->xvld(src_vreg, g_ptr);
                  a->xvand_v(src_vreg, src_vreg, mask_vreg);
                  a->addi_d(la64::sp, la64::sp, -8);
                  a->st_d(h, ptr(la64::sp));
                  a->addi_d(la64::sp, la64::sp, static_cast<int>(-vlen * sizeof(float16)));
                  a->slli_d(tempGp, scratchReg1, 1);
                  a->add_d(tempGp, tempGp, w);
                  for (int r = 0; r < remainder; ++r) {
                    a->ld_h(h, la64::ptr(tempGp, ((vec_idx + v) * vlen + r) * sizeof(dataType)));
                    a->st_h(h, la64::ptr(la64::sp, sizeof(dataType) * r));
                  }

                  a->xvld(out_vreg, la64::ptr(la64::sp));
                  a->xvpermi_d(out_vreg, out_vreg, 0x10);
                  a->xvfcvtl_s_h(out_vreg, out_vreg);
                  a->xvfmadd_s(out_vreg, float_step_vreg, src_vreg, out_vreg);
                  if (use_stochastic_rounding) {
                    a->xvadd_w(out_vreg, r0_vreg, out_vreg);
                  }
                  // Truncate rounding to 'counterwork' the random added part
                  a->xvpermi_q(tmpReg, out_vreg, 0x1);
                  mov_imm(a, tempGp, 0x100);
                  a->movfcsr2gr(tempGp2, la64::fcsr0);
                  a->or_(tempGp, tempGp2, tempGp);
                  a->movgr2fcsr(la64::fcsr0, tempGp);
                  a->xvfcvt_h_s(tmpReg, tmpReg, out_vreg);
                  a->movgr2fcsr(la64::fcsr0, tempGp2);
                  a->vst(tmpReg,la64::ptr(la64::sp));

                  a->slli_d(tempGp, scratchReg1, 1);
                  a->add_d(tempGp, tempGp, w);
                  // Copy results back
                  for (int r = 0; r < remainder; ++r) {
                    a->ld_h(h, la64::ptr(la64::sp, sizeof(dataType) * r));
                    a->st_h(h, la64::ptr(tempGp, ((vec_idx + v) * vlen + r) * sizeof(dataType)));
                  }
                  a->addi_d(la64::sp, la64::sp, static_cast<int>(vlen * sizeof(float16)));
                  a->ld_d(h, la64::ptr(la64::sp));
                  a->addi_d(la64::sp, la64::sp, 8);
                }
              } else {
                a->xvld(out_vreg, w_ptr);
                a->xvpermi_d(out_vreg, out_vreg, 0x10);
                a->xvfcvtl_s_h(out_vreg, out_vreg);
                a->xvld(tmpReg, g_ptr);
                a->xvfmadd_s(out_vreg, float_step_vreg, tmpReg, out_vreg);
                if (use_stochastic_rounding) {
                  a->xvadd_w(out_vreg, r0_vreg, out_vreg);
                }
                // Truncate rounding
                a->xvpermi_q(tmpReg, out_vreg, 0x1);
                mov_imm(a, tempGp, 0x100);
                a->movfcsr2gr(tempGp2, la64::fcsr0);
                a->or_(tempGp, tempGp2, tempGp);
                a->movgr2fcsr(la64::fcsr0, tempGp);
                a->xvfcvt_h_s(tmpReg, tmpReg, out_vreg);
                a->movgr2fcsr(la64::fcsr0, tempGp2);
                a->vst(tmpReg,w_ptr);
              }
            }

            constexpr int CACHE_LINE_LEN = 64;
            constexpr int BYTES_PER_VLOAD = vlen * sizeof(dataType);
            constexpr int VLOAD_PER_CACHE_LINE =
                CACHE_LINE_LEN / BYTES_PER_VLOAD;
            if (prefetch && (vec_idx + v) % VLOAD_PER_CACHE_LINE == 0) {
              a->slli_d(tempGp, scratchReg2, areWeightsFp16 ? 1 : 2);
              a->addi_d(tempGp, tempGp, (vec_idx + v) * BYTES_PER_VLOAD);
              a->add_d(tempGp, tempGp, w);
              a->preld(8, tempGp, 0);
            }
          }
        }

        a->b(LoopDataIndexBegin);
        a->bind(LoopDataIndexEnd);

        a->addi_d(lengths, lengths, static_cast<asmjit::Imm>(sizeof(offsetType)));
        mov_imm(a, tempGp, grad_stride * sizeof(float));
        a->add_d(g, g, tempGp);

        a->b(LoopRangeIndexBegin);
        a->bind(LoopRangeIndexEnd);

        a->bne(indices, index_size, error);
        mov_imm(a, scratchReg1, 1);
        a->b(exit);
        a->bind(error);
        mov_imm(a, scratchReg1, 0);
        a->bind(exit);

        if (areWeightsFp16 && use_stochastic_rounding) {
          if (instSet == inst_set_t::lasx) {
            a->xvst(S0_vreg,ptr(rand_buffer));
            a->xvst(S1_vreg,ptr(rand_buffer, 1 * vlen * sizeof(uint32_t)));
            a->xvst(S2_vreg,ptr(rand_buffer, 2 * vlen * sizeof(uint32_t)));
            a->xvst(S3_vreg,ptr(rand_buffer, 3 * vlen * sizeof(uint32_t)));
          }
        }

        a->add_w(la64::a0, scratchReg1, la64::zero);
        a->emitEpilog(frame);

        // jit_fused8bitembedding_kernel fn;
        typename ReturnFunctionSignature<indxType, offsetType, dataType>::
            jit_sparse_adagrad_kernel fn;
        asmjit::Error err;
        {
          unique_lock<mutex> lock(rtMutex_);
          err = runtime().add(&fn, &code);
        }
        if (err) {
          cout << "Error: in fn add" << endl;
          return nullptr;
        }

#if defined(FBGEMM_LOG_CODE)
        fclose(codeLogFile);
        delete codeLogger;
#endif
        return fn;
      });
} // getOrCreate

// Per-thread global buffer for random number generating, with max vector size
constexpr size_t VLEN_MAX = simd_info<inst_set_t::lasx>::WIDTH_32BIT_ELEMS;
alignas(64) static thread_local uint32_t g_rnd128v_buffer[4 * VLEN_MAX];
static thread_local bool g_rnd128v_initialized = false;

void rand_initialize() {
  // Splitmix64: http://prng.di.unimi.it/splitmix64.c
  auto rnd128_init_next = [](uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  };

  if (!g_rnd128v_initialized) {
    uint64_t h0 = std::hash<std::thread::id>{}(std::this_thread::get_id());
    for (auto i = 0; i < 4; ++i) {
      g_rnd128v_buffer[i * VLEN_MAX] = rnd128_init_next(h0);
      uint64_t h1 = g_rnd128v_buffer[i * VLEN_MAX];
      for (size_t v = 1; v < VLEN_MAX; ++v) {
        g_rnd128v_buffer[i * VLEN_MAX + v] = rnd128_init_next(h1);
      }
    }
    g_rnd128v_initialized = true;
  }
}

} // namespace

template <typename IndexType, typename OffsetType, typename DataType>
FBGEMM_API typename RowWiseSparseAdaGradFusedSignature<
    IndexType,
    OffsetType,
    DataType>::Type
GenerateRowWiseSparseAdaGradFused(
    int block_size, // number of parameters per row
    int prefetch,
    bool use_offsets,
    bool use_stochastic_rounding,
    int grad_stride) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if (grad_stride == -1) {
    grad_stride = block_size;
  }

  if (fbgemmHasLasxSupport()) {
    static GenRowWiseSparseAdagradFused<
        IndexType,
        OffsetType,
        DataType,
        inst_set_t::lasx>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        internal::lasx_ps_or_epi32_combined_mask,
        block_size,
        prefetch,
        use_offsets,
        use_stochastic_rounding,
        grad_stride);
    const auto lambda_func = [=](int64_t output_size,
                                 int64_t index_size,
                                 int64_t data_size,
                                 DataType* w,
                                 const float* g,
                                 float* h,
                                 const IndexType* indices,
                                 const OffsetType* offsets_or_lengths,
                                 float epsilon,
                                 float lr) {
      // Initialize random buffer in the first execution
      // TODO: JIT
      if (std::is_same<DataType, float16>::value && use_stochastic_rounding) {
        rand_initialize();
      }

      return original_func(
          output_size,
          index_size,
          data_size,
          w, // input/output parameters
          g, // input gradients
          h, // input/output momentums
          indices, // indices of each row
          offsets_or_lengths,
          epsilon,
          lr,
          g_rnd128v_buffer);
    };
    return lambda_func;
  } else {
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               DataType* w,
               const float* g,
               float* h,
               const IndexType* indices,
               const OffsetType* offsets_or_lengths,
               float epsilon,
               float lr) {
      return rowwise_sparse_adagrad_fused_ref(
          block_size,
          output_size,
          index_size,
          data_size,
          w,
          g,
          h,
          indices,
          offsets_or_lengths,
          epsilon,
          lr,
          use_offsets,
          use_stochastic_rounding,
          /*emu_vector_size=*/8,
          grad_stride);
    };
  }
}

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int64_t, int32_t, float>::Type
    GenerateRowWiseSparseAdaGradFused<int64_t, int32_t, float>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int64_t, int64_t, float>::Type
    GenerateRowWiseSparseAdaGradFused<int64_t, int64_t, float>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int32_t, int32_t, float>::Type
    GenerateRowWiseSparseAdaGradFused<int32_t, int32_t, float>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int32_t, int64_t, float>::Type
    GenerateRowWiseSparseAdaGradFused<int32_t, int64_t, float>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int64_t, int32_t, float16>::Type
    GenerateRowWiseSparseAdaGradFused<int64_t, int32_t, float16>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int64_t, int64_t, float16>::Type
    GenerateRowWiseSparseAdaGradFused<int64_t, int64_t, float16>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int32_t, int32_t, float16>::Type
    GenerateRowWiseSparseAdaGradFused<int32_t, int32_t, float16>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

template FBGEMM_API
    typename RowWiseSparseAdaGradFusedSignature<int32_t, int64_t, float16>::Type
    GenerateRowWiseSparseAdaGradFused<int32_t, int64_t, float16>(
        int block_size, // number of parameters per row
        int prefetch,
        bool use_offsets,
        bool use_stochastic_rounding,
        int grad_stride);

} // namespace fbgemm
