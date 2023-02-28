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
#include <cmath>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include "./CodeCache.h"
#include "./MaskLasx.h"
#include "./RefImplementations.h"
#include "fbgemm/Types.h"
#include "./CodeGenHelpers.h"

using namespace std;

namespace fbgemm {

namespace {

template <typename T>
T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

namespace la64 = asmjit::la64;

template <
    typename indxType,
    typename offsetType,
    typename outType,
    bool ROWWISE_SPARSE>
class ReturnFunctionSignature {};

template <typename indxType, typename offsetType, typename outType>
class ReturnFunctionSignature<indxType, offsetType, outType, false> {
 public:
  using jit_embedding_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t data_size,
      const uint8_t* input,
      const indxType* indices,
      const offsetType* offsets_or_lengths,
      const float* weights,
      outType* out,
      const int* mask);
};

template <typename indxType, typename offsetType, typename outType>
class ReturnFunctionSignature<indxType, offsetType, outType, true> {
 public:
  using jit_embedding_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t uncompressed_data_size,
      // int64_t compressed_data_size,
      const uint8_t* input,
      const indxType* indices,
      const offsetType* offsets_or_lengths,
      const float* weights,
      outType* out,
      const int32_t* compressed_indices_table,
      const int* mask);
};

template <
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE = false>
class GenEmbeddingSpMDMNBitLookup {
 public:
  GenEmbeddingSpMDMNBitLookup() {}
  typename ReturnFunctionSignature<
      indxType,
      offsetType,
      outType,
      ROWWISE_SPARSE>::jit_embedding_kernel
  getOrCreate(
      int bit_rate,
      int block_size,
      bool has_weight,
      bool is_weight_positional,
      bool normalize_by_lengths,
      int prefetch,
      bool use_offsets,
      int output_stride,
      int input_stride,
      bool scale_bias_last);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                  // depents on other static
                                  // variables.  Required to prevent
                                  // initialization order fiasco
    return rt;
  }

  static mutex rtMutex_; ///< Controll access to runtime;

  // The hash depends on bit_rate, embedding dimension (block size), weighted
  // sls, positional weights, normalize by lenths, prefetch distance,
  // use_offsets, output_stride, input_stride, and scale_bias_last
  static CodeCache<
      tuple<int, int, bool, bool, bool, int, bool, int, int, bool>,
      typename ReturnFunctionSignature<
          indxType,
          offsetType,
          outType,
          ROWWISE_SPARSE>::jit_embedding_kernel>
      codeCache_; ///< JIT Code Cache for reuse.
}; // GenEmbeddingSpmDMLookup

template <
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE>
mutex GenEmbeddingSpMDMNBitLookup<
    indxType,
    offsetType,
    outType,
    instSet,
    ROWWISE_SPARSE>::rtMutex_;

template <
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE>
CodeCache<
    tuple<int, int, bool, bool, bool, int, bool, int, int, bool>,
    typename ReturnFunctionSignature<
        indxType,
        offsetType,
        outType,
        ROWWISE_SPARSE>::jit_embedding_kernel>
    GenEmbeddingSpMDMNBitLookup<
        indxType,
        offsetType,
        outType,
        instSet,
        ROWWISE_SPARSE>::codeCache_;

template <
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE>
typename ReturnFunctionSignature<
    indxType,
    offsetType,
    outType,
    ROWWISE_SPARSE>::jit_embedding_kernel
GenEmbeddingSpMDMNBitLookup<
    indxType,
    offsetType,
    outType,
    instSet,
    ROWWISE_SPARSE>::
    getOrCreate(
        int bit_rate,
        int block_size,
        bool has_weight,
        bool is_weight_positional,
        bool normalize_by_lengths,
        int prefetch,
        bool use_offsets,
        int output_stride,
        int input_stride,
        bool scale_bias_last) {
  tuple<int, int, bool, bool, bool, int, bool, int, int, bool> kernelSig =
      make_tuple(
          bit_rate,
          block_size,
          has_weight,
          is_weight_positional,
          normalize_by_lengths,
          prefetch,
          use_offsets,
          output_stride,
          input_stride,
          scale_bias_last);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() -> typename ReturnFunctionSignature<
                indxType,
                offsetType,
                outType,
                ROWWISE_SPARSE>::jit_embedding_kernel {
        // TODO: Make this tunable
        int pref_dist = prefetch;
        bool areIndices64b = is_same<indxType, int64_t>::value;

        asmjit::CodeHolder code;
        code.init(runtime().environment());
        la64::Assembler assembler(&code);
        la64::Emitter* a = assembler.as<la64::Emitter>();
#if defined(FBGEMM_LOG_CODE)
        string filename = "embeddinglookup_" + to_string(bit_rate) + "bit";
        filename += "_emd_dim_" + to_string(block_size);
        filename += areIndices64b ? "_64bit" : "_32bit";
        filename += instSet == inst_set_t::lasx ? "_lasx" : "_loongarch";
        if (prefetch) {
          filename += "_prefetch";
        }
        if (has_weight) {
          filename += "_hasweight";
        }
        if (normalize_by_lengths) {
          filename += "_normalize_by_lengths";
        }
        if (!use_offsets) {
          filename += "_use_lengths";
        }
        if (ROWWISE_SPARSE) {
          filename += "_rowwise_sparse";
        }
        if (!scale_bias_last) {
          filename += "_scale_bias_first"
        }
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif
        // arguments to the function created
        la64::Gp output_size = la64::a0;
        // index_size will be overwritten to hold the end address of indices
        la64::Gp index_size = la64::a1;
        la64::Gp data_size = la64::a2;
        la64::Gp input = la64::a3;
        la64::Gp indices = la64::a4;
        la64::Gp lengths = la64::a5;
        la64::Gp weights = la64::a6;
        la64::Gp out = la64::a7;

        la64::Gp compressed_indices_table = la64::s0;
        la64::Gp scratchReg1_ = la64::s1;
        la64::Gp lengths_R_ = la64::s2;
        la64::Gp scratchReg2_ = la64::s3;
        la64::Gp temp_gp_0 = la64::s4;
        la64::Gp temp_gp_1 = la64::s5;

        asmjit::FuncDetail func;

        if (ROWWISE_SPARSE) {
          func.init(
              asmjit::FuncSignatureT<
                  bool,
                  int64_t, // output_size
                  int64_t, // index_size
                  int64_t, // uncompressed_data_size
                  const uint8_t*, // input uint8_t or float
                  const indxType*, // indices
                  const offsetType*, // offsets or lengths
                  const float*, // weights
                  float*, // out
                  const int32_t* /* compressed_indices_table */,
                  const int* /* mask */>(asmjit::CallConv::kIdHost),
              a->environment());
        } else {
          func.init(
              asmjit::FuncSignatureT<
                  bool,
                  int64_t, // output_size
                  int64_t, // index_size
                  int64_t, // data_size
                  const uint8_t*, // input uint8_t or float
                  const indxType*, // indices
                  const offsetType*, // offsets or lengths
                  const float*, // weights
                  float*, // out
                  const int* /* mask */>(asmjit::CallConv::kIdHost),
              a->environment());
        }

        asmjit::FuncFrame frame;
        frame.init(func);

        frame.setDirtyRegs(
            la64::Reg::kGroupVec,
            asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15) |
                asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
                asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31));

        frame.setDirtyRegs(
            la64::Reg::kGroupGp,
            asmjit::Support::bitMask(23, 24, 25, 26, 27, 28, 29, 30, 31));

        asmjit::FuncArgsAssignment args(&func);
        if (ROWWISE_SPARSE) {
          args.assignAll(
              output_size,
              index_size,
              data_size,
              input,
              indices,
              lengths,
              weights,
              out,
              compressed_indices_table,
              scratchReg1_);
        } else {
          args.assignAll(
              output_size,
              index_size,
              data_size,
              input,
              indices,
              lengths,
              weights,
              out,
              scratchReg1_);
        }

        args.updateFuncFrame(frame);
        frame.finalize();

        a->emitProlog(frame);
        a->emitArgsAssignment(frame, args);

        constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
        constexpr int NUM_VEC_REG = simd_info<instSet>::NUM_VEC_REGS;
        int unroll_factor = NUM_VEC_REG;

        typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

        int num_vec_regs_per_block = ceil_div(block_size, vlen);
        const int remainder = block_size % vlen;

        // Compute a remainder for vector load
        // Since every row is followed by 2 fp16 (scale and bias), luckily
        // we don't need mask at bit-rate granularity but just at 32-bit
        // granularity.
        int num_elem_per_32bit = 32 / bit_rate;
        // multiply by 4 because we're handling 4 vlen per iteration
        int num_of_32bit_per_vload = vlen * 4 / num_elem_per_32bit;
        int remainder_32bit_granularity =
            ceil_div(block_size, num_elem_per_32bit) % num_of_32bit_per_vload;

        // We need 2 vec registers for 1. scale 2. bias
        --unroll_factor;
        vec_reg_t scale_vreg = vec_reg_t(unroll_factor); // holds scale
        --unroll_factor;
        vec_reg_t bias_vreg = vec_reg_t(unroll_factor); // holds bias
        --unroll_factor;
        vec_reg_t src_vreg = vec_reg_t(unroll_factor); // for holding embedding value temporarily

        vec_reg_t w_vreg; // for weighted sls -- weights
        vec_reg_t vlen_inv_vreg; // used for normalize by lengths -- 1/ lengths[i]

        la64::VecX mask_vreg; // mask
        la64::VecV mask2_vreg;
        la64::VecV mask_fp16_vreg;

        // temporary register for bit manipulation instructions
        --unroll_factor;
        vec_reg_t temp_vreg = vec_reg_t(unroll_factor);
        --unroll_factor;
        vec_reg_t temp2_vreg = vec_reg_t(unroll_factor);

        // Create a mask that extracts lower bit_rate bits from each 8-bit block
        --unroll_factor;
        vec_reg_t extract_mask_vreg = vec_reg_t(unroll_factor);

        if (bit_rate == 4) {
          mov_imm(a, temp_gp_0, 0x0f0f);
          a->xvreplgr2vr_h(extract_mask_vreg, temp_gp_0);
        } else {
          mov_imm(a, temp_gp_0, 0x0303);
          a->xvreplgr2vr_h(extract_mask_vreg, temp_gp_0);
        }

        if (has_weight) {
          --unroll_factor;
          w_vreg = vec_reg_t(unroll_factor);
        }

        if (remainder && instSet == inst_set_t::lasx) {
          --unroll_factor;
          mask_vreg = la64::VecX(unroll_factor);

          if (remainder > 1 && std::is_same<outType, float16>::value) {
            --unroll_factor;
            mask_fp16_vreg = la64::VecV(unroll_factor);
          }
        }

        // Creating a mask for vector load
        if (remainder_32bit_granularity && instSet == inst_set_t::lasx) {
          --unroll_factor;
          mask2_vreg = la64::VecV(unroll_factor);
        }

        if (normalize_by_lengths) {
          --unroll_factor;
          vlen_inv_vreg = vec_reg_t(unroll_factor);
        }

        // Make unroll_factor a multiple of 4
        unroll_factor = unroll_factor / 4 * 4;

        if (remainder) {
          if (instSet == inst_set_t::lasx) {
            a->xvld(mask_vreg,
                  la64::ptr(scratchReg1_,
                            (vlen - remainder) % vlen * sizeof(int32_t)));
            if (std::is_same<outType, float16>::value) {
              if (remainder > 1) {
              }
              // We need to keep using the stack during the main loop
            }
          }
        }

        if (remainder_32bit_granularity) {
          if (instSet == inst_set_t::lasx) {
            a->vxor_v(mask2_vreg, mask2_vreg, mask2_vreg);
            mov_imm(a, temp_gp_0, -1);
            for (int i = 0; i < remainder_32bit_granularity; i++) {
              a->vinsgr2vr_w(mask2_vreg, temp_gp_0, i);
            }

          }
        }

        // Compute the end address of indices
        a->slli_d(temp_gp_0, index_size, areIndices64b ? 3:2);
        a->add_d(index_size, indices, temp_gp_0);

        asmjit::Label exit = a->newLabel();
        asmjit::Label error = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        // rangeIndex loop begins (iterate output_size times)
        a->bind(LoopRangeIndexBegin);
        a->addi_d(output_size, output_size, -1);
        a->blt(output_size, la64::zero, LoopRangeIndexEnd);

        if (normalize_by_lengths) {
          asmjit::Label IfLengthsBegin = a->newLabel();
          asmjit::Label IfLengthsEnd = a->newLabel();
          a->bind(IfLengthsBegin);

          // lengths_R_ : len
          if (use_offsets) {
            a->ld_w(lengths_R_, la64::ptr(lengths, sizeof(offsetType)));
            a->ld_w(temp_gp_0, la64::ptr(lengths));
            a->sub_d(lengths_R_, lengths_R_, temp_gp_0);
          } else {
            a->ld_w(lengths_R_, la64::ptr(lengths));
          }

          mov_imm(a, temp_gp_0, 1);
          // Initialize vlen_inv as 0 in case lengths is 0
          a->xvxor_v(vlen_inv_vreg, vlen_inv_vreg, vlen_inv_vreg);
          a->blt(lengths_R_, temp_gp_0, IfLengthsEnd);

          vec_reg_t temp_vreg0(0);
          // vlen_inv_vreg : float scale = 1.f / len
          if (instSet == inst_set_t::lasx) {
            mov_imm(a, scratchReg1_, 1);
            a->vinsgr2vr_d(vlen_inv_vreg, scratchReg1_, 0);
            a->ffint_s_l(vlen_inv_vreg, vlen_inv_vreg);
            a->vinsgr2vr_d(temp_vreg, lengths_R_, 0);
            a->ffint_s_l(temp_vreg, temp_vreg);
            a->fdiv_s(vlen_inv_vreg, vlen_inv_vreg, temp_vreg);
            a->xvreplve0_w(vlen_inv_vreg, vlen_inv_vreg);
          }
          a->bind(IfLengthsEnd);
        }

        for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
             vec_idx += unroll_factor) {
          int cur_unroll_factor =
              std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

          // Initialize output regs
          for (int v = 0; v < cur_unroll_factor; ++v) {
            vec_reg_t out_vreg = vec_reg_t(v);
            a->xvxor_v(out_vreg, out_vreg, out_vreg);
          }

          // lengths_R_ : len
          if (use_offsets) {
            a->ld_w(lengths_R_, la64::ptr(lengths, sizeof(offsetType)));
            a->ld_w(temp_gp_0, la64::ptr(lengths));
            a->sub_d(lengths_R_, lengths_R_, temp_gp_0);
          } else {
            a->ld_w(lengths_R_, la64::ptr(lengths));
          }

          // Array out of bound check:if (current + len > index_size)
          a->slli_d(temp_gp_0, lengths_R_, areIndices64b ? 3 : 2);
          a->add_d(scratchReg1_, temp_gp_0, indices);
          a->blt(index_size, scratchReg1_, error);

          asmjit::Label LoopDataIndexBegin = a->newLabel();
          asmjit::Label LoopDataIndexEnd = a->newLabel();
          asmjit::Label ValidIndexLabel = a->newLabel();

          // dataIndex loop begins (iterate lengths_R_ times)
          a->bind(LoopDataIndexBegin);
          a->addi_d(lengths_R_, lengths_R_, -1);
          a->blt(lengths_R_, la64::zero, LoopDataIndexEnd);

          // Array out of bound check
          // scratchReg1_ : idx = indices[current];
          if (areIndices64b) {
            a->ld_d(scratchReg1_, la64::ptr(indices));
          } else {
            a->ld_w(scratchReg1_, la64::ptr(indices));
          }

          if (!scale_bias_last) {
            // When scale_bias_last == false, assume this is for table batched
            // embedding (TBE) that can get -1 for pruned rows.
            mov_imm(a, temp_gp_0, -1);
            a->bne(scratchReg1_, temp_gp_0, ValidIndexLabel);
            a->addi_d(indices, indices, sizeof(indxType));
            a->b(LoopDataIndexBegin);

            a->bind(ValidIndexLabel);
          }
          // A trick to check x >= data_size or x < 0 in one shot by treating
          // scratchReg1_ as if it has unsigned value
          a->bge(scratchReg1_, data_size, error);

          if (ROWWISE_SPARSE) {
            a->slli_d(scratchReg1_, scratchReg1_, 2);
            a->add_d(scratchReg1_, scratchReg1_, compressed_indices_table);
            a->ld_w(scratchReg1_, la64::ptr(scratchReg1_));
          }

          int num_elem_per_byte = 8 / bit_rate;
          int fused_block_size = input_stride;
          if (pref_dist) {
            asmjit::Label pref_dist_reset_start = a->newLabel();
            asmjit::Label pref_dist_reset_end = a->newLabel();
            // out of bound handling for prefetch
            mov_imm(a, temp_gp_0, pref_dist * sizeof(indxType));
            a->add_d(scratchReg2_, temp_gp_0, indices);
            a->bge(scratchReg2_, index_size, pref_dist_reset_start);

            if (areIndices64b) {
              a->ld_d(scratchReg2_, la64::ptr(scratchReg2_));
            } else {
              a->ld_w(scratchReg2_, la64::ptr(scratchReg2_));
            }

            a->b(pref_dist_reset_end);

            a->bind(pref_dist_reset_start);
            // things are not okay just get the current row
            // this can be improved to getting the max dist row.
            if (areIndices64b) {
              a->ld_d(scratchReg2_, la64::ptr(indices));
            } else {
              a->ld_w(scratchReg2_, la64::ptr(indices));
            }

            a->bind(pref_dist_reset_end);
            if (ROWWISE_SPARSE) {
              asmjit::Label rowwise_sparse_pref_corner_case_begin =
                  a->newLabel();
              asmjit::Label rowwise_sparse_pref_corner_case_end = a->newLabel();
              a->bge(scratchReg2_, data_size, rowwise_sparse_pref_corner_case_begin);

              a->slli_d(scratchReg2_, scratchReg2_, 2);
              a->add_d(scratchReg2_, scratchReg2_, compressed_indices_table);
              a->ld_w(scratchReg2_, la64::ptr(scratchReg2_));
              a->and_(temp_gp_0, scratchReg2_, scratchReg2_);
              // Check negative
              a->bge(temp_gp_0, la64::zero, rowwise_sparse_pref_corner_case_end);

              a->bind(rowwise_sparse_pref_corner_case_begin);
              // For corner case, just set prefetch row id to 0.
              a->xor_(scratchReg2_, scratchReg2_, scratchReg2_);

              a->bind(rowwise_sparse_pref_corner_case_end);
            }
            // This has to be fused_block_size
            mov_imm(a, temp_gp_0, fused_block_size);
            a->mul_d(scratchReg2_, temp_gp_0, scratchReg2_);
          }

          a->addi_d(indices, indices, sizeof(indxType));

          if (has_weight) {
            a->xvldrepl_w(w_vreg, la64::ptr(weights));
            a->addi_d(weights, weights, sizeof(float));
          }

          if (ROWWISE_SPARSE) {
            mov_imm(a, temp_gp_0, -1);
            a->beq(scratchReg1_, temp_gp_0, LoopDataIndexBegin);
          }

          // scratchReg1_ : input_stride * idx
          mov_imm(a, temp_gp_0, fused_block_size);
          a->mul_d(scratchReg1_, temp_gp_0, scratchReg1_);

          // broadcast the scale
          int scale_offset =
              scale_bias_last ? ceil_div(block_size, num_elem_per_byte) : 0;
          mov_imm(a, temp_gp_0, scale_offset);
          a->add_d(temp_gp_0, temp_gp_0, input);
          a->add_d(temp_gp_0, temp_gp_0, scratchReg1_);
          a->xvldrepl_h(scale_vreg, la64::ptr(temp_gp_0));
          a->addi_d(temp_gp_0, temp_gp_0, sizeof(float16));
          a->xvldrepl_h(bias_vreg, la64::ptr(temp_gp_0));
          a->xvfcvtl_s_h(scale_vreg, scale_vreg);
          a->xvfcvtl_s_h(bias_vreg, bias_vreg);

          constexpr unsigned int CACHE_LINE_LEN = 64;
          if (pref_dist && fused_block_size % CACHE_LINE_LEN > 0 &&
              fused_block_size % CACHE_LINE_LEN <= 2 * sizeof(float16)) {
            mov_imm(a, temp_gp_0, fused_block_size / CACHE_LINE_LEN * CACHE_LINE_LEN);
            a->add_d(temp_gp_0, temp_gp_0, input);
            a->add_d(temp_gp_0, temp_gp_0, scratchReg2_);
            a->preld(0, temp_gp_0, 0);
          }

          if (has_weight) {
            a->xvfmul_s(scale_vreg, scale_vreg, w_vreg);
            a->xvfmul_s(bias_vreg, bias_vreg, w_vreg);
          }

          // The main computation
          int src_addr_offset = scale_bias_last ? 0 : 2 * sizeof(float16);
          for (int v = 0; v < cur_unroll_factor; v += 4) {
            int bytes_per_vload = (vlen / num_elem_per_byte) * sizeof(uint8_t);
            mov_imm(a, temp_gp_1, src_addr_offset + (vec_idx + v) * bytes_per_vload);
            a->add_d(temp_gp_1, temp_gp_1, input);
            a->add_d(temp_gp_1, temp_gp_1, scratchReg1_);

            if (bit_rate == 4) {
              if (num_vec_regs_per_block - (vec_idx + v) < 4 &&
                  remainder_32bit_granularity) {
                if (instSet == inst_set_t::avx512) {
                } else {
                  a->vld(src_vreg, la64::ptr(temp_gp_1));
                  a->vand_v(src_vreg.half(), src_vreg.half(), mask2_vreg);
                }
                a->vext2xv_hu_bu(src_vreg, src_vreg);
              } else {
                a->vld(src_vreg, la64::ptr(temp_gp_1));
                a->vext2xv_hu_bu(src_vreg, src_vreg);
              }
              a->xvslli_w(temp_vreg, src_vreg, 4);

              if (instSet == inst_set_t::avx512) {
              } else {
                a->xvor_v(src_vreg, src_vreg, temp_vreg);
                a->xvand_v(src_vreg, src_vreg, extract_mask_vreg);
              }
            } else {
              if (num_vec_regs_per_block - (vec_idx + v) < 4 &&
                  remainder_32bit_granularity) {
                if (instSet == inst_set_t::avx512) {
                } else {
                  a->xvld(src_vreg, la64::ptr(temp_gp_1));
                  a->vand_v(src_vreg.half(), src_vreg.half(), mask2_vreg);
                  a->vext2xv_wu_bu(src_vreg, src_vreg);
                }
              } else {
                a->xvld(src_vreg, la64::ptr(temp_gp_1));
                a->vext2xv_wu_bu(src_vreg, src_vreg);
              }

              a->xvslli_w(temp_vreg, src_vreg, 2 * 8 + 2);
              a->xvslli_w(temp2_vreg, src_vreg, 8 + 4);

              if (instSet == inst_set_t::avx512) {
              } else {
                a->xvor_v(temp_vreg, temp_vreg, temp2_vreg);
              }
              a->xvslli_w(temp2_vreg, src_vreg, 6);

              if (instSet == inst_set_t::avx512) {
              } else {
                a->xvor_v(temp_vreg, temp_vreg, temp2_vreg);
                a->xvor_v(src_vreg, temp_vreg, src_vreg);
                a->xvand_v(src_vreg, src_vreg, extract_mask_vreg);
              }
            }

            for (int i = 0;
                 i < std::min(4, num_vec_regs_per_block - (vec_idx + v));
                 ++i) {
              vec_reg_t out_vreg = vec_reg_t(v + i);
              if (i == 0) {
                a->vext2xv_w_b(temp_vreg, src_vreg);

                if (instSet == inst_set_t::lasx) {
                  a->xvbsll_v(temp2_vreg, src_vreg, 0);
                }
              } else {
                {
                  if (i == 1) {
                    a->xvbsrl_v(src_vreg, src_vreg, 8);
                  } else if (i == 2) {
                    if(1 & (i >> 1)){
                      // temp2_vreg[255:128] -> src_vreg[127:0]
                      a->xvpermi_d(src_vreg, temp2_vreg, 0x0e);
                    }else{
                      // temp2_vreg[127:0] -> src_vreg[127:0]
                      a->vbsll_v(src_vreg, temp2_vreg, 0);
                    }
                  } else {
                    a->xvbsrl_v(src_vreg, src_vreg, 8);
                  }
                  a->vext2xv_w_b(temp_vreg, src_vreg);

                } // lasx
              } // i > 0
              a->xvffint_s_w(temp_vreg, temp_vreg);
              a->xvfadd_s(out_vreg, out_vreg, bias_vreg);
              a->xvfmadd_s(out_vreg, temp_vreg, scale_vreg, out_vreg);
            } // for each i

            int vload_per_cache_line = CACHE_LINE_LEN / bytes_per_vload;
            int v_aligned = ceil_div(vec_idx + v, 4) * 4;
            if (pref_dist && v_aligned % vload_per_cache_line == 0) {
              mov_imm(a, temp_gp_0, v_aligned * bytes_per_vload);
              a->add_d(temp_gp_0, temp_gp_0, input);
              a->add_d(temp_gp_0, temp_gp_0, scratchReg2_);
              a->preld(0, temp_gp_0, 0);
            }
          }

          a->b(LoopDataIndexBegin);

          a->bind(LoopDataIndexEnd);

          // This loop is for writing back out_vreg (results)
          // back to memory
          for (int v = 0; v < cur_unroll_factor; ++v) {
            mov_imm(a, temp_gp_1, (vec_idx + v) * vlen * sizeof(outType));
            a->add_d(temp_gp_1, temp_gp_1, out);

            vec_reg_t out_vreg = vec_reg_t(v);

            if (normalize_by_lengths) {
              a->xvfmul_s(out_vreg, out_vreg, vlen_inv_vreg);
            }

            if (std::is_same<outType, float>::value) {
              if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                if (instSet == inst_set_t::avx512) {
                } else {
                  a->add_d(temp_gp_0, temp_gp_1, la64::zero);
                  for(int st_idx = 0; st_idx < remainder; st_idx++){
                    a->xvstelm_w(out_vreg, temp_gp_0, 0, st_idx);
                    a->addi_d(temp_gp_0, temp_gp_0, sizeof(float));
                  }
                }
              } else {
                a->xvst(out_vreg, la64::ptr(temp_gp_1));
              }
            } else {
              // fp16 output
              if (instSet == inst_set_t::lasx) {
                // round nearest with no exception
                a->xvpermi_d(temp_vreg, out_vreg, 0x0e);      // out_vreg[255:128] -> temp_xv_0[127:0]
                a->xvfcvt_h_s(out_vreg, temp_vreg, out_vreg); // float -> f16

                if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                  a->add_d(temp_gp_0, temp_gp_1, la64::zero);
                  for(int st_idx = 0; st_idx < remainder; st_idx++){
                    a->xvstelm_h(out_vreg, temp_gp_0, 0, st_idx);
                    a->addi_d(temp_gp_0, temp_gp_0, sizeof(outType));
                  }
                } else {
                  a->vst(out_vreg, la64::ptr(temp_gp_1));
                }
              }
            }
          }

          if (vec_idx + unroll_factor < num_vec_regs_per_block ||
              (has_weight && is_weight_positional)) {
            // Reset lengths_R_, indices, weights to run the dataIndex loop
            // again
            if (use_offsets) {
              a->ld_w(lengths_R_, la64::ptr(lengths, sizeof(offsetType)));
              a->ld_w(temp_gp_0, la64::ptr(lengths));
              a->sub_w(lengths_R_, lengths_R_, temp_gp_0);
            } else {
              a->ld_w(lengths_R_, la64::ptr(lengths));
            }

            if (has_weight) {
              mov_imm(a, temp_gp_0, sizeof(float));
              a->mul_d(scratchReg1_, lengths_R_, temp_gp_0);
              a->sub_d(weights, weights, scratchReg1_);

              if (vec_idx + unroll_factor < num_vec_regs_per_block) {
                mov_imm(a, temp_gp_0, sizeof(indxType) / sizeof(float));
                a->mul_d(scratchReg1_, scratchReg1_, temp_gp_0);
                a->sub_d(indices, indices, scratchReg1_);
              }
            } else {
              mov_imm(a, temp_gp_0, sizeof(indxType));
              a->mul_d(scratchReg1_, lengths_R_, temp_gp_0);
              a->sub_d(indices, indices, scratchReg1_);
            }
          }
        }

        a->addi_d(lengths, lengths, sizeof(offsetType));
        mov_imm(a, temp_gp_0, output_stride * sizeof(outType));
        a->add_d(out, out, temp_gp_0);

        a->b(LoopRangeIndexBegin);

        a->bind(LoopRangeIndexEnd);

        a->bne(indices, index_size, error);
        mov_imm(a, la64::a0, 1); // return true;
        a->b(exit);

        a->bind(error);
        mov_imm(a, la64::a0, 0);

        a->bind(exit);

        a->emitEpilog(frame);

        // jit_fused8bitembedding_kernel fn;
        typename ReturnFunctionSignature<
            indxType,
            offsetType,
            outType,
            ROWWISE_SPARSE>::jit_embedding_kernel fn;
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
}

} // namespace

template <typename indxType, typename offsetType, typename outType>
typename EmbeddingSpMDMKernelSignature<uint8_t, indxType, offsetType, outType>::
    Type
    GenerateEmbeddingSpMDMNBitWithStrides(
        int bit_rate,
        const int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch,
        bool is_weight_positional,
        bool use_offsets,
        int64_t output_stride /*=-1*/,
        int64_t input_stride /*=-1*/,
        bool scale_bias_last /*=true*/) {
  assert((bit_rate == 2 || bit_rate == 4) && "bit_rate must be 2 or 4");

  if (!cpuinfo_initialize()) {
    throw runtime_error("Failed to initialize cpuinfo!");
  }
  if (output_stride == -1) {
    output_stride = block_size;
  }
  if (input_stride == -1) {
    int64_t num_elem_per_byte = 8 / bit_rate;
    input_stride =
        ceil_div(block_size, num_elem_per_byte) + 2 * sizeof(float16);
  }

  if (fbgemmHasLasxSupport()) {
    static GenEmbeddingSpMDMNBitLookup<
        indxType,
        offsetType,
        outType,
        inst_set_t::lasx>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        bit_rate,
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets,
        output_stride,
        input_stride,
        scale_bias_last);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               outType* out) {
      return original_func(
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          internal::lasx_ps_or_epi32_combined_mask);
    };
  } else {
#ifdef VLOG
    VLOG(0) << "LASX not found, taking the slow path";
#endif
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               outType* out) {
      return EmbeddingSpMDMNBit_ref(
          bit_rate,
          block_size,
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          normalize_by_lengths,
          out,
          is_weight_positional,
          use_offsets,
          output_stride,
          input_stride,
          scale_bias_last);
    };
  }
}

template <typename IndexType, typename OffsetType, typename OutType>
FBGEMM_API typename EmbeddingSpMDMKernelSignature<
    std::uint8_t,
    IndexType,
    OffsetType,
    OutType>::Type
GenerateEmbeddingSpMDMNBit(
    int bit_rate,
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets) {
  return GenerateEmbeddingSpMDMNBitWithStrides<IndexType, OffsetType, OutType>(
      bit_rate,
      block_size,
      has_weight,
      normalize_by_lengths,
      prefetch,
      is_weight_positional,
      use_offsets);
}

template <typename indxType, typename offsetType>
typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    uint8_t,
    indxType,
    offsetType>::Type
GenerateEmbeddingSpMDMNBitRowWiseSparse(
    int bit_rate,
    const int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets) {
  assert((bit_rate == 2 || bit_rate == 4) && "bit_rate must be 2 or 4");

  if (!cpuinfo_initialize()) {
    throw runtime_error("Failed to initialize cpuinfo!");
  }
  int64_t num_elem_per_byte = 8 / bit_rate;
  int64_t input_stride =
      ceil_div(block_size, num_elem_per_byte) + 2 * sizeof(float16);

  if (fbgemmHasLasxSupport()) {
    static GenEmbeddingSpMDMNBitLookup<
        indxType,
        offsetType,
        /*outType=*/float,
        inst_set_t::lasx,
        /*rowwise_sparse=*/true>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        bit_rate,
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets,
        /*output_stride=*/block_size,
        input_stride,
        /*scale_bias_last=*/true);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t uncompressed_data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out,
               const int32_t* compressed_indices_table) {
      return original_func(
          output_size,
          index_size,
          uncompressed_data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          compressed_indices_table,
          internal::lasx_ps_or_epi32_combined_mask);
    };
  } else {
#ifdef VLOG
    VLOG(0) << "LASX not found, taking the slow path";
#endif
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t uncompressed_data_size,
               const uint8_t* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out,
               const int32_t* compressed_indices_table) {
      return EmbeddingSpMDMNBitRowWiseSparse_ref(
          bit_rate,
          block_size,
          output_size,
          index_size,
          uncompressed_data_size,
          // compressed_data_size,
          input,
          indices,
          compressed_indices_table,
          offsets_or_lengths,
          weights,
          normalize_by_lengths,
          out,
          is_weight_positional,
          use_offsets);
    };
  }
}

#define INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE)           \
  template FBGEMM_API typename EmbeddingSpMDMKernelSignature<               \
      uint8_t,                                                              \
      INDEX_TYPE,                                                           \
      OFFSET_TYPE,                                                          \
      OUT_TYPE>::Type                                                       \
  GenerateEmbeddingSpMDMNBit<INDEX_TYPE, OFFSET_TYPE, OUT_TYPE>(            \
      int bit_rate,                                                         \
      const int64_t block_size,                                             \
      bool has_weight,                                                      \
      bool normalize_by_lengths,                                            \
      int prefetch,                                                         \
      bool is_weight_positional,                                            \
      bool use_offsets);                                                    \
  template FBGEMM_API typename EmbeddingSpMDMKernelSignature<               \
      uint8_t,                                                              \
      INDEX_TYPE,                                                           \
      OFFSET_TYPE,                                                          \
      OUT_TYPE>::Type                                                       \
  GenerateEmbeddingSpMDMNBitWithStrides<INDEX_TYPE, OFFSET_TYPE, OUT_TYPE>( \
      int bit_rate,                                                         \
      const int64_t block_size,                                             \
      bool has_weight,                                                      \
      bool normalize_by_lengths,                                            \
      int prefetch,                                                         \
      bool is_weight_positional,                                            \
      bool use_offsets,                                                     \
      int64_t output_stride,                                                \
      int64_t input_stride,                                                 \
      bool scale_bias_last);

#define INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, OFFSET_TYPE)                   \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, float)                   \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE, float16)                 \
  template FBGEMM_API typename EmbeddingSpMDMRowWiseSparseKernelSignature< \
      uint8_t,                                                             \
      INDEX_TYPE,                                                          \
      OFFSET_TYPE>::Type                                                   \
  GenerateEmbeddingSpMDMNBitRowWiseSparse<INDEX_TYPE, OFFSET_TYPE>(        \
      int bit_rate,                                                        \
      const int64_t block_size,                                            \
      bool has_weight,                                                     \
      bool normalize_by_lengths,                                           \
      int prefetch,                                                        \
      bool is_weight_positional,                                           \
      bool use_offsets);

#define INSTANTIATE_SPMDM_OFFSET_T(INDEX_TYPE) \
  INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, int32_t) \
  INSTANTIATE_SPMDM_OUT_T(INDEX_TYPE, int64_t)

INSTANTIATE_SPMDM_OFFSET_T(int32_t)
INSTANTIATE_SPMDM_OFFSET_T(int64_t)

#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_OUT_T
#undef INSTANTIATE_SPMDM_BASE

} // namespace fbgemm
