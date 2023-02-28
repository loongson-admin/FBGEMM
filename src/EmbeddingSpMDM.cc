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

namespace fbgemm {

namespace {

namespace la64 = asmjit::la64;

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    bool ROWWISE_SPARSE>
class ReturnFunctionSignature {};

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType>
class ReturnFunctionSignature<inType, indxType, offsetType, outType, false> {
 public:
  using jit_embedding_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t data_size,
      const inType* input,
      const indxType* indices,
      const offsetType* offsets_or_lengths,
      const float* weights,
      outType* out,
      const int* mask);
};

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType>
class ReturnFunctionSignature<inType, indxType, offsetType, outType, true> {
 public:
  using jit_embedding_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t uncompressed_data_size,
      // int64_t compressed_data_size,
      const inType* input,
      const indxType* indices,
      const offsetType* offsets_or_lengths,
      const float* weights,
      outType* out,
      const int32_t* compressed_indices_table,
      const int* mask);
};

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE = false>
class GenEmbeddingSpMDMLookup {
 public:
  GenEmbeddingSpMDMLookup() {}
  typename ReturnFunctionSignature<
      inType,
      indxType,
      offsetType,
      outType,
      ROWWISE_SPARSE>::jit_embedding_kernel
  getOrCreate(
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

  static std::mutex rtMutex_; ///< Controll access to runtime;

  // The hash depends on embedding dimension (block size), weighted sls,
  // positional weights, normalize by lenths, prefetch distance, use_offsets,
  // output_stride, input_stride, and scale_bias_last
  static CodeCache<
      std::tuple<int, bool, bool, bool, int, bool, int, int, bool>,
      typename ReturnFunctionSignature<
          inType,
          indxType,
          offsetType,
          outType,
          ROWWISE_SPARSE>::jit_embedding_kernel>
      codeCache_; ///< JIT Code Cache for reuse.
}; // GenEmbeddingSpmDMLookup

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE>
std::mutex GenEmbeddingSpMDMLookup<
    inType,
    indxType,
    offsetType,
    outType,
    instSet,
    ROWWISE_SPARSE>::rtMutex_;

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE>
CodeCache<
    std::tuple<int, bool, bool, bool, int, bool, int, int, bool>,
    typename ReturnFunctionSignature<
        inType,
        indxType,
        offsetType,
        outType,
        ROWWISE_SPARSE>::jit_embedding_kernel>
    GenEmbeddingSpMDMLookup<
        inType,
        indxType,
        offsetType,
        outType,
        instSet,
        ROWWISE_SPARSE>::codeCache_;

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE>
typename ReturnFunctionSignature<
    inType,
    indxType,
    offsetType,
    outType,
    ROWWISE_SPARSE>::jit_embedding_kernel
GenEmbeddingSpMDMLookup<
    inType,
    indxType,
    offsetType,
    outType,
    instSet,
    ROWWISE_SPARSE>::
    getOrCreate(
        int block_size,
        bool has_weight,
        bool is_weight_positional,
        bool normalize_by_lengths,
        int prefetch,
        bool use_offsets,
        int output_stride,
        int input_stride,
        bool scale_bias_last) {
  std::tuple<int, bool, bool, bool, int, bool, int, int, bool> kernelSig =
      std::make_tuple(
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
                inType,
                indxType,
                offsetType,
                outType,
                ROWWISE_SPARSE>::jit_embedding_kernel {
        bool is8bit = std::is_same<inType, uint8_t>::value;
        bool is16bit = std::is_same<inType, float16>::value;

        // TODO: Make this tunable
        int pref_dist = prefetch;
        bool areIndices64b = std::is_same<indxType, int64_t>::value;

        asmjit::CodeHolder code;
        code.init(runtime().environment());
        la64::Assembler assembler(&code);
        la64::Emitter* a = assembler.as<la64::Emitter>();
#if defined(FBGEMM_LOG_CODE)
        std::string filename = "embeddinglookup";
        if (is8bit) {
          filename += "_8bit";
        } else if (is16bit) {
          filename += "_fp16";
        }
        filename += "_emd_dim_" + std::to_string(block_size);
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
        filename += "_out_stride_" + std::to_string(output_stride);
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
                  const inType*, // input uint8_t or float
                  const indxType*, // indices
                  const offsetType*, // offsets or lengths
                  const float*, // weights
                  outType*, // out
                  const int32_t*, // compressed_indices_table and then mask
                  const int*>(asmjit::CallConv::kIdHost),
              a->environment());
        } else {
          func.init(
              asmjit::FuncSignatureT<
                  bool,
                  int64_t, // output_size
                  int64_t, // index_size
                  int64_t, // data_size
                  const inType*, // input uint8_t or float
                  const indxType*, // indices
                  const offsetType*, // offsets or lengths
                  const float*, // weights
                  outType*, // out and then mask
                  const int*>(asmjit::CallConv::kIdHost),
              a->environment());
        }

        asmjit::FuncFrame frame;
        frame.init(func);

        if (instSet == inst_set_t::lasx) {
          frame.setDirtyRegs(
              la64::Reg::kGroupVec,
              asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                  asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15) |
                  asmjit::Support::bitMask(16,17, 18, 19, 20, 21, 22, 23) |
                  asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31) );
        }

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

        int num_vec_regs_per_block = (block_size + vlen - 1) / vlen;
        int remainder = block_size % vlen;

        vec_reg_t scale_vreg; // holds scale
        vec_reg_t bias_vreg; // holds bias
        vec_reg_t w_vreg; // for weighted sls -- weights
        vec_reg_t
            vlen_inv_vreg; // used for normalize by lengths -- 1/ lengths[i]
        vec_reg_t src_vreg; // for holding embedding value temporarily
        la64::VecX mask_vreg;

        --unroll_factor;
        vec_reg_t temp_xv_0 = vec_reg_t(unroll_factor);

        if (is8bit) {
          // We need 2 vec registers for 1. scale 2. bias
          --unroll_factor;
          scale_vreg = vec_reg_t(unroll_factor);
          --unroll_factor;
          bias_vreg = vec_reg_t(unroll_factor);
        }

        if (is8bit || is16bit || (remainder && instSet == inst_set_t::lasx)) {
          --unroll_factor;
          src_vreg = vec_reg_t(unroll_factor);
        }

        if (has_weight) {
          --unroll_factor;
          w_vreg = vec_reg_t(unroll_factor);
        }

        if (remainder && instSet == inst_set_t::lasx) {
          --unroll_factor;
          mask_vreg = la64::VecX(unroll_factor);
        }

        if (normalize_by_lengths) {
          --unroll_factor;
          vlen_inv_vreg = vec_reg_t(unroll_factor);
        }

        if (remainder) {
          if (instSet == inst_set_t::lasx) {
            a->xvld(mask_vreg,
                    la64::ptr(scratchReg1_, (vlen - remainder) % vlen * sizeof(int32_t)));

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

          // OK to use vreg0 because it's for out_vreg used in the main loop
          vec_reg_t temp_vreg = vec_reg_t(0);
          if (instSet == inst_set_t::lasx) {
            // vlen_inv_vreg : float scale = 1.f / len

            mov_imm(a, scratchReg1_, 1);
            a->vinsgr2vr_d(temp_xv_0, scratchReg1_, 0);
            a->ffint_s_l(vlen_inv_vreg, temp_xv_0);
            a->vinsgr2vr_d(temp_xv_0, lengths_R_, 0);
            a->ffint_s_l(temp_vreg, temp_xv_0);
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

          if (use_offsets) {
            a->ld_w(lengths_R_, la64::ptr(lengths, sizeof(offsetType)));
            a->ld_w(temp_gp_0, la64::ptr(lengths));
            a->sub_d(lengths_R_, lengths_R_, temp_gp_0);
          } else {
            a->ld_w(lengths_R_, la64::ptr(lengths));
          }

          // Array out of bound check
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
            // scratchReg1_ : idx = compressed_indices_table[uncompressed_idx]
            a->slli_d(scratchReg1_, scratchReg1_, 2);
            a->add_d(scratchReg1_, scratchReg1_, compressed_indices_table);
            a->ld_w(scratchReg1_, la64::ptr(scratchReg1_));
          }

          int fused_block_size = input_stride * sizeof(inType);

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
              asmjit::Label rowwise_sparse_pref_corner_case_begin = a->newLabel();
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
            mov_imm(a, temp_gp_0, fused_block_size);
            a->mul_d(scratchReg2_, temp_gp_0, scratchReg2_);
          }

          a->addi_d(indices, indices, sizeof(indxType));

          if (has_weight) {
            a->xvldrepl_w(w_vreg, la64::ptr(weights));
            a->addi_d(weights, weights, sizeof(float));
          }

          if (ROWWISE_SPARSE) {
            // if (idx == -1)
            mov_imm(a, temp_gp_0, -1);
            a->beq(scratchReg1_, temp_gp_0, LoopDataIndexBegin);
          }

          // scratchReg1_ : fused_block_size * idx
          mov_imm(a, temp_gp_0, fused_block_size);
          a->mul_d(scratchReg1_, temp_gp_0, scratchReg1_);

          // broadcast the scale
          constexpr unsigned int CACHE_LINE_LEN = 64;
          if (is8bit) {
            if (scale_bias_last) {
              mov_imm(a, temp_gp_0, block_size * sizeof(uint8_t));
              a->add_d(temp_gp_0, temp_gp_0, scratchReg1_);
              a->add_d(temp_gp_0, temp_gp_0, input);
              a->xvldrepl_w(scale_vreg, la64::ptr(temp_gp_0));
              a->addi_d(temp_gp_0, temp_gp_0, sizeof(float));
              a->xvldrepl_w(bias_vreg, la64::ptr(temp_gp_0));
            } else {
              a->add_d(temp_gp_0, input, scratchReg1_);
              a->xvldrepl_h(scale_vreg, la64::ptr(temp_gp_0));
              a->addi_d(temp_gp_0, temp_gp_0, sizeof(float16));
              a->xvldrepl_h(bias_vreg, la64::ptr(temp_gp_0));
              a->xvfcvtl_s_h(scale_vreg, scale_vreg);
              a->xvfcvtl_s_h(bias_vreg, bias_vreg);
            }

            if (pref_dist && fused_block_size % CACHE_LINE_LEN > 0 &&
                fused_block_size % CACHE_LINE_LEN <= 2 * sizeof(float)) {
              mov_imm(a, temp_gp_0, fused_block_size / CACHE_LINE_LEN * CACHE_LINE_LEN);
              a->add_d(temp_gp_0, temp_gp_0, input);
              a->add_d(temp_gp_0, temp_gp_0, scratchReg2_);
              a->preld(0, temp_gp_0, 0);
            }
          }

          if (has_weight && is8bit) {
            a->xvfmul_s(scale_vreg, scale_vreg, w_vreg);
            a->xvfmul_s(bias_vreg, bias_vreg, w_vreg);
          }

          // The main computation
          int src_addr_offset = is8bit && !scale_bias_last ? 2 * sizeof(float16) : 0;
          for (int v = 0; v < cur_unroll_factor; ++v) {
            constexpr int BYTES_PER_VLOAD = vlen * sizeof(inType);
            mov_imm(a, temp_gp_1, src_addr_offset + (vec_idx + v) * BYTES_PER_VLOAD);
            a->add_d(temp_gp_1, temp_gp_1, input);
            a->add_d(temp_gp_1, temp_gp_1, scratchReg1_);

            vec_reg_t out_vreg = vec_reg_t(v);

            // For 8bit SLS convert usigned 8-bit to 32bit int, then to float
            // multiply with scale and then add with bias
            if (is8bit) {
              if (remainder && vec_idx + v == num_vec_regs_per_block - 1 &&
                  instSet == inst_set_t::avx512) {
              } else {
                // We don't use a mask for AVX2 since we can use the extra
                // "padding" of the 2 floats (= 8 chars) scale and bias
                // this ensures we never access out of bound data
                a->xvld(temp_xv_0, la64::ptr(temp_gp_1));   // load 64bit
                a->vext2xv_wu_bu(src_vreg, temp_xv_0);
              }
              a->xvffint_s_wu(src_vreg, src_vreg);
              a->xvfadd_s(out_vreg, out_vreg, bias_vreg);
              a->xvfmadd_s(out_vreg, src_vreg, scale_vreg, out_vreg);
            } else if (is16bit) {
              if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                if (instSet == inst_set_t::lasx) {
                  // load remainder * 16 bit
                  a->vld(src_vreg, la64::ptr(temp_gp_1));
                  // get mask float16 verg
                  a->vxor_v(temp_xv_0, temp_xv_0, temp_xv_0);
                  mov_imm(a, temp_gp_0, 0xffffffff);
                  for(int ld_idx = 0; ld_idx < remainder; ld_idx++){
                    a->vinsgr2vr_h(temp_xv_0, temp_gp_0, ld_idx);
                  }
                  a->vand_v(src_vreg, src_vreg, temp_xv_0);

                  a->xvpermi_d(src_vreg, src_vreg, 0x10);   // [64:127] -> [128:191]
                  a->xvfcvtl_s_h(src_vreg, src_vreg);
                }
              } else {
                // no remainder
                a->vld(temp_xv_0, la64::ptr(temp_gp_1));
                a->xvpermi_d(temp_xv_0, temp_xv_0, 0x10);   // [64:127] -> [128:191]
                a->xvfcvtl_s_h(src_vreg, temp_xv_0);
              }
              if (has_weight) {
                a->xvfmadd_s(out_vreg, w_vreg, src_vreg, out_vreg);
              } else {
                a->xvfadd_s(out_vreg, out_vreg, src_vreg);
              }
            } else {
              // This part for FP32 SLS
              if (remainder && vec_idx + v == num_vec_regs_per_block - 1 &&
                  instSet == inst_set_t::lasx) {
                a->xvld(src_vreg, la64::ptr(temp_gp_1));
                a->xvand_v(src_vreg, src_vreg, mask_vreg);
              }
              if (has_weight) {
                if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                  if (instSet == inst_set_t::lasx) {
                    a->xvfmadd_s(out_vreg, w_vreg, src_vreg, out_vreg);
                  }
                } else {
                  a->xvld(temp_xv_0, la64::ptr(temp_gp_1));
                  a->xvfmadd_s(out_vreg, w_vreg, temp_xv_0, out_vreg);
                }
              } else {
                if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                  if (instSet == inst_set_t::lasx) {
                    a->xvfadd_s(out_vreg, out_vreg, src_vreg);
                  }
                } else {
                  a->xvld(temp_xv_0, la64::ptr(temp_gp_1));
                  a->xvfadd_s(out_vreg, out_vreg, temp_xv_0);
                }
              }
            }

            constexpr int VLOAD_PER_CACHE_LINE =
                CACHE_LINE_LEN / BYTES_PER_VLOAD;
            if (pref_dist && (vec_idx + v) % VLOAD_PER_CACHE_LINE == 0) {
              mov_imm(a, temp_gp_0, (vec_idx + v) * BYTES_PER_VLOAD);
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
                if (instSet == inst_set_t::lasx) {
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
                a->xvpermi_d(temp_xv_0, out_vreg, 0x0e);      // out_vreg[255:128] -> temp_xv_0[127:0]
                a->xvfcvt_h_s(out_vreg, temp_xv_0, out_vreg); // float -> f16

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
            inType,
            indxType,
            offsetType,
            outType,
            ROWWISE_SPARSE>::jit_embedding_kernel fn;
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
        fclose(codeLogFile);
        delete codeLogger;
#endif
        return fn;
      });
}

} // namespace

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType>
typename EmbeddingSpMDMKernelSignature<inType, indxType, offsetType, outType>::
    Type
    GenerateEmbeddingSpMDMWithStrides(
        const int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch,
        bool is_weight_positional,
        bool use_offsets,
        int64_t output_stride /*=-1*/,
        int64_t input_stride /*=-1*/,
        bool scale_bias_last /*=true*/) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if (output_stride == -1) {
    output_stride = block_size;
  }
  if (input_stride == -1) {
    if (std::is_same<inType, uint8_t>::value) {
      const auto scale_bias_offset =
          2 * (scale_bias_last ? sizeof(float) : sizeof(float16));
      input_stride = block_size + scale_bias_offset;
    } else {
      input_stride = block_size;
    }
  }
  const inst_set_t isa = fbgemmInstructionSet();
  if ((std::is_same<inType, float>::value ||
       std::is_same<inType, float16>::value) &&
      block_size == 1 && isa == inst_set_t::lasx && output_stride == block_size &&
      input_stride == block_size && std::is_same<outType, float>::value) {
    return
        [=](int64_t output_size,
            int64_t index_size,
            int64_t data_size,
            const inType* input,
            const indxType* indices,
            const offsetType* offsets_or_lengths,
            const float* weights, // optional, can be null for non-weighted sum
            outType* out) {
          return internal::EmbeddingSpMDMBlockSize1_(
              output_size,
              index_size,
              data_size,
              input,
              indices,
              offsets_or_lengths,
              weights,
              normalize_by_lengths,
              reinterpret_cast<float*>(out),
              is_weight_positional,
              use_offsets);
        };
  } else if(isa == inst_set_t::lasx){
    static GenEmbeddingSpMDMLookup<
        inType,
        indxType,
        offsetType,
        outType,
        inst_set_t::lasx>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
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
               const inType* input,
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
               const inType* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               outType* out) {
      return EmbeddingSpMDM_ref(
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

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType>
typename EmbeddingSpMDMKernelSignature<inType, indxType, offsetType, outType>::
    Type
    GenerateEmbeddingSpMDM(
        const int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch,
        bool is_weight_positional,
        bool use_offsets) {
  return GenerateEmbeddingSpMDMWithStrides<
      inType,
      indxType,
      offsetType,
      outType>(
      block_size,
      has_weight,
      normalize_by_lengths,
      prefetch,
      is_weight_positional,
      use_offsets);
}

template <typename inType, typename indxType, typename offsetType>
typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    inType,
    indxType,
    offsetType>::Type
GenerateEmbeddingSpMDMRowWiseSparse(
    const int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  int64_t input_stride = block_size;
  if (std::is_same<inType, uint8_t>::value) {
    const auto scale_bias_offset = 2 * sizeof(float);
    input_stride = block_size + scale_bias_offset;
  }
  inst_set_t isa = fbgemmInstructionSet();
  if (isa == inst_set_t::lasx){
    static GenEmbeddingSpMDMLookup<
        inType,
        indxType,
        offsetType,
        /*outType=*/float,
        inst_set_t::lasx,
        /*rowwise_sparse=*/true>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
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
               const inType* input,
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
    return
        [=](int64_t output_size,
            int64_t index_size,
            int64_t uncompressed_data_size,
            const inType* input,
            const indxType* indices,
            const offsetType* offsets_or_lengths,
            const float* weights, // optional, can be null for non-weighted sum
            float* out,
            const int32_t* compressed_indices_table) {
          return EmbeddingSpMDMRowWiseSparse_ref(
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

#define INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  template FBGEMM_API typename EmbeddingSpMDMKernelSignature<              \
      IN_TYPE,                                                             \
      INDEX_TYPE,                                                          \
      OFFSET_TYPE,                                                         \
      OUT_TYPE>::Type                                                      \
  GenerateEmbeddingSpMDMWithStrides<                                       \
      IN_TYPE,                                                             \
      INDEX_TYPE,                                                          \
      OFFSET_TYPE,                                                         \
      OUT_TYPE>(                                                           \
      const int64_t block_size,                                            \
      bool has_weight,                                                     \
      bool normalize_by_lengths,                                           \
      int prefetch,                                                        \
      bool is_weight_positional,                                           \
      bool use_offsets,                                                    \
      int64_t output_stride,                                               \
      int64_t input_stride,                                                \
      bool scale_bias_last);                                               \
  template FBGEMM_API typename EmbeddingSpMDMKernelSignature<              \
      IN_TYPE,                                                             \
      INDEX_TYPE,                                                          \
      OFFSET_TYPE,                                                         \
      OUT_TYPE>::Type                                                      \
  GenerateEmbeddingSpMDM<IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE>(      \
      const int64_t block_size,                                            \
      bool has_weight,                                                     \
      bool normalize_by_lengths,                                           \
      int prefetch,                                                        \
      bool is_weight_positional,                                           \
      bool use_offsets);

#define INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, OFFSET_TYPE)          \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, float)          \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, float16)        \
  template FBGEMM_API typename EmbeddingSpMDMRowWiseSparseKernelSignature< \
      IN_TYPE,                                                             \
      INDEX_TYPE,                                                          \
      OFFSET_TYPE>::Type                                                   \
  GenerateEmbeddingSpMDMRowWiseSparse<IN_TYPE, INDEX_TYPE, OFFSET_TYPE>(   \
      const int64_t block_size,                                            \
      bool has_weight,                                                     \
      bool normalize_by_lengths,                                           \
      int prefetch,                                                        \
      bool is_weight_positional,                                           \
      bool use_offsets);

#define INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, INDEX_TYPE) \
  INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, int32_t) \
  INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, int64_t)

#define INSTANTIATE_SPMDM_INDEX_T(IN_TYPE)     \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, int32_t) \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, int64_t)

INSTANTIATE_SPMDM_INDEX_T(float)
INSTANTIATE_SPMDM_INDEX_T(float16)
INSTANTIATE_SPMDM_INDEX_T(uint8_t)

#undef INSTANTIATE_SPMDM_INDEX_T
#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_OUT_T
#undef INSTANTIATE_SPMDM_BASE

template <typename IndexType>
void compressed_indices_remap(
    std::int32_t offsets_len,
    const IndexType* indices,
    const int32_t* compressed_indices_mapping,
    const IndexType* offsets,
    const float* weights, // optional, can be null,
    IndexType* out_indices,
    IndexType* out_offsets,
    float* out_weights) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  // const inst_set_t isa = fbgemmInstructionSet();

  {
    compressed_indices_remap_ref<IndexType>(
        offsets_len,
        indices,
        compressed_indices_mapping,
        offsets,
        weights,
        out_indices,
        out_offsets,
        out_weights);
  }
}

#define INSTANTIATE_REMAP_BASE(INDEX_TYPE)           \
  template FBGEMM_API void compressed_indices_remap( \
      std::int32_t offsets_numel,                    \
      const INDEX_TYPE* indices,                     \
      const int32_t* compressed_indices_mapping,     \
      const INDEX_TYPE* offsets,                     \
      const float* weights,                          \
      INDEX_TYPE* out_indices,                       \
      INDEX_TYPE* out_offsets,                       \
      float* out_weights);

INSTANTIATE_REMAP_BASE(int32_t)
INSTANTIATE_REMAP_BASE(int64_t)

#undef INSTANTIATE_REMAP_BASE

} // namespace fbgemm
