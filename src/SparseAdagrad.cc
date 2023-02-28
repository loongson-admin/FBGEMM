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
#include <cmath>
#include <iostream>
#include <mutex>
#include <string>
#include <tuple>
#include "./CodeCache.h"
#include "./MaskLasx.h"
#include "./RefImplementations.h"
#include "fbgemm/Utils.h"
#include "./CodeGenHelpers.h"

namespace fbgemm {

namespace {
namespace la64 = asmjit::la64;

template <typename indxType = std::int64_t>
class ReturnFunctionSignature {
 public:
  using jit_sparse_adagrad_kernel = int (*)(
      int num_rows, // number of rows reading
      std::uint64_t param_size, // total number of parameters
      float* w, // input/output parameters
      const float* g, // input gradients
      float* h, // input/output momentums
      const indxType* indices, // indices of each row
      float epsilon,
      float lr,
      const int* mask_lasx,
      float weight_decay,
      const double* counter,
      std::int64_t counter_halflife);
};

template <
    typename indxType = std::int64_t,
    inst_set_t instSet = inst_set_t::lasx>
class GenSparseAdagrad {
 public:
  GenSparseAdagrad() {}
  void genSparseAdagrad(
      la64::Emitter* a,
      int unroll_factor,
      int num_vec_regs_per_block,
      int remainder,
      int prefetch,
      typename simd_info<instSet>::vec_reg_t epsilon_vreg,
      typename simd_info<instSet>::vec_reg_t lr_vreg,
      la64::VecX mask_vreg,
      typename simd_info<instSet>::vec_reg_t temp_vreg,
      typename simd_info<instSet>::vec_reg_t weight_decay_vreg,
      bool has_weight_decay);

  void genRowwiseSparseAdagrad(
      la64::Emitter* a,
      int block_size,
      int unroll_factor,
      int num_vec_regs_per_block,
      int remainder,
      int prefetch,
      typename simd_info<instSet>::vec_reg_t epsilon_vreg,
      typename simd_info<instSet>::vec_reg_t lr_vreg,
      la64::VecX mask_vreg,
      typename simd_info<instSet>::vec_reg_t temp_vreg,
      typename simd_info<instSet>::vec_reg_t weight_decay_vreg,
      bool has_weight_decay);

  typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
  getOrCreate(
      int block_size,
      int prefetch,
      bool rowwise,
      bool has_weight_decay);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; // JIT Runtime for asmjit
    return rt;
  }

  static std::mutex rtMutex_; /// Controll access to runtime;

  // The hash depends on embedding dimension (block size), prefetch distance,
  // rowwise, and has_weight_decay
  static CodeCache<
      std::tuple<int, int, bool, bool>,
      typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel>
      codeCache_; ///< JIT Code Cache for reuse.

  // These are register we share accross SparseAdagrad and RowwiseSparseAdagrad

  la64::Gp w;
  la64::Gp g;
  la64::Gp h;
  la64::Gp indices;
  la64::Gp base_offset;
  la64::Gp temp1_; // loop counter
  la64::Gp temp2_; // prefetch offset
  la64::Gp temp3_; // prefetch offset of moment in rowwise adagrad
  la64::Gp temp_gp; // add
  la64::Gp index_offset; //add
  la64::VecX temp0_xv; // add
  la64::VecX temp1_xv; // add

}; // GenEmbeddingLookup

template <typename indxType, inst_set_t instSet>
std::mutex GenSparseAdagrad<indxType, instSet>::rtMutex_;

template <typename indxType, inst_set_t instSet>
CodeCache<
    std::tuple<int, int, bool, bool>,
    typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel>
    GenSparseAdagrad<indxType, instSet>::codeCache_;

template <typename indxType, inst_set_t instSet>
void GenSparseAdagrad<indxType, instSet>::genSparseAdagrad(
    la64::Emitter* a,
    int unroll_factor,
    int num_vec_regs_per_block,
    int remainder,
    int prefetch,
    typename simd_info<instSet>::vec_reg_t epsilon_vreg,
    typename simd_info<instSet>::vec_reg_t lr_vreg,
    la64::VecX mask_vreg,
    typename simd_info<instSet>::vec_reg_t temp_vreg,
    typename simd_info<instSet>::vec_reg_t weight_decay_vreg,
    bool has_weight_decay) {
  typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;
  constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
  for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
       vec_idx += unroll_factor) {
    int cur_unroll_factor =
        std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

    for (int v = 0; v < cur_unroll_factor; ++v) {
      mov_imm(a, index_offset, (vec_idx + v) * vlen * sizeof(float));

      vec_reg_t out_vreg = vec_reg_t(v);
      vec_reg_t g_vreg = vec_reg_t(v + cur_unroll_factor);

      if (prefetch && ((vec_idx + v) % (64 / (vlen * sizeof(float))) == 0)) {
        // Intel SDE (wrongly) thinks prefetchwt1 is not available in BDW
        a->add_d(temp_gp, h, temp2_);
        a->add_d(temp_gp, temp_gp, index_offset);
        a->preld(8, temp_gp, 0);

        a->add_d(temp_gp, w, temp2_);
        a->add_d(temp_gp, temp_gp, index_offset);
        a->preld(8, temp_gp, 0);
      }

      if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
          if (instSet == inst_set_t::lasx) {
          a->add_d(temp_gp, g, index_offset);
          a->xvld(g_vreg, la64::ptr(temp_gp));
          a->xvand_v(g_vreg, g_vreg, mask_vreg);

          if (has_weight_decay) {
            // TODO(@taiqing) use a vreg for weights to avoid duplicate indexing
            a->add_d(temp_gp, w, base_offset);
            a->add_d(temp_gp, temp_gp, index_offset);
            a->xvld(temp_vreg, la64::ptr(temp_gp));
            a->xvand_v(temp_vreg, temp_vreg, mask_vreg);

            a->xvfmadd_s(g_vreg, temp_vreg, weight_decay_vreg, g_vreg);
          }

          a->xvfmul_s(out_vreg, g_vreg, g_vreg);
          a->add_d(temp_gp, h, base_offset);
          a->add_d(temp_gp, temp_gp, index_offset);
          a->xvld(temp_vreg, la64::ptr(temp_gp));
          a->xvand_v(temp_vreg, temp_vreg, mask_vreg);
          a->xvfadd_s(out_vreg, out_vreg, temp_vreg);

          for(int st_idx = 0; st_idx < remainder; st_idx++){
            a->xvstelm_w(out_vreg, temp_gp, 0, st_idx);
            a->addi_d(temp_gp, temp_gp, sizeof(float));
          }
          a->xvfsqrt_s(out_vreg, out_vreg);
          a->xvfadd_s(out_vreg, out_vreg, epsilon_vreg);
          a->xvfmul_s(g_vreg, lr_vreg, g_vreg);
          a->xvfdiv_s(out_vreg, g_vreg, out_vreg);

          a->add_d(temp_gp, w, base_offset);
          a->add_d(temp_gp, temp_gp, index_offset);
          a->xvld(temp_vreg, la64::ptr(temp_gp));
          a->xvand_v(temp_vreg, temp_vreg, mask_vreg);
          a->xvfadd_s(out_vreg, out_vreg, temp_vreg);

          for(int st_idx = 0; st_idx < remainder; st_idx++){
            a->xvstelm_w(out_vreg, temp_gp, 0, st_idx);
            a->addi_d(temp_gp, temp_gp, sizeof(float));
          }

        }
      } else {
        a->add_d(temp_gp, g, index_offset);
        a->xvld(g_vreg, la64::ptr(temp_gp));

        if (has_weight_decay) {
          // float gj = std::fma(weight_decay * freq, w_[j], g_[j]);
          a->add_d(temp_gp, w, base_offset);
          a->add_d(temp_gp, temp_gp, index_offset);
          a->xvld(temp0_xv, la64::ptr(temp_gp));
          a->xvfmadd_s(g_vreg, weight_decay_vreg, temp0_xv, g_vreg);
        }

        a->xvfmul_s(out_vreg, g_vreg, g_vreg);
        a->add_d(temp_gp, h, base_offset);
        a->add_d(temp_gp, temp_gp, index_offset);
        a->xvld(temp0_xv, la64::ptr(temp_gp));
        a->xvfadd_s(out_vreg, out_vreg, temp0_xv);

        a->xvst(out_vreg, la64::ptr(temp_gp));
        a->xvfsqrt_s(out_vreg, out_vreg);
        a->xvfadd_s(out_vreg, out_vreg, epsilon_vreg);

        a->xvfmul_s(g_vreg, lr_vreg, g_vreg);
        a->xvfdiv_s(out_vreg, g_vreg, out_vreg);
        a->add_d(temp_gp, w, base_offset);
        a->add_d(temp_gp, temp_gp, index_offset);
        a->xvld(temp0_xv, la64::ptr(temp_gp));
        a->xvfadd_s(out_vreg, out_vreg, temp0_xv);

        a->xvst(out_vreg, la64::ptr(temp_gp));
      }
    }
  }
}

template <typename indxType, inst_set_t instSet>
void GenSparseAdagrad<indxType, instSet>::genRowwiseSparseAdagrad(
    la64::Emitter* a,
    int block_size,
    int unroll_factor,
    int num_vec_regs_per_block,
    int remainder,
    int prefetch,
    typename simd_info<instSet>::vec_reg_t epsilon_vreg,
    typename simd_info<instSet>::vec_reg_t lr_vreg,
    la64::VecX mask_vreg,
    typename simd_info<instSet>::vec_reg_t temp_vreg,
    typename simd_info<instSet>::vec_reg_t weight_decay_vreg,
    bool has_weight_decay) {
  typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;
  constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;

  // Reduce the unroll factor by 1 for partial sum
  --unroll_factor;
  vec_reg_t partial_sum_vreg = vec_reg_t(unroll_factor);

  if (prefetch) {
    a->add_d(temp_gp, h, temp3_);
    a->preld(8, temp_gp, 0);
  }

  bool areIndices64b = std::is_same<indxType, std::int64_t>::value;

  if (has_weight_decay) {
    // set base_offset for fetching w in the calculation of gradient square sum
    a->slli_d(temp_gp, temp1_, areIndices64b ? 3:2);
    a->add_d(temp_gp, indices, temp_gp);
    if(areIndices64b){
      a->ld_d(temp_gp, la64::ptr(temp_gp));
    }else{
      a->ld_w(temp_gp, la64::ptr(temp_gp));
    }
    // base_offset <- offsetIdx = idx * block_size
    mov_imm(a, base_offset, block_size * sizeof(float));
    a->mul_d(base_offset, base_offset, temp_gp);
  }

  constexpr int vlen_lasx = simd_info<inst_set_t::lasx>::WIDTH_32BIT_ELEMS;
  int num_vec_regs_per_block_lasx = (block_size + vlen_lasx - 1) / vlen_lasx;

  la64::VecX partial_sum_vreg0(0);

  a->xvxor_v(partial_sum_vreg0, partial_sum_vreg0, partial_sum_vreg0);

  // TODO: need to do a tree-reduction to fully take advantage of unrolling
  for (int vec_idx = 0; vec_idx < num_vec_regs_per_block_lasx;
       vec_idx += unroll_factor - 1) {
    int cur_unroll_factor =
        std::min(unroll_factor - 1, num_vec_regs_per_block_lasx - vec_idx);

    for (int v = 0; v < cur_unroll_factor; ++v) {
      mov_imm(a, index_offset, (vec_idx + v) * vlen_lasx * sizeof(float));

      la64::VecX out_vreg = la64::VecX(v + 1);

      if (has_weight_decay && prefetch &&
          ((vec_idx + v) % (64 / (vlen_lasx * sizeof(float))) == 0)) {
        a->add_d(temp_gp, w, temp2_);
        a->add_d(temp_gp, temp_gp, index_offset);
        a->preld(8, temp_gp, 0);
      }

      if (block_size % simd_info<inst_set_t::lasx>::WIDTH_32BIT_ELEMS &&
          vec_idx + v == num_vec_regs_per_block_lasx - 1) {
        if (instSet == inst_set_t::lasx) {
          a->add_d(temp_gp, g, index_offset);
          a->xvld(out_vreg, la64::ptr(temp_gp));
          a->xvand_v(out_vreg, out_vreg, mask_vreg);

          if (has_weight_decay) {
            a->add_d(temp_gp, w, index_offset);
            a->add_d(temp_gp, temp_gp, base_offset);
            a->xvld(temp_vreg, la64::ptr(temp_gp));
            a->xvand_v(temp_vreg, temp_vreg, mask_vreg);

            a->xvfmadd_s(out_vreg, temp_vreg, weight_decay_vreg, out_vreg);
          }
        }
      } else {
        a->add_d(temp_gp, g, index_offset);
        a->xvld(out_vreg, la64::ptr(temp_gp));

        if (has_weight_decay) {
          a->add_d(temp_gp, w, index_offset);
          a->add_d(temp_gp, temp_gp, base_offset);
          a->xvld(temp0_xv, la64::ptr(temp_gp));
          a->xvfmadd_s(out_vreg, weight_decay_vreg, temp0_xv, out_vreg);
        }
      }
      a->xvfmul_s(out_vreg, out_vreg, out_vreg);
      a->xvfadd_s(partial_sum_vreg0, partial_sum_vreg0, out_vreg);
    }
  }

  la64::VecX partial_sum_vreg1(1);

  // Reduce sum to 1 value
  // compute final_sum
  a->xvbsll_v(partial_sum_vreg1, partial_sum_vreg0, 0);
  a->xvsrai_d(partial_sum_vreg0, partial_sum_vreg0, 32);
  a->xvfadd_s(partial_sum_vreg0, partial_sum_vreg0, partial_sum_vreg1);
  a->xvbsll_v(partial_sum_vreg1, partial_sum_vreg0, 0);
  // partial_sum[0, 1, 2, 3]
  a->xvpickve_w(temp0_xv, partial_sum_vreg1, 2);
  a->fadd_s(partial_sum_vreg0, partial_sum_vreg0, temp0_xv);
  // partial_sum[4, 5, 6, 7]
  a->xvpickve_w(temp0_xv, partial_sum_vreg1, 4);
  a->xvpickve_w(temp1_xv, partial_sum_vreg1, 6);
  a->fadd_s(temp0_xv, temp0_xv, temp1_xv);
  // partial_sum[0,1, ..., 7]
  a->fadd_s(partial_sum_vreg0, temp0_xv, partial_sum_vreg0);

  // This fragment moves block size (N) to stack and bcasts it to xmm reg
  mov_imm(a, temp_gp, block_size);
  a->xvreplgr2vr_w(partial_sum_vreg1, temp_gp);
  a->ffint_s_w(partial_sum_vreg1, partial_sum_vreg1);  // int32 -> float

  if (has_weight_decay) {
    // set base_offset for fetching h
    a->slli_d(temp_gp, temp1_, areIndices64b ? 3:2);
    a->add_d(temp_gp, indices, temp_gp);
    if(areIndices64b){
      a->ld_d(temp_gp, la64::ptr(temp_gp));
    }else{
      a->ld_w(temp_gp, la64::ptr(temp_gp));
    }
    mov_imm(a, base_offset, sizeof(float));
    a->mul_d(base_offset, base_offset, temp_gp);
  }

  // final_sum /= N
  a->fdiv_s(partial_sum_vreg0, partial_sum_vreg0, partial_sum_vreg1);
  // load h
  a->add_d(temp_gp, h, base_offset);
  a->xvld(partial_sum_vreg1, la64::ptr(temp_gp));
  // *h + final_sum
  a->fadd_s(partial_sum_vreg0, partial_sum_vreg0, partial_sum_vreg1);
  // store h
  a->xvstelm_w(partial_sum_vreg0, temp_gp, 0, 0);
  // sqrt(hi)
  a->fsqrt_s(partial_sum_vreg0, partial_sum_vreg0);
  // bcast partial to all of ymm/zmm reg
  a->xvreplve0_w(partial_sum_vreg, partial_sum_vreg0);
  // lr / sqrt(hi) + epsilon
  a->xvfadd_s(partial_sum_vreg, partial_sum_vreg, epsilon_vreg);
  a->xvfdiv_s(partial_sum_vreg, lr_vreg, partial_sum_vreg);
  // partial_sum_vreg now has float_step

  // set base_offset for fetching w in updating weights
  a->slli_d(temp_gp, temp1_, areIndices64b ? 3:2);
  a->add_d(temp_gp, indices, temp_gp);
  if(areIndices64b){
    a->ld_d(temp_gp, la64::ptr(temp_gp));
  }else{
    a->ld_w(temp_gp, la64::ptr(temp_gp));
  }
  mov_imm(a, base_offset, block_size * sizeof(float));
  a->mul_d(base_offset, base_offset, temp_gp);

  for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
       vec_idx += unroll_factor) {
    int cur_unroll_factor =
        std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

    for (int v = 0; v < cur_unroll_factor; ++v) {
      mov_imm(a, index_offset, (vec_idx + v) * vlen * sizeof(float));
      vec_reg_t out_vreg = vec_reg_t(v);

      if (!has_weight_decay && prefetch &&
          ((vec_idx + v) % (64 / (vlen * sizeof(float))) == 0)) {
        a->add_d(temp_gp, w, temp2_);
        a->add_d(temp_gp, temp_gp, index_offset);
        a->preld(8, temp_gp, 0);
      }

      if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
        if (instSet == inst_set_t::lasx) {
          a->add_d(temp_gp, g, index_offset);
          a->xvld(temp_vreg, la64::ptr(temp_gp));
          a->xvand_v(temp_vreg, temp_vreg, mask_vreg);

          if (has_weight_decay) {
            a->add_d(temp_gp, w, base_offset);
            a->add_d(temp_gp, temp_gp, index_offset);
            a->xvld(out_vreg, la64::ptr(temp_gp));
            a->xvand_v(out_vreg, out_vreg, mask_vreg);

            // TODO(@taiqing): have vreg for weights
            a->xvfmadd_s(temp_vreg, weight_decay_vreg, out_vreg, temp_vreg);
          }
          a->xvfmul_s(temp_vreg, partial_sum_vreg, temp_vreg);

          a->add_d(temp_gp, w, base_offset);
          a->add_d(temp_gp, temp_gp, index_offset);
          a->xvld(out_vreg, la64::ptr(temp_gp));
          a->xvand_v(out_vreg, out_vreg, mask_vreg);

          a->xvfadd_s(out_vreg, temp_vreg, out_vreg);
          for(int st_idx = 0; st_idx < remainder; st_idx++){
            a->xvstelm_w(out_vreg, temp_gp, 0, st_idx);
            a->addi_d(temp_gp, temp_gp, sizeof(float));
          }

        }
      } else {
        if (has_weight_decay) {
          a->add_d(temp_gp, g, index_offset);
          a->xvld(out_vreg, la64::ptr(temp_gp));
          a->add_d(temp_gp, w, base_offset);
          a->add_d(temp_gp, temp_gp, index_offset);
          a->xvld(temp0_xv, la64::ptr(temp_gp));
          a->xvfmadd_s(out_vreg, weight_decay_vreg, temp0_xv, out_vreg);
          a->xvfmul_s(out_vreg, partial_sum_vreg, out_vreg);
        } else {
          a->add_d(temp_gp, g, index_offset);
          a->xvld(out_vreg, la64::ptr(temp_gp));
          a->xvfmul_s(out_vreg, out_vreg, partial_sum_vreg);
        }
        a->add_d(temp_gp, w, base_offset);
        a->add_d(temp_gp, temp_gp, index_offset);
        a->xvld(temp0_xv, la64::ptr(temp_gp));
        a->xvfadd_s(out_vreg, out_vreg, temp0_xv);
        a->xvst(out_vreg, la64::ptr(temp_gp));
      }
    }
  }
}

template <typename indxType, inst_set_t instSet>
typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
GenSparseAdagrad<indxType, instSet>::getOrCreate(
    int block_size,
    int prefetch,
    bool rowwise,
    bool has_weight_decay) {
  std::tuple<int, int, bool, bool> kernelSig =
      std::make_tuple(block_size, prefetch, rowwise, has_weight_decay);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() ->
      typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel {
        asmjit::CodeHolder code;
        code.init(runtime().environment());
        la64::Assembler assembler(&code);
        la64::Emitter* a = assembler.as<la64::Emitter>();
        bool areIndices64b = std::is_same<indxType, std::int64_t>::value;
#if defined(FBGEMM_LOG_CODE)
        std::string filename = "SparseAdagrad";
        filename += "_emd_dim_" + std::to_string(block_size);
        if (rowwise) {
          filename += "_rowwise";
        }
        filename += areIndices64b ? "_64bit" : "_32bit";
        filename += instSet == inst_set_t::lasx ? "_lasx" : "loongarch";
        if (prefetch) {
          filename += "_prefetch";
        }
        if (has_weight_decay) {
          filename += "weight_decay";
        }
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif

        la64::Gp num_rows = la64::a0;
        la64::Gp param_size = la64::a1;
        w = la64::a2;
        g = la64::a3;
        h = la64::a4;
        indices = la64::a5;
        la64::VecV epsilon = la64::VecV(0);
        la64::VecV lr = la64::VecV(1);
        la64::Gp mask_lasx = la64::a6;
        la64::VecV weight_decay = la64::VecV(2);
        la64::Gp counter = la64::a7;
        la64::Gp counter_halflife = la64::s0;

        base_offset = la64::a6;
        temp1_ = la64::s1;
        temp2_ = la64::s2;
        temp3_ = la64::s3;
        temp_gp = la64::s4;
        index_offset = la64::s5;

        asmjit::FuncDetail func;
        func.init(
            asmjit::FuncSignatureT<
                int, // return type
                int, // num rows
                std::uint64_t, // param_size
                float*, // w
                const float*, // g
                float*, // h
                const indxType*, // indices
                float, // epsilon
                float, // lr
                const int*, // mask_
                float, // weight_decay
                const double*, // counter then counter_halflife
                std::int64_t>(asmjit::CallConv::kIdHost),
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
            num_rows,
            param_size,
            w,
            g,
            h,
            indices,
            epsilon,
            lr,
            mask_lasx,
            weight_decay,
            counter,
            counter_halflife);

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

        vec_reg_t epsilon_vreg;
        vec_reg_t lr_vreg;
        vec_reg_t weight_decay_vreg;
        vec_reg_t adjusted_weight_decay_vreg;
        la64::VecX mask_vreg; // mask
        vec_reg_t temp_vreg; // temp vreg to handle remainder computation

        --unroll_factor;
        temp0_xv = la64::VecX(unroll_factor);
        --unroll_factor;
        temp1_xv = la64::VecX(unroll_factor);

        --unroll_factor;
        epsilon_vreg = vec_reg_t(unroll_factor);
        --unroll_factor;
        lr_vreg = vec_reg_t(unroll_factor);
        if (has_weight_decay) {
          --unroll_factor;
          weight_decay_vreg = vec_reg_t(unroll_factor);
          --unroll_factor;
          adjusted_weight_decay_vreg = vec_reg_t(unroll_factor);
        }

        if (remainder) {
          if (instSet == inst_set_t::lasx) {
            --unroll_factor;
            temp_vreg = vec_reg_t(unroll_factor);
          // }

          // Creating masks for non multiples of vlen iterations
          // if (instSet == inst_set_t::lasx) {
            --unroll_factor;
            mask_vreg = la64::VecX(unroll_factor);
            a->xvld(mask_vreg, la64::ptr(mask_lasx));
          // } else {
          }
        }

        if (!rowwise) {
          unroll_factor = unroll_factor / 2; // accont for g_vreg
        }

        asmjit::Label exit = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        a->vpickve2gr_w(temp_gp, epsilon, 0);
        a->xvreplgr2vr_w(epsilon_vreg, temp_gp);
        a->vpickve2gr_w(temp_gp, lr, 0);
        a->xvreplgr2vr_w(lr_vreg, temp_gp);
        if (has_weight_decay) {
          a->vpickve2gr_w(temp_gp, weight_decay, 0);
          a->xvreplgr2vr_w(weight_decay_vreg, temp_gp);
        }

        a->xor_(temp1_, temp1_, temp1_);

        a->bind(LoopRangeIndexBegin);
        a->bge(temp1_, num_rows, LoopRangeIndexEnd);

        a->slli_d(temp_gp, temp1_, areIndices64b ? 3:2);
        a->add_d(temp_gp, indices, temp_gp);
        // temp2_ <- idx = indices[i]
        if (areIndices64b) {
          a->ld_d(temp2_, la64::ptr(temp_gp));
        } else {
          a->ld_w(temp2_, la64::ptr(temp_gp));
        }
        mov_imm(a, base_offset, (rowwise ? 1 : block_size) * sizeof(float));
        a->mul_d(base_offset, base_offset, temp2_);

        if (has_weight_decay) {
          // Check counter != nullptr && counter[idx] > 0
          a->xvbsll_v(adjusted_weight_decay_vreg, weight_decay_vreg, 0);

          asmjit::Label skip_adjust_freq = a->newLabel();

          a->beq(la64::zero, counter, skip_adjust_freq);

          // temp3_ : counter[idx]
          a->slli_d(temp_gp, temp2_, 3);
          a->add_d(temp_gp, counter, temp_gp);
          a->ld_d(temp3_, la64::ptr(temp_gp));
          a->bge(la64::zero, temp3_, skip_adjust_freq);

          // OK to use Xmm registers with small ids that are reserved for temp
          // values in the inner most loop.
          // vec_reg_t counter_halflife_vreg(0);
          la64::VecX counter_halflife_vreg = la64::VecX(0);
          la64::VecX counter_vreg = la64::VecX(1);

          a->vinsgr2vr_d(temp0_xv, counter_halflife, 0);
          a->ffint_d_l(counter_halflife_vreg, temp0_xv);
          a->xvinsgr2vr_d(counter_vreg, temp3_, 0);
          a->fdiv_d(counter_halflife_vreg, counter_halflife_vreg, counter_vreg);
          a->fcvt_s_d(counter_halflife_vreg, counter_halflife_vreg);
          a->xvreplve0_w(counter_halflife_vreg, counter_halflife_vreg);
          a->xvfmul_s(
              adjusted_weight_decay_vreg,
              adjusted_weight_decay_vreg,
              counter_halflife_vreg);

          a->bind(skip_adjust_freq);
        }

        a->addi_d(temp2_, temp2_, 1);
        mov_imm(a, temp_gp, block_size);
        a->mul_d(temp2_, temp2_, temp_gp);  // (idx + 1) * block_size
        a->blt(param_size, temp2_, exit);

        if (prefetch) {
          asmjit::Label pref_dist_reset_start = a->newLabel();
          asmjit::Label pref_dist_reset_end = a->newLabel();

          a->add_d(temp2_, temp1_, la64::zero);
          mov_imm(a, temp_gp, prefetch);
          a->add_d(temp2_, temp2_, temp_gp);
          a->bge(temp2_, num_rows, pref_dist_reset_start);


          if (rowwise) {
            a->slli_d(temp_gp, temp2_, areIndices64b ? 3:2);
            a->add_d(temp_gp, temp_gp, indices);
            if (areIndices64b) {
              a->ld_d(temp_gp, la64::ptr(temp_gp));
            } else {
              a->ld_w(temp_gp, la64::ptr(temp_gp));
            }
            mov_imm(a, temp3_, sizeof(float));
            a->mul_d(temp3_, temp3_, temp_gp);
          }
          a->slli_d(temp_gp, temp2_, areIndices64b ? 3:2);
          a->add_d(temp_gp, temp_gp, indices);
          if (areIndices64b) {
            a->ld_d(temp_gp, la64::ptr(temp_gp));
          } else {
            a->ld_w(temp_gp, la64::ptr(temp_gp));
          }
          mov_imm(a, temp2_, block_size * sizeof(float));
          a->mul_d(temp2_, temp2_, temp_gp);

          a->b(pref_dist_reset_end);

          a->bind(pref_dist_reset_start);

          a->slli_d(temp_gp, temp1_, areIndices64b ? 3:2);
          a->add_d(temp_gp, indices, temp_gp);
          if (areIndices64b) {
            a->ld_d(temp_gp, la64::ptr(temp_gp));
          } else {
            a->ld_w(temp_gp, la64::ptr(temp_gp));
          }
          mov_imm(a, temp2_, block_size * sizeof(float));
          a->mul_d(temp2_, temp2_, temp_gp);

          if (rowwise) {
            a->slli_d(temp_gp, temp1_, areIndices64b ? 3:2);
            a->add_d(temp_gp, indices, temp_gp);
            if (areIndices64b) {
              a->ld_d(temp_gp, la64::ptr(temp_gp));
            } else {
              a->ld_w(temp_gp, la64::ptr(temp_gp));
            }
            mov_imm(a, temp3_, sizeof(float));
            a->mul_d(temp3_, temp3_, temp_gp);
          }

          a->bind(pref_dist_reset_end);
        } // prefetch

        if (rowwise) {
          genRowwiseSparseAdagrad(
              a,
              block_size,
              unroll_factor,
              num_vec_regs_per_block,
              remainder,
              prefetch,
              epsilon_vreg,
              lr_vreg,
              mask_vreg,
              temp_vreg,
              adjusted_weight_decay_vreg,
              has_weight_decay);
        } else {
          genSparseAdagrad(
              a,
              unroll_factor,
              num_vec_regs_per_block,
              remainder,
              prefetch,
              epsilon_vreg,
              lr_vreg,
              mask_vreg,
              temp_vreg,
              adjusted_weight_decay_vreg,
              has_weight_decay);
        }

        mov_imm(a, temp_gp, block_size * sizeof(float));
        a->add_d(g, g, temp_gp);
        a->addi_d(temp1_, temp1_, 1);
        a->b(LoopRangeIndexBegin);

        a->bind(LoopRangeIndexEnd);

        a->bind(exit);
        a->add_d(la64::a0, la64::zero, temp1_);  // return i
        a->emitEpilog(frame);

        typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
            fn;
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
} // getOrCreate

// Specialization for block size 1 internally called by GenerateSparseAdaGrad
template <typename IndexType>
int SparseAdaGradBlockSize1_(
    int num_rows, // number of rows reading
    std::uint64_t param_size, // total number of parameters
    float* w, // input/output parameters
    const float* g, // input gradients
    float* h, // input/output momentums
    const IndexType* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise,
    float weight_decay,
    const double* counter,
    std::int64_t counter_halflife) {
  if (weight_decay != 0.0f) {
    for (int i = 0; i < num_rows; ++i) {
      IndexType idx = indices[i];
      if (idx >= static_cast<int64_t>(param_size)) {
        return i;
      }

      float freq = (counter && counter[idx] > 0)
          ? counter_halflife / counter[idx]
          : 1.0f;
      float gi = std::fma(freq * weight_decay, w[idx], g[i]);
      float hi = h[idx] = h[idx] + gi * gi;
      if (rowwise) {
        w[idx] += lr / (std::sqrt(hi) + epsilon) * gi;
      } else {
        w[idx] += lr * gi / (std::sqrt(hi) + epsilon);
      }
    }
  } else {
    for (int i = 0; i < num_rows; ++i) {
      IndexType idx = indices[i];
      if (idx >= static_cast<int64_t>(param_size)) {
        return i;
      }
      float gi = g[i];
      float hi = h[idx] = h[idx] + gi * gi;
      if (rowwise) {
        w[idx] += lr / (std::sqrt(hi) + epsilon) * gi;
      } else {
        w[idx] += lr * gi / (std::sqrt(hi) + epsilon);
      }
    }
  }
  return num_rows;
}

template int SparseAdaGradBlockSize1_(
    int num_rows, // number of rows reading
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int64_t* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise,
    float weight_decay,
    const double* counter,
    std::int64_t counter_halflife);

template int SparseAdaGradBlockSize1_(
    int num_rows, // number of rows reading
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int32_t* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise,
    float weight_decay,
    const double* counter,
    std::int64_t counter_halflife);

} // namespace

template <typename IndexType>
typename SparseAdaGradSignature<IndexType>::Type GenerateSparseAdaGrad(
    int block_size,
    bool rowwise,
    int prefetch,
    bool use_weight_decay) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  if (fbgemmHasLasxSupport()) {
    if (block_size == 1) {
      return [=](int num_rows, // number of rows reading
                 std::uint64_t param_size, // total number of parameters
                 float* w, // input/output parameters
                 const float* g, // input gradients
                 float* h, // input/output momentums
                 const IndexType* indices, // indices of each row
                 float epsilon,
                 float lr,
                 float weight_decay,
                 const double* counter,
                 std::int64_t counter_halflife) {
        return SparseAdaGradBlockSize1_(
            num_rows,
            param_size,
            w,
            g,
            h,
            indices,
            epsilon,
            lr,
            rowwise,
            weight_decay,
            counter,
            counter_halflife);
      };
    }
    static GenSparseAdagrad<IndexType, inst_set_t::lasx> kernel_generator;
    constexpr int VLEN = simd_info<inst_set_t::lasx>::WIDTH_32BIT_ELEMS;
    const int* mask_lasx = &internal::lasx_ps_or_epi32_combined_mask
                               [(VLEN - (block_size % VLEN)) % VLEN];
    const auto original_func = kernel_generator.getOrCreate(
        block_size, prefetch, rowwise, use_weight_decay);
    return [=](int num_rows, // number of rows reading
               std::uint64_t param_size, // total number of parameters
               float* w, // input/output parameters
               const float* g, // input gradients
               float* h, // input/output momentums
               const IndexType* indices, // indices of each row
               float epsilon,
               float lr,
               float weight_decay,
               const double* counter,
               std::int64_t counter_halflife) {
      return original_func(
          num_rows, // number of rows reading
          param_size, // total number of parameters
          w, // input/output parameters
          g, // input gradients
          h, // input/output momentums
          indices, // indices of each row
          epsilon,
          lr,
          mask_lasx,
          weight_decay,
          counter,
          counter_halflife);
    };
  } else {
#ifdef VLOG
    VLOG(0) << "LASX not found, taking the slow path";
#endif
    return [=](int num_rows, // number of rows reading
               std::uint64_t param_size, // total number of parameters
               float* w, // input/output parameters
               const float* g, // input gradients
               float* h, // input/output momentums
               const IndexType* indices, // indices of each row
               float epsilon,
               float lr,
               float weight_decay,
               const double* counter,
               std::int64_t counter_halflife) {
      if (rowwise) {
        return rowwise_sparse_adagrad_ref(
            num_rows, // number of rows reading
            block_size, // number of parameters per rows
            param_size, // total number of parameters
            w, // input/output parameters
            g, // input gradients
            h, // input/output momentums
            indices,
            epsilon,
            lr,
            weight_decay,
            counter,
            counter_halflife);
      } else {
        return sparse_adagrad_ref(
            num_rows, // number of rows reading
            block_size, // number of parameters per rows
            param_size, // total number of parameters
            w, // input/output parameters
            g, // input gradients
            h, // input/output momentums
            indices,
            epsilon,
            lr,
            weight_decay,
            counter,
            counter_halflife);
      }
    };
  }
}

template FBGEMM_API typename SparseAdaGradSignature<std::int64_t>::Type
GenerateSparseAdaGrad<std::int64_t>(
    int block_size, // number of parameters per rows
    bool rowwise,
    int prefetch,
    bool use_weight_decay);

template FBGEMM_API typename SparseAdaGradSignature<std::int32_t>::Type
GenerateSparseAdaGrad<std::int32_t>(
    int block_size, // number of parameters per rows
    bool rowwise,
    int prefetch,
    bool use_weight_decay);

} // namespace fbgemm
