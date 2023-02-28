/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "./GenerateI8Depthwise.h"

#include <asmjit/asmjit.h>
#include <cassert>
#include <iostream>
#include <numeric>

#include "./CodeCache.h"
#include "./CodeGenHelpers.h"
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace {
asmjit::JitRuntime& runtime() {
  static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                // depents on other static
                                // variables.  Required to prevent
                                // initialization order fiasco
  return rt;
}

// Controll access to runtime;
std::mutex rtMutex_;

// The hash depends on D, K_T, K_H, K_W, oc_per_g, compute_a_sum,
// remainder, prev_skip, next_skip, top_skip, bottom_skip, left_skip, and
// right_skip.
CodeCache<
    std::
        tuple<int, int, int, int, int, bool, int, int, int, int, int, int, int>,
    GenI8Depthwise::jit_kernel_signature>
    codeCache_;
} // namespace

namespace la64 = asmjit::la64;

// c = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
// A is in uint8_t
// B is in int8_t and pre-interleaved
// C is in int32_t and 4 registers have results in the following layout:
// c0_v:   c[0:4], c[16:20]
// c1_v:   c[4:8], c[20:24]
// c2_v:  c[8:12], c[24:28]
// c3_v: c[12:16], c[28:32]
static void genMaddEpi16xNPacked(
    la64::Emitter* e,
    la64::VecX a[4],
    la64::Gp b,
    la64::VecX c[4],
    la64::VecX* a_sum,
    int n,
    int remainder,
    bool accumulation,
    la64::VecX one_epi8,
    la64::VecX one_epi16,
    la64::VecX zero,
    la64::VecX tmpReg1_V_,
    la64::VecX tmpReg2_V_) {
  // Interleave inputs corresponding to 4 filter positions.
  // Reuse a[1] and a[3] to save registers
  la64::VecX a01_lo(0), a01_hi(1), a23_lo(a[1]), a23_hi(a[3]);
  e->xvilvl_b(a01_lo, n == 1 ? zero : a[1], a[0]);
  if (remainder >= 8) {
    e->xvilvh_b(a01_hi, n == 1 ? zero : a[1], a[0]);
  }
  if (n > 2) {
    e->xvilvl_b(a23_lo, n == 3 ? zero : a[3], a[2]);
    if (remainder >= 8) {
      e->xvilvh_b(a23_hi, n == 3 ? zero : a[3], a[2]);
    }
  }

  // Compute row_wise sum of A for row_offsets
  if (a_sum) {
    if (accumulation) {
      e->xvmulwev_h_bu_b (a[0], a01_lo, one_epi8);
      e->xvmulwod_h_bu_b(tmpReg1_V_, a01_lo, one_epi8);
      e->xvsadd_h(a[0], a[0], tmpReg1_V_);
      e->xvsadd_h(a_sum[0], a[0], a_sum[0]);

      if (remainder >= 8) {
        e->xvmulwev_h_bu_b (a[2], a01_hi, one_epi8);
        e->xvmulwod_h_bu_b(tmpReg1_V_, a01_hi, one_epi8);
        e->xvsadd_h(a[2], a[2], tmpReg1_V_);
        e->xvsadd_h(a_sum[1], a[2], a_sum[1]);
      }
    } else {
      e->xvmulwev_h_bu_b (a_sum[0], a01_lo, one_epi8);
      e->xvmulwod_h_bu_b(tmpReg1_V_, a01_lo, one_epi8);
      e->xvsadd_h(a_sum[0], a_sum[0], tmpReg1_V_);
      if (remainder >= 8) {
        e->xvmulwev_h_bu_b (a_sum[1], a01_hi, one_epi8);
        e->xvmulwod_h_bu_b(tmpReg1_V_, a01_hi, one_epi8);
        e->xvsadd_h(a_sum[1], a_sum[1], tmpReg1_V_);
      }
    }

    if (n > 2) {
      e->xvmulwev_h_bu_b (a[0], a23_lo, one_epi8);
      e->xvmulwod_h_bu_b(tmpReg1_V_, a23_lo, one_epi8);
      e->xvsadd_h(a[0], a[0], tmpReg1_V_);
      e->xvsadd_h(a_sum[0], a[0], a_sum[0]);

      if (remainder >= 8) {
        e->xvmulwev_h_bu_b (a[2], a23_hi, one_epi8);
        e->xvmulwod_h_bu_b(tmpReg1_V_, a23_hi, one_epi8);
        e->xvsadd_h(a[2], a[2], tmpReg1_V_);
        e->xvsadd_h(a_sum[1], a[2], a_sum[1]);
      }
    }
  }

  if (n > 2) {
    // Reusing a
    e->xvilvl_h(a[0], a23_lo, a01_lo);
    e->xvilvh_h(a[1], a23_lo, a01_lo);
    if (remainder >= 16) {
      e->xvilvl_h(a[2], a23_hi, a01_hi);
      e->xvilvh_h(a[3], a23_hi, a01_hi);
    }

    e->xvld(tmpReg1_V_, ptr(b));
    e->xvmulwev_h_bu_b (tmpReg2_V_, a[0], tmpReg1_V_);
    e->xvmulwod_h_bu_b(a[0], a[0], tmpReg1_V_);
    e->xvsadd_h(a[0], a[0], tmpReg2_V_);

    e->xvld(tmpReg1_V_, ptr(b, 32));
    e->xvmulwev_h_bu_b (tmpReg2_V_, a[1], tmpReg1_V_);
    e->xvmulwod_h_bu_b(a[1], a[1], tmpReg1_V_);
    e->xvsadd_h(a[1], a[1], tmpReg2_V_);

    if (remainder >= 16) {
      e->xvld(tmpReg1_V_, ptr(b, 64));
      e->xvmulwev_h_bu_b (tmpReg2_V_, a[2], tmpReg1_V_);
      e->xvmulwod_h_bu_b(a[2], a[2], tmpReg1_V_);
      e->xvsadd_h(a[2], a[2], tmpReg2_V_);

      e->xvld(tmpReg1_V_, ptr(b, 96));
      e->xvmulwev_h_bu_b (tmpReg2_V_, a[3], tmpReg1_V_);
      e->xvmulwod_h_bu_b(a[3], a[3], tmpReg1_V_);
      e->xvsadd_h(a[3], a[3], tmpReg2_V_);
    }

    if (accumulation) {
      e->xvmulwev_w_h (tmpReg1_V_, a[0], one_epi16);
      e->xvmaddwod_w_h(tmpReg1_V_, a[0], one_epi16);
      e->xvor_v(a[0], tmpReg1_V_, tmpReg1_V_);
      e->xvadd_w(c[0], c[0], a[0]);
      e->xvmulwev_w_h (tmpReg1_V_, a[1], one_epi16);
      e->xvmaddwod_w_h(tmpReg1_V_, a[1], one_epi16);
      e->xvor_v(a[1], tmpReg1_V_, tmpReg1_V_);
      e->xvadd_w(c[1], c[1], a[1]);

      if (remainder >= 16) {
        e->xvmulwev_w_h (tmpReg1_V_, a[2], one_epi16);
        e->xvmaddwod_w_h(tmpReg1_V_, a[2], one_epi16);
        e->xvor_v(a[2], tmpReg1_V_, tmpReg1_V_);
        e->xvadd_w(c[2], c[2], a[2]);

        e->xvmulwev_w_h (tmpReg1_V_, a[3], one_epi16);
        e->xvmaddwod_w_h(tmpReg1_V_, a[3], one_epi16);
        e->xvor_v(a[3], tmpReg1_V_, tmpReg1_V_);
        e->xvadd_w(c[3], c[3], a[3]);
      }
    } else {
      e->xvmulwev_w_h (c[0], a[0], one_epi16);
      e->xvmaddwod_w_h(c[0], a[0], one_epi16);

      e->xvmulwev_w_h (c[1], a[1], one_epi16);
      e->xvmaddwod_w_h(c[1], a[1], one_epi16);

      if (remainder >= 16) {
        e->xvmulwev_w_h (c[2], a[2], one_epi16);
        e->xvmaddwod_w_h(c[2], a[2], one_epi16);
        e->xvmulwev_w_h (c[3], a[3], one_epi16);
        e->xvmaddwod_w_h(c[3], a[3], one_epi16);
      }
    }
  } else {
    // Reusing a
    e->xvld(tmpReg1_V_, ptr(b));
    e->xvmulwev_h_bu_b (a[0], a01_lo, tmpReg1_V_);
    e->xvmulwod_h_bu_b(tmpReg2_V_, a01_lo, tmpReg1_V_);
    e->xvsadd_h(a[0], a[0], tmpReg2_V_);

    e->xvld(tmpReg1_V_, ptr(b, 32));
    e->xvmulwev_h_bu_b (a[1], a01_hi, tmpReg1_V_);
    e->xvmulwod_h_bu_b(tmpReg2_V_, a01_hi, tmpReg1_V_);
    e->xvsadd_h(a[1], a[1], tmpReg2_V_);

    if (accumulation) {
      e->xvpermi_d(tmpReg1_V_, a[0], 0x10);
      e->xvsllwil_w_h(a[2], tmpReg1_V_, 0);

      e->xvadd_w(c[0], c[0], a[2]);

      e->xvpermi_d(tmpReg1_V_, a[1], 0x10);
      e->xvsllwil_w_h(a[3], tmpReg1_V_, 0);
      e->xvadd_w(c[1], c[1], a[3]);

      if (remainder >= 16) {
        e->xvpermi_d(a[0], a[0], 0x32);
        e->xvsllwil_w_h(a[0], a[0], 0);
        e->xvadd_w(c[2], c[2], a[0]);
        e->xvpermi_d(a[1], a[1], 0x32);
        e->xvsllwil_w_h(a[1], a[1], 0);
        e->xvadd_w(c[3], c[3], a[1]);
      }
    } else {
      e->xvpermi_d(tmpReg1_V_, a[0], 0x10);
      e->xvsllwil_w_h(c[0], tmpReg1_V_, 0);
      e->xvpermi_d(tmpReg1_V_, a[1], 0x10);
      e->xvsllwil_w_h(c[1], tmpReg1_V_, 0);

      if (remainder >= 16) {
        e->xvpermi_d(tmpReg1_V_, a[0], 0x32);
        e->xvsllwil_w_h(c[2], tmpReg1_V_, 0);
        e->xvpermi_d(tmpReg1_V_, a[1], 0x32);
        e->xvsllwil_w_h(c[3], tmpReg1_V_, 0);
      }
    }
  }
}

GenI8Depthwise::jit_kernel_signature GenI8Depthwise::getOrCreate(
    int D,
    std::array<int, 3> F,
    int oc_per_g,
    bool compute_a_sum,
    int remainder,
    int prev_skip,
    int next_skip,
    int top_skip,
    int bottom_skip,
    int left_skip,
    int right_skip) {
  std::tuple<int, int, int, int, int, bool, int, int, int, int, int, int, int>
      kernelSig = std::make_tuple(
          D,
          F[0],
          F[1],
          F[2],
          oc_per_g,
          compute_a_sum,
          remainder,
          prev_skip,
          next_skip,
          top_skip,
          bottom_skip,
          left_skip,
          right_skip);

  return codeCache_.getOrCreate(kernelSig, [&]() -> jit_kernel_signature {
    asmjit::CodeHolder code;
    code.init(runtime().environment());
    la64::Assembler assembler(&code);
    la64::Emitter* e = assembler.as<la64::Emitter>();
#ifdef FBGEMM_LOG_CODE
    std::string filename = "dwconv_" + std::to_string(D) + "d_";
    for (int i = 3 - D; i < 3; ++i) {
      filename += std::to_string(K[i]);
      if (i < 2) {
        filename += "x"
      }
    }
    filename += "_" + std::to_string(oc_per_g);
    if (compute_a_sum) {
      filename += "_asum";
    }
    if (remainder) {
      filename += "_remainder" + std::to_string(remainder);
    }
    if (prev_skip) {
      filename += "_prev_skip" + std::to_string(prev_skip);
    }
    if (next_skip) {
      filename += "_next_skip" + std::to_string(next_skip);
    }
    if (top_skip) {
      filename += "_top_skip" + std::to_string(top_skip);
    }
    if (bottom_skip) {
      filename += "_bottom_skip" + std::to_string(bottom_skip);
    }
    if (left_skip) {
      filename += "_left_skip" + std::to_string(left_skip);
    }
    if (right_skip) {
      filename += "_right_skip" + std::to_string(right_skip);
    }
    filename += ".txt";
    FILE* codeLogFile = fopen(filename.c_str(), "w");
    asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
    code.setLogger(codeLogger);
#endif

    la64::Gp a_addr = la64::a0;
    la64::Gp b_addr = la64::a1;
    la64::Gp c_addr = la64::a2;
    la64::Gp a_sum_addr = la64::a3;
    la64::Gp h = la64::a4;
    la64::Gp w = la64::a5;
    la64::Gp ic = la64::a6;
    la64::Gp mask_addr = la64::a7;
    la64::Gp a_zero_point = la64::s0;
    la64::Gp b_zero_point_addr = la64::s1;
    la64::Gp ic_loop_count = la64::s2;
    la64::Gp a_addr_save = la64::s3;

    la64::Gp   tmpReg1_Gp_ = la64::s4;
    la64::VecX tmpReg1_V_ = la64::VecX(16);
    la64::VecX tmpReg2_V_ = la64::VecX(17);

    asmjit::FuncDetail func;
    func.init(
        asmjit::FuncSignatureT<
            void,
            const std::uint8_t*,
            const std::int8_t*,
            std::int32_t*,
            std::int32_t*,
            int,
            int,
            int,
            const int*,
            int,
            const std::int32_t*>(asmjit::CallConv::kIdHost),
        e->environment());

    asmjit::FuncFrame frame;
    frame.init(func);

    frame.setDirtyRegs(
        la64::Reg::kGroupVec,
        asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
            asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18));
    frame.setDirtyRegs(
        la64::Reg::kGroupGp,
        asmjit::Support::bitMask(23, 24, 25, 26, 27));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(
        a_addr,
        b_addr,
        c_addr,
        a_sum_addr,
        h,
        w,
        ic,
        mask_addr,
        a_zero_point,
        b_zero_point_addr);

    args.updateFuncFrame(frame);
    frame.finalize();

    e->emitProlog(frame);
    e->emitArgsAssignment(frame, args);

    // Assign vector registers
    la64::VecX a[4];
    la64::VecX c[4];
    la64::VecX a_sum[2];

    int vreg_id = 2; // reserve 2 for temp vreg
    for (int i = 0; i < 4; ++i, ++vreg_id) {
      a[i] = la64::VecX(vreg_id);
    }
    for (int i = 0; i < 4; ++i, ++vreg_id) {
      c[i] = la64::VecX(vreg_id);
    }
    if (compute_a_sum) {
      a_sum[0] = la64::VecX(vreg_id);
      ++vreg_id;
      a_sum[1] = la64::VecX(vreg_id);
      ++vreg_id;
    }
    la64::VecX mask_vreg(vreg_id);
    constexpr int vlen = simd_info<inst_set_t::lasx>::WIDTH_32BIT_ELEMS;
    if (remainder != simd_info<inst_set_t::lasx>::WIDTH_BYTES) {
      ++vreg_id;
      e->xvld(
          mask_vreg,
          ptr(
              mask_addr,
              (vlen - remainder / 4 / oc_per_g) % vlen * sizeof(int32_t)));  // *4, aligned default?
    }
    la64::VecX one_epi8(vreg_id);
    if (compute_a_sum) {
      ++vreg_id;
      gen8BitVectorOne(e, one_epi8);
    }

    int K = std::accumulate(F.begin(), F.end(), 1, std::multiplies<int>());
    la64::VecX one_epi16(vreg_id);
    if (K > 2) {
      ++vreg_id;
      gen16BitVectorOne<inst_set_t::lasx, la64::VecX>(e, one_epi16);
    }

    bool has_pad = prev_skip || next_skip || top_skip || bottom_skip ||
        left_skip || right_skip;
    bool need_zero = K % 4 == 3 || K % 4 == 1;
    // When out of registers, zero and A_zero_point_vreg need to share.
    bool recompute_zero = vreg_id == 15 && need_zero;

    la64::VecX a_zero_point_vreg(vreg_id);
    if (!recompute_zero && has_pad) {
      e->xvreplgr2vr_b(a_zero_point_vreg, a_zero_point);
    }
    if (vreg_id < 15) {
      ++vreg_id;
    }
    la64::VecX zero(vreg_id);
    if (need_zero && (!recompute_zero || !has_pad)) {
      e->xvxor_v(zero, zero, zero);
    }

    // Assign scalar registers
    e->mul_d(w, w, ic);
    e->mul_d(h, h, w);
    if (D >= 3) {
      e->add_d(a_addr_save, w, la64::zero);
      mov_imm(e, tmpReg1_Gp_, F[1]);
      e->mul_d(a_addr_save, a_addr_save, tmpReg1_Gp_);
      e->sub_d(h, h, a_addr_save); // h * w * ic - F[1] * w * ic
    }
    e->add_d(a_addr_save, ic, la64::zero);
    mov_imm(e, tmpReg1_Gp_, F[2]);
    e->mul_d(a_addr_save, a_addr_save, tmpReg1_Gp_);
    e->sub_d(w, w, a_addr_save); // w * ic - F[2] * ic

    e->add_d(ic_loop_count, ic, la64::zero);
    e->addi_d(ic_loop_count, ic_loop_count, asmjit::Imm(32 / oc_per_g - 1));
    e->srai_d(ic_loop_count, ic_loop_count, asmjit::Imm(oc_per_g == 1 ? 5 : 4));

    e->add_d(a_addr_save, a_addr, la64::zero);
    asmjit::Label ic_loop_begin = e->newLabel(), ic_loop_end = e->newLabel();

    // main_loop == false: the last vector iteration across input channels
    for (bool main_loop : {true, false}) {
      if (main_loop) {
        e->bind(ic_loop_begin);
        e->addi_d(ic_loop_count, ic_loop_count, -1);
        e->bge(la64::zero, ic_loop_count, ic_loop_end);
      }

      if (recompute_zero && has_pad) {
        e->xvreplgr2vr_b(a_zero_point_vreg, a_zero_point);
      }

      int i = 0;
      // Iterate across the reduction (filter) dimension
      for (int f_t = 0; f_t < ((D == 2) ? 1 : F[0]); ++f_t) {
        for (int f_h = 0; f_h < F[1]; ++f_h) {
          for (int f_w = 0; f_w < F[2]; ++f_w, ++i) {
            bool pad = false;
            if (D > 2) {
              if (f_t < prev_skip || f_t >= F[0] - next_skip) {
                pad = true;
              }
            }
            if (f_h < top_skip || f_h >= F[1] - bottom_skip ||
                f_w < left_skip || f_w >= F[2] - right_skip) {
              pad = true;
            }

            // Load A
            if (pad) {
              e->xvor_v(a[i % 4], a_zero_point_vreg, a_zero_point_vreg);
            } else {
              if (oc_per_g == 1) {
                if (!main_loop && remainder != 32) {
                  e->xvld(a[i % 4], ptr(a_addr));
                  e->xvand_v(a[i % 4], a[i % 4], mask_vreg);  //mask word MUST be 0/-1
                } else {
                  e->xvld(a[i % 4], ptr(a_addr));
                }
              } else {
                assert(oc_per_g == 2);
                if (!main_loop && remainder != 32) {
                  e->vld(a[i % 4].half(), ptr(a_addr));
                  e->vand_v(a[i % 4].half(), a[i % 4].half(), mask_vreg.half());  //mask word MUST be 0/-1
                } else {
                  e->vld(a[i % 4], ptr(a_addr));
                }
                // Duplicate each byte.
                e->xvpermi_d(a[i % 4], a[i % 4], 0x40);
                e->xvexth_hu_bu(a[i % 4], a[i % 4]);

                e->xvslli_h(la64::VecX(i % 2), a[i % 4], asmjit::Imm(8));
                e->xvadd_h(a[i % 4], a[i % 4], la64::VecX(i % 2));
              }
            }

            // Compute when we have 4 inputs or this is the last iteration
            if (i % 4 == 3 || i == K - 1) {
              if (i == K - 1 && (i / 4 * 4 == K - 3 || i / 4 * 4 == K - 1)) {
                if (recompute_zero && has_pad) {
                  e->xvxor_v(zero, zero, zero);
                }
              }

              genMaddEpi16xNPacked(
                  e,
                  a,
                  b_addr,
                  c,
                  compute_a_sum ? a_sum : nullptr,
                  /*n=*/std::min(K - i / 4 * 4, 4),
                  main_loop ? 32 : remainder,
                  /*accumulation=*/i / 4 > 0,
                  one_epi8,
                  one_epi16,
                  zero,
                  tmpReg1_V_,
                  tmpReg2_V_);

              if (i != K - 1) {
                e->addi_d(b_addr, b_addr, asmjit::Imm(32 * 4));
              } else if (main_loop) {
                e->addi_d(b_addr, b_addr, asmjit::Imm(32 * (K - i / 4 * 4 + 1) / 2 * 2));
              }

              if (K - i / 4 * 4 >= 3 && K - i / 4 * 4 <= 6) {
                for (int r = 0; r < (main_loop ? 4 : remainder / 8); ++r) {
                  // fix? output layout (see genMaddEpi16xNPacked for details)
                  e->xvor_v(a[r], c[r % 2 * 2 + 1], c[r % 2 * 2 + 1]);
                  e->xvpermi_q(
                      a[r],
                      c[r % 2 * 2],
                      asmjit::Imm(r < 2 ? 0x20 : 0x31));
                }
                for (int r = 0; r < (main_loop ? 4 : remainder / 8); ++r) {
                  e->xvor_v(c[r], a[r], a[r]);
                }
              }
            }
            if (i != K - 1) {
              e->add_d(a_addr, a_addr, ic); // advance to next pixel
            }
          }
          if (i != K - 1) {
            e->add_d(a_addr,a_addr, w); // advance to next row
          }
        }
        if (D >= 3 && i != K - 1) {
          e->add_d(a_addr, a_addr, h); // advance to next frame
        }
      }

      for (int r = 0; r < (main_loop ? 4 : remainder / 8); ++r) {
        e->xvst(c[r], ptr(c_addr, r * 32));
      }

      if (compute_a_sum) {
        if (oc_per_g == 1) {
          e->xvpermi_d(tmpReg1_V_, a_sum[0], 0x40);
          e->xvexth_w_h(a[0], tmpReg1_V_);
          e->xvst(a[0], ptr(a_sum_addr));
        } else {
          // Rollback duplication
          e->xvsrli_w(a_sum[0], a_sum[0], asmjit::Imm(16));
          e->vst(a_sum[0].half(), ptr(a_sum_addr));
        }

        if (main_loop || remainder >= 8) {
          if (oc_per_g == 1) {
            e->xvpermi_d(tmpReg1_V_, a_sum[1], 0x40);
            e->xvexth_w_h(a[1], tmpReg1_V_);
            e->xvst(a[1], ptr(a_sum_addr, 32));
          } else {
            // Rollback duplication
            e->xvsrli_w(a_sum[1], a_sum[1], asmjit::Imm(16));
            e->vst(a_sum[1].half(), ptr(a_sum_addr, 16));
          }
        }

        if (main_loop || remainder >= 16) {
          if (oc_per_g == 1) {
            e->xvpermi_d(tmpReg1_V_, a_sum[0], 0xC8);  //With vextracti128, perm: 3,0,2,0
            e->xvexth_w_h(a_sum[0], tmpReg1_V_);
            e->xvst(a_sum[0], ptr(a_sum_addr, 64));
          } else {
            e->xvpermi_d(a_sum[0], a_sum[0], 0x0E);
            e->vst(a_sum[0].half(), ptr(a_sum_addr, 32));
          }
        }

        if (main_loop || remainder >= 24) {
          if (oc_per_g == 1) {
            e->xvpermi_d(tmpReg1_V_, a_sum[1], 0xC8);  //With vextracti128
            e->xvexth_w_h(a_sum[1], tmpReg1_V_);
            e->xvst(a_sum[1], ptr(a_sum_addr, 96));
          } else {
            e->xvpermi_d(a_sum[1], a_sum[1], 0x0E);  //Put vextracti128
            e->vst(a_sum[1].half(), ptr(a_sum_addr, 48));
          }
        }

        if (main_loop) {
          e->addi_d(a_sum_addr, a_sum_addr, asmjit::Imm(128 / oc_per_g));
        }
      }

      if (main_loop) {
        e->addi_d(c_addr, c_addr, asmjit::Imm(128));
        e->addi_d(a_addr_save, a_addr_save, asmjit::Imm(32 / oc_per_g));
        e->add_d(a_addr, a_addr_save, la64::zero);
        e->b(ic_loop_begin);

        e->bind(ic_loop_end);
      }
    }

    e->emitEpilog(frame);

    jit_kernel_signature fn;
    asmjit::Error err;
    {
      std::unique_lock<std::mutex> lock(rtMutex_);
      err = runtime().add(&fn, &code);
    }
    if (err) {
      std::cout << "Error: in fn add" << std::endl;
      return nullptr;
    }

#ifdef FBGEMM_LOG_CODE
    fclose(codeLogFile);
    delete codeLogger;
#endif

    return fn;
  });
}

} // namespace fbgemm
