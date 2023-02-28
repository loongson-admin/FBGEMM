/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <asmjit/asmjit.h>
#include "./CodeGenHelpers.h"
#include "./GroupwiseConv.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

using namespace std;

namespace la64 = asmjit::la64;

GCONV_INST_DEF_LASX_HEADER
GenConvKernel<SPATIAL_DIM, INST_SET>::genConstForPermutations(la64::Emitter* a) {
  if (this->C_per_G_ == 4) {
    la64::GpX permute_const_reg = la64::r23;
    la64::VecX const_reg_xmm = la64::xr11;
    // We have 1st group in even lanes and 2nd group in odd lanes.
    // Permute to put 1st group to lower 128-bit and 2nd group in upper
    // 128-bit.
    // load 7, 5, 3, 1, 6, 4, 2, 0 in a 64-bit reg
    mov_imm(a, permute_const_reg, 0x0705030106040200);
    a->xvreplgr2vr_d(const_reg_xmm, permute_const_reg);
    // Zero extend 8 packed 8-bit integers in the low 8 bytes of const_reg_xmm
    // to 8 packed 32-bit integers in stPermReg_V_
    a->xvsllwil_hu_bu(stPermReg_V_, const_reg_xmm, 0);
    a->xvpermi_d(stPermReg_V_, stPermReg_V_, 0x10);
    a->xvsllwil_wu_hu(stPermReg_V_, stPermReg_V_, 0);
  } else {
    // this->C_per_G_ == 2
    la64::GpX permute_const_reg = la64::r23;
    la64::VecX const_reg_xmm = la64::xr11;
    // We have 1st group in position 0 and 4, 2nd group 1 and 5 and so on.
    // Permute to put 1st group to lower 64-bit and 2nd group to next
    // 64-bit and so on.
    // load 7, 3, 6, 2, 5, 1, 4, 0 in a 64-bit reg
    mov_imm(a, permute_const_reg, 0x0703060205010400);
    a->xvreplgr2vr_d(const_reg_xmm, permute_const_reg);
    a->xvsllwil_hu_bu(stPermReg_V_, const_reg_xmm, 0);
    a->xvpermi_d(stPermReg_V_, stPermReg_V_, 0x10);
    a->xvsllwil_wu_hu(stPermReg_V_, stPermReg_V_, 0);
  }
}

GCONV_INST_DEF_LASX_HEADER
GenConvKernel<SPATIAL_DIM, INST_SET>::genForLoadingWeights(la64::Emitter* a) {
  using WRegs = la64::VecX;
  int paddedICPerG = (this->C_per_G_ + 3) / 4 * 4;
  // load weights
  for (int r = 0; r < this->R_; ++r) {
    for (int s = 0; s < this->S_; ++s) {
      // For other cases, weights are too big to be kept in registers
      // and are loaded as they are used.
      if (this->C_per_G_ == 4 || this->C_per_G_ == 2) {
        a->xvld(
            WRegs(r * this->S_ + s),
            ptr(wghts_R_,
            (r * this->S_ + s) * this->K_per_G_ * GTogether_ *paddedICPerG * sizeof(int8_t)));  //TODO: maybe overflow
      }
    }
  }
}

GCONV_INST_DEF_LASX_HEADER GenConvKernel<SPATIAL_DIM, INST_SET>::storeResult(
    la64::Emitter* a) {
  using Ymm = la64::VecX;
  if (GTogether_ > 1) {
    // store with permutation
    a->xvperm_w(Ymm(9), Ymm(9), stPermReg_V_);
    if (this->accum_) {
      a->xvld(tmpReg1_V_, ptr(out_acts_R_));
      a->xvadd_w(Ymm(9), Ymm(9), tmpReg1_V_);
    }
    a->xvst(Ymm(9), ptr(out_acts_R_));
  } else {
    // horizontal add and store
    if (this->C_per_G_ == 8) {
      a->xvpickev_w(tmpReg1_V_, Ymm(8), Ymm(9));
      a->xvpickod_w(Ymm(9), Ymm(8), Ymm(9));
      a->xvadd_w(Ymm(9), Ymm(9), tmpReg1_V_);
      a->xvpermi_d(Ymm(9), Ymm(9), static_cast<asmjit::Imm>(0xd8));
      if (this->accum_) {
        a->xvld(tmpReg1_V_, ptr(out_acts_R_));
        a->xvadd_w(Ymm(9), Ymm(9), tmpReg1_V_);
      }
      a->xvst(Ymm(9), ptr(out_acts_R_));
    } else if (this->K_per_G_ == 16) {
      a->xvpickev_w(tmpReg1_V_, Ymm(8), Ymm(9));
      a->xvpickod_w(Ymm(9), Ymm(8), Ymm(9));
      a->xvadd_w(Ymm(9), Ymm(9), tmpReg1_V_);
      a->xvpermi_d(Ymm(9), Ymm(9), static_cast<asmjit::Imm>(0xd8));

      a->xvpickev_w(tmpReg1_V_, Ymm(6), Ymm(7));
      a->xvpickod_w(Ymm(7), Ymm(6), Ymm(7));
      a->xvadd_w(Ymm(7), Ymm(7), tmpReg1_V_);
      a->xvpermi_d(Ymm(7), Ymm(7), static_cast<asmjit::Imm>(0xd8));

      a->xvpickev_w(tmpReg1_V_, Ymm(4), Ymm(5));
      a->xvpickod_w(Ymm(5), Ymm(4), Ymm(5));
      a->xvadd_w(Ymm(5), Ymm(5), tmpReg1_V_);
      a->xvpermi_d(Ymm(5), Ymm(5), static_cast<asmjit::Imm>(0xd8));

      a->xvpickev_w(tmpReg1_V_, Ymm(2), Ymm(3));
      a->xvpickod_w(Ymm(3), Ymm(2), Ymm(3));
      a->xvadd_w(Ymm(3), Ymm(3), tmpReg1_V_);
      a->xvpermi_d(Ymm(3), Ymm(3), static_cast<asmjit::Imm>(0xd8));

      a->xvpickev_w(tmpReg1_V_, Ymm(7), Ymm(9));
      a->xvpickod_w(Ymm(9), Ymm(7), Ymm(9));
      a->xvadd_w(Ymm(9), Ymm(9), tmpReg1_V_);
      a->xvpermi_d(Ymm(9), Ymm(9), static_cast<asmjit::Imm>(0xd8));

      a->xvpickev_w(tmpReg1_V_, Ymm(3), Ymm(5));
      a->xvpickod_w(Ymm(5), Ymm(3), Ymm(5));
      a->xvadd_w(Ymm(5), Ymm(5), tmpReg1_V_);
      a->xvpermi_d(Ymm(5), Ymm(5), static_cast<asmjit::Imm>(0xd8));

      if (this->accum_) {
        a->xvld(tmpReg1_V_, ptr(out_acts_R_));
        a->xvadd_w(Ymm(9), Ymm(9), tmpReg1_V_);
        a->xvld(tmpReg1_V_, ptr(out_acts_R_, 32));
        a->xvadd_w(Ymm(5), Ymm(5), tmpReg1_V_);
      }
      a->xvst(Ymm(9), ptr(out_acts_R_));
      a->xvst(Ymm(5), ptr(out_acts_R_, 32));
    }
  }
}

GCONV_INST_DEF_LASX_HEADER GenConvKernel<SPATIAL_DIM, INST_SET>::storeOffset(
    la64::Emitter* a) {
  switch (this->C_per_G_) {
    case 2:
      // store 128-bits containing rowoffset for four groups
      if (this->accum_) {
        a->vld(tmpReg1_V_.half(), ptr(row_offset_R_));
        a->vadd_w(rowOffsetReg_V_.half(), rowOffsetReg_V_.half(), tmpReg1_V_.half());
      }
      a->vst(rowOffsetReg_V_.half(), ptr(row_offset_R_));
      break;
    case 4:
      // store 64-bits containing rowoffset for two groups
      if (this->accum_) {
        a->vld(tmpReg1_V_.half(), ptr(row_offset_R_));
        a->vadd_w(rowOffsetReg_V_.half(), rowOffsetReg_V_.half(), tmpReg1_V_.half());
      }
      a->fst_d(rowOffsetReg_V_.half().half(), ptr(row_offset_R_));
      break;
    case 8:
      if (this->accum_) {
        a->vld(tmpReg1_V_.half(), ptr(row_offset_R_));
        a->vadd_w(rowOffsetReg_V_.half(), rowOffsetReg_V_.half(), tmpReg1_V_.half());
      }
      a->fst_s(rowOffsetReg_V_.half().half(), ptr(row_offset_R_));
      break;
    case 16:
      // rowOffsetReg_V_[0:63] has sum for first 8 and
      // rowOffsetReg_V_[64:127] has sum for second 8
      // execute vphaddd twice to sum the two
      a->xvhaddw_d_w(rowOffsetReg_V_, rowOffsetReg_V_, rowOffsetReg_V_);  //TODO: assum not overflow
      a->xvhaddw_q_d(rowOffsetReg_V_, rowOffsetReg_V_, rowOffsetReg_V_);

      if (this->accum_) {
        a->vld(tmpReg1_V_.half(), ptr(row_offset_R_));
        a->vadd_w(rowOffsetReg_V_.half(), rowOffsetReg_V_.half(), tmpReg1_V_.half());
      }
      a->fst_s(rowOffsetReg_V_.half().half(), ptr(row_offset_R_));
      break;
    default:
      assert(0 && "not supported case");
  }
}

GCONV_INST_DEF_LASX_HEADER
GenConvKernel<SPATIAL_DIM, INST_SET>::genForSingleFilterPoint(
    la64::Emitter* a,
    int r,
    int s,
    int act_s,
    bool use_zero_reg) {
  using WRegs = la64::VecX;
  if (GTogether_ > 1) {
    if (this->C_per_G_ == 2) { // group together = 4
      if (use_zero_reg) {
        a->xvor_v(actReg_V_, zeroPTReg_V_, zeroPTReg_V_); // 32 * 8 bit zero points
      } else {
        a->ld_d(scratchReg1_, ptr(in_acts_R_, (act_s * this->C_) * sizeof(uint8_t)));
        a->xvreplgr2vr_d(actReg_V_, scratchReg1_);
      }
      // 8 * 16 bit activation to 8 * 32 bit activation( C_per_G = 2)
      // zero extend because vpmaddubsw and vpmaddwd together sum 4 consecutive
      // elements
      a->xvpermi_d(actReg_V_, actReg_V_, 0x10);
      a->xvsllwil_wu_hu(actReg_V_, actReg_V_, 0);
    } else if (this->C_per_G_ == 4) { // group together = 2
      if (use_zero_reg) {
        a->xvor_v(actReg_V_, zeroPTReg_V_, zeroPTReg_V_); // 32 * 8 bit zero points
      } else {
        a->ld_d(scratchReg1_, ptr(in_acts_R_, (act_s * this->C_) * sizeof(uint8_t)));
        a->xvreplgr2vr_d(actReg_V_, scratchReg1_);
      }
    }
    // row offset
    if (this->needRowOffset_) {
      genU8Sum4<INST_SET>(
          a, actReg_V_, rowOffsetReg_V_, oneReg16Bit_V_, tmpReg1_V_, tmpReg2_V_);
    }
    // 32 * int8 weight product 32 * uint8 activation -> 8
    // output(K_per_g * group_together)
    genU8I8S32FMA<INST_SET>(
        a,
        actReg_V_,
        WRegs(r * this->S_ + s),
        la64::VecX(9),
        oneReg16Bit_V_,
        tmpReg2_V_,  //Add
        tmpReg1_V_);
  } else {
    if (this->C_per_G_ == 8) {
      if (use_zero_reg) {
        a->xvor_v(actReg_V_, zeroPTReg_V_, zeroPTReg_V_);
      } else {
        a->ld_d(scratchReg1_, ptr(in_acts_R_, (act_s * this->C_) * sizeof(uint8_t)));
        a->xvreplgr2vr_d(actReg_V_, scratchReg1_);
      }
    } else {
      // this->C_per_G_ == 16
      if (use_zero_reg) {
        a->xvor_v(actReg_V_, zeroPTReg_V_, zeroPTReg_V_);
      } else {
        a->vld(actReg_V_.half(), ptr(in_acts_R_, act_s * this->C_ * sizeof(uint8_t)));
        a->xvpermi_q(actReg_V_, actReg_V_, 0x00);
      }
    }
    // row offset
    if (this->needRowOffset_) {
      genU8Sum8(a, actReg_V_, rowOffsetReg_V_, tmpReg1_V_);
    }
    int kLoopMultiplier = 32 / this->C_per_G_;
    for (int k = 0; k < kLoopIters_; ++k) {
      LDST_MACRO_INST(a, xvld,
          WRegs(0),
          ptr(wghts_R_,
          (((r * this->S_ + s) * this->K_per_G_) + k * kLoopMultiplier) * this->C_per_G_ * sizeof(int8_t)),
          scratchReg1_);

      // FMA result is not final reduction on C_per_G, producing 8 output in
      // which consectutive 2 elements if summedforms one final output over
      // K_Per_G dimension
      genU8I8S32FMA<INST_SET>(
          a, actReg_V_, WRegs(0), la64::VecX(9 - k), oneReg16Bit_V_, tmpReg1_V_, tmpReg2_V_);
    }
  }
}

#define GENCONVKERNEL_FUNCS(S, IN)                                       \
  template void GenConvKernel<S, IN>::genForLoadingWeights<IN>(          \
      la64::Emitter * a);                                                 \
  template void GenConvKernel<S, IN>::genConstForPermutations<IN>(       \
      la64::Emitter * a);                                                 \
  template void GenConvKernel<S, IN>::genForSingleFilterPoint<IN>(       \
      la64::Emitter * a, int r, int s, int act_s, bool use_zero_reg);     \
  template void GenConvKernel<S, IN>::storeResult<IN>(la64::Emitter * a); \
  template void GenConvKernel<S, IN>::storeOffset<IN>(la64::Emitter * a);
GENCONVKERNEL_FUNCS(1, inst_set_t::lasx)
GENCONVKERNEL_FUNCS(2, inst_set_t::lasx)
GENCONVKERNEL_FUNCS(3, inst_set_t::lasx)
#undef GENCONVKERNEL_FUNCS

template class GenConvKernel<1, inst_set_t::lasx>;
template class GenConvKernel<2, inst_set_t::lasx>;
template class GenConvKernel<3, inst_set_t::lasx>;

} // namespace fbgemm
