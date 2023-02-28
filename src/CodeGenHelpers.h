/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <asmjit/asmjit.h>
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace la64 = asmjit::la64;

template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<instSet == inst_set_t::lasx, int>::type = 0>
void gen16BitVectorOne(la64::Emitter* a, T dest) {
  a->xvseq_h(dest, dest, dest);
  a->xvsrli_h(dest, dest, 15);
}

//TODO: support Mem
template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<instSet == inst_set_t::lasx, int>::type = 0>
void emitLoadDWord(la64::Emitter* a, T dest, const la64::Mem& ptr) {
  a->xvld(dest, ptr);
}

template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<instSet == inst_set_t::lasx, int>::type = 0>
void emitLoadDWord(la64::Emitter* a, T dest, const la64::Gp src, int offset) {
  a->xvld(dest, src, offset);  //TODO: T
}

template <
    inst_set_t instSet,
    typename T,
    typename std::enable_if<instSet == inst_set_t::lasx, int>::type = 0>
void emitExtractHalfVector(
    la64::Emitter* a,
    la64::VecV half,
    la64::VecX vec,
    int idx) {
  a->xvxor_v(half, half, half);
  a->xvpermi_q(half, vec, (2<<4)+idx);  //ui8[5:4]=2, ui8[1:0]=idx
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, la64::VecX>::value, int>::type = 0>
void gen8BitVectorOne(la64::Emitter* a, T dest) {
  a->xvseq_h(dest, dest, dest);
  a->xvneg_b(dest, dest);
}

template <
    inst_set_t INST_SET,
    typename std::enable_if<
        INST_SET == inst_set_t::lasx,
        int>::type = 0>
void genU8I8S32FMA(
    la64::Emitter* a,
    typename simd_info<INST_SET>::vec_reg_t aReg,
    typename simd_info<INST_SET>::vec_reg_t bReg,
    typename simd_info<INST_SET>::vec_reg_t cReg,
    typename simd_info<INST_SET>::vec_reg_t oneReg16Bit,
    typename simd_info<INST_SET>::vec_reg_t tmpReg1,  //add a tmp reg
    typename simd_info<INST_SET>::vec_reg_t tmpReg) {
  a->xvmulwev_h_bu_b (tmpReg, aReg, bReg);
  a->xvmulwod_h_bu_b (tmpReg1, aReg, bReg);
  a->xvsadd_h(tmpReg, tmpReg, tmpReg1);
  a->xvmulwev_w_h (tmpReg1, oneReg16Bit, tmpReg);
  a->xvmaddwod_w_h(tmpReg1, oneReg16Bit, tmpReg);
  a->xvadd_w(cReg, tmpReg1, cReg);
}

template <
    inst_set_t INST_SET,
    typename std::enable_if<
        INST_SET == inst_set_t::lasx,
        int>::type = 0>
void genU8Sum4(
    la64::Emitter* a,
    typename simd_info<INST_SET>::vec_reg_t src,
    typename simd_info<INST_SET>::vec_reg_t dest,
    typename simd_info<INST_SET>::vec_reg_t oneReg16Bit,
    typename simd_info<INST_SET>::vec_reg_t tmpReg1,
    typename simd_info<INST_SET>::vec_reg_t tmpReg) {
  la64::VecX tmpReg2 = la64::xr17;
  gen8BitVectorOne(a, tmpReg);
  a->xvmulwev_h_bu_b (tmpReg1, src, tmpReg);
  a->xvmulwod_h_bu_b (tmpReg2, src, tmpReg);
  a->xvsadd_h(tmpReg1, tmpReg1, tmpReg2);
  a->xvmulwev_w_h (tmpReg, tmpReg1, oneReg16Bit);
  a->xvmaddwod_w_h(tmpReg, tmpReg1, oneReg16Bit);
  a->xvadd_w(dest, tmpReg, dest);
}

template <typename T>
void genU8Sum8(la64::Emitter* a, T src, T dest, T tmpReg) {
  a->xvxor_v(tmpReg, tmpReg, tmpReg);
  a->xvabsd_bu(tmpReg, src, tmpReg);
  a->xvhaddw_hu_bu(tmpReg, tmpReg, tmpReg);
  a->xvhaddw_wu_hu(tmpReg, tmpReg, tmpReg);
  a->xvhaddw_du_wu(tmpReg, tmpReg, tmpReg);
  a->xvslli_d(tmpReg, tmpReg, 48);
  a->xvsrli_d(tmpReg, tmpReg, 48);
  a->xvadd_w(dest, tmpReg, dest);
}

template <typename T>
void broadcast8Bit(la64::Emitter* a, la64::Gp src, T dest) {
  // move src to dest
  a->xvreplgr2vr_b(dest, src);
}

template <typename TGp>
static void mov_imm_general(la64::Emitter *a, const TGp &dst, uint64_t imm) {
  //64bit = 12bit_4 + 20bit_3 + 20bit_2 + 12bit_1
  uint32_t imm_12bit_1 = imm & 0xfff;
  uint32_t imm_20bit_2 = (imm>>12) & 0xfffff;
  uint32_t signBit12 = 0x800;
  uint32_t signBit20 = 0x80000;

  if( 0 != (imm_20bit_2 & signBit20) ) {
    imm_20bit_2 |= 0xfff00000;  //sign extend
  }

  a->lu12i_w(dst, imm_20bit_2);
  a->ori(dst, dst, imm_12bit_1);

  // if high 32bit not 0
  if (imm & 0xffffffff80000000) {
    uint32_t imm_20bit_3 = (imm>>32) & 0xfffff;
    if( 0 != (imm_20bit_3 & signBit20) ) {
      imm_20bit_3 |= 0xfff00000;  //sign extend
    }
    a->lu32i_d(dst, imm_20bit_3);

    uint32_t imm_12bit_4 = (imm>>52) & 0xfff;
    if( 0 != (imm_12bit_4 & signBit12) ) {
      imm_12bit_4 |= 0xfffff000;  //sign extend
    }
    a->lu52i_d(dst, dst, imm_12bit_4);
  }

  return;
}

template <typename TGp, typename T, typename std::enable_if<std::is_unsigned<T>::value, std::nullptr_t>::type = nullptr>
void mov_imm(la64::Emitter *a, const TGp &dst, T imm) {
  uint64_t bit_ptn = static_cast<uint64_t>(imm);

  //si12
  if ((bit_ptn >> 11) == 0) {
    a->addi_d(dst, la64::zero,(int32_t)bit_ptn);
  }
  //si16<<16
  else if ( (bit_ptn & 0xffff) == 0 &&
            (bit_ptn & 0x80000000) == 0 &&
            (bit_ptn >> 32) == 0 ) {
    a->addu16i_d(dst, la64::zero, (int32_t)(bit_ptn >> 16));
  }
  else {
    mov_imm_general(a, dst, bit_ptn);
  }

  return;
}

template <typename TGp, typename T, typename std::enable_if<std::is_signed<T>::value, std::nullptr_t>::type = nullptr>
void mov_imm(la64::Emitter *a, const TGp &dst, T imm) {
  int64_t bit_ptn = static_cast<int64_t>(imm);

  //si12
  int64_t s12min = -1*(1<<11);
  int64_t s12max = 0x7ff;
  if ( (bit_ptn >= s12min)  && (bit_ptn <= s12max) ) {
    a->addi_d(dst, la64::zero, (int32_t)bit_ptn);
    return;
  }

  //si16<<16
  uint32_t signBit = bit_ptn & 0x80000000;
  uint32_t upper32 = bit_ptn >> 32;

  if ( (bit_ptn & 0xffff) == 0 &&
       (  (signBit == 0 && upper32 == 0)
    ||(signBit != 0 && upper32 == 0xffffffff)
       ) ){
    a->addu16i_d(dst, la64::zero, (int32_t)(bit_ptn >> 16));
    return;
  }

  mov_imm_general(a, dst, (uint64_t)bit_ptn);
  return;
}

//mem = base + offset
#define LDST_MACRO_INST(emitp, inst, dst, mem, tmp) do { \
    int64_t offset = mem.offset(); \
    if ((offset>=-2048) && (offset <= 2047)) \
      emitp->inst(dst, mem); \
    else { \
      mov_imm(emitp, tmp, offset); \
      emitp->add_d(tmp, la64::GpX(mem.baseId()), tmp); \
      emitp->inst(dst, ptr(tmp)); \
    } \
} while(0)

} // namespace fbgemm
