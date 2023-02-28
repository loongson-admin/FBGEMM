#ifndef INTRIN_LOONGARCH
#define INTRIN_LOONGARCH

#if defined(__loongarch_asx)

#define X86_2_LOONGARCH_INTRIN

#include <lasxintrin.h>
#include <lsxintrin.h>
#include <larchintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>

#define _MM_FROUND_TO_NEAREST_INT   0x00
#define _MM_FROUND_NO_EXC           0x08
typedef long long __m256i_u __attribute__ ((__vector_size__ (32), aligned(4)));
#define __DEFAULT_FN_ATTRS __attribute__((__gnu_inline__, __always_inline__, __artificial__))

static __m256i _mm256_set_epi8(char __b31, char __b30, char __b29, char __b28, char __b27, char __b26, char __b25, char __b24, char __b23, char __b22, char __b21, char __b20, char __b19, char __b18, char __b17, char __b16, char __b15, char __b14, char __b13, char __b12, char __b11, char __b10, char __b09, char __b08, char __b07, char __b06, char __b05, char __b04, char __b03, char __b02, char __b01, char __b00)
{
  return (__m256i)(v32i8){ __b00, __b01, __b02, __b03, __b04, __b05, __b06, __b07, __b08, __b09, __b10, __b11, __b12, __b13, __b14, __b15, __b16, __b17, __b18, __b19, __b20, __b21, __b22, __b23, __b24, __b25, __b26, __b27, __b28, __b29, __b30, __b31 };
}

static __m256i _mm256_castps_si256(__m256 __a)
{
  return (__m256i)__a;
}

static __m256i _mm256_set1_epi8(char __b)
{
  return (__m256i)(v32i8){ __b, __b, __b, __b, __b, __b, __b, __b,
                            __b, __b, __b, __b, __b, __b, __b, __b,
                            __b, __b, __b, __b, __b, __b, __b, __b,
                            __b, __b, __b, __b, __b, __b, __b, __b };
}

static __m256i _mm256_set1_epi16(short __w)
{
  return (__m256i)(v16i16){__w, __w, __w, __w,
                          __w, __w, __w, __w,
                          __w, __w, __w, __w,
                          __w, __w, __w, __w};
}

static __m256i _mm256_set1_epi32(int __i)
{
  return (__m256i)(v8i32){__i, __i, __i, __i, __i, __i, __i, __i};
}

static __m256i _mm256_set1_epi64(int64_t __k)
{
  return (__m256i)(v4i64){__k, __k, __k, __k};
}

static __inline __m256i __DEFAULT_FN_ATTRS
_mm256_set_epi64x(long long __a, long long __b, long long __c, long long __d)
{
  return (__m256i)(v4i64){ __d, __c, __b, __a };
}

static __m256i _mm256_set1_epi64x(long long __q)
{
  return _mm256_set_epi64x(__q, __q, __q, __q);
}

#define _mm256_srli_epi16(/*__m256i*/_1,  /*int*/ _2)     __lasx_xvsrli_h(_1, _2)

#define _mm256_srli_epi32(/*__m256i*/ _1, /*int*/ _2)     __lasx_xvsrli_w(_1, _2)

static __m256i _mm256_srli_epi64(__m256i _a, uint64_t _count)
{
  int64_t *n = (int64_t *) &_a;
  if ( _count > 63 )
  {
    _a = _mm256_set1_epi64x(0);
  }
  else
  {
    n[0] = n[0] >> _count;
    n[1] = n[1] >> _count;
    n[2] = n[2] >> _count;
    n[3] = n[3] >> _count;
  }
  return _a;
}

static __m256i _mm256_srli_si256(__m256i _a, int imm)
{
  if (imm > 15)
  {
    imm = 16;
  }
  __m128i *n = (__m128i *) &_a;
  int aImm = imm * 8;
  n[0] >>= aImm;
  n[1] >>= aImm;
  return _a;
}

#define _mm256_slli_epi32(/*__m256i*/ _1, /*int*/ _2)     __lasx_xvslli_w(_1, _2)

static __m256i _mm256_slli_si256(__m256i _a, int imm)
{
  if (imm > 15)
  {
    imm = 16;
  }
  __m128i *n = (__m128i *) &_a;
  int aImm = imm * 8;
  n[0] <<= aImm;
  n[1] <<= aImm;
  return _a;
}

#define _mm256_permute4x64_epi64(/*__m256i*/ _2, /*uint8_t*/ _3)    __lasx_xvpermi_d(_2, _3)

static __m256 _mm256_loadu_ps(float const *__p)
{
  return (__m256)__lasx_xvld(__p, 0);
}

static void _mm256_storeu_si256(__m256i *__p, __m256i __a)
{
    __lasx_xvst(__a, __p, 0);
}

static __m128i _mm_lddqu_si128(__m128i const *__p)
{
  return (__lsx_vld(__p, 0));
}

static __m256i _mm_lddqu_si256(__m256i const *__p)
{
  return (__lasx_xvld(__p, 0));
}

static __m128i _mm_set1_epi32(int __i)
{
  return (__m128i)(v4i32){ __i, __i, __i, __i };
}

static __m128i _mm256_cvtps_ph(__m256 _src, int32_t imm)
{
  (void)imm;
  __m256i tmp_hp = __lasx_xvfcvt_h_s(_src, _src);
  __m256i tmp = __lasx_xvpermi_d(tmp_hp, 0x08);
  return *(__m128i*)&tmp;
}

static void _mm_storeu_si128(__m128i *__p, __m128i __b)
{
  __lsx_vst(__b, __p, 0);
}

static __m256 _mm256_set1_ps(float __w)
{
  return (__m256)(v8f32){ __w, __w, __w, __w, __w, __w, __w, __w };
}

static __m256 _mm256_min_ps(__m256 _1, __m256 _2)
{
  return __lasx_xvfmin_s(_1, _2);
}

static __m256 _mm256_max_ps(__m256 _1, __m256 _2)
{
  return __lasx_xvfmax_s(_1, _2);
}

static __m128i _mm_loadu_si128(__m128i const *__p)
{
  return (__m128i)__lsx_vld(__p, 0);
}

static __m256 _mm256_cvtph_ps(__m128i _1)
{
  __m256i perm = __lasx_xvpermi_d(*(__m256i*)&_1, 0x10);
  return __lasx_xvfcvtl_s_h(perm);
}

static void _mm256_storeu_ps(float *__p, __m256 __a)
{
  __m256i _tmp = *(__m256i*)(&__a);
  __lasx_xvst(_tmp, __p, 0);
}

static __m256 _mm256_setzero_ps(void)
{
  return (__m256)(v8f32){ 0, 0, 0, 0, 0, 0, 0, 0 };
}

static __m256i _mm256_setzero_si256(void)
{
  return (__m256i){ 0LL, 0LL, 0LL, 0LL };
}


static __m256i _mm256_set_epi32(int __i0, int __i1, int __i2, int __i3,
                 int __i4, int __i5, int __i6, int __i7)
{
  return (__m256i)(v8i32){ __i7, __i6, __i5, __i4, __i3, __i2, __i1, __i0 };
}

static __m256i _mm256_set_epi16(short __s0, short __s1, short __s2, short __s3, short __s4, short __s5, short __s6, short __s7, short __s8, short __s9, short __s10, short __s11, short __s12, short __s13, short __s14, short __s15)
{
  return (__m256i)(v16i16){ __s15, __s14, __s13, __s12, __s11, __s10, __s9, __s8, __s7, __s6, __s5, __s4, __s3, __s2, __s1, __s0 };
}

static __m256i _mm256_loadu_si256(__m256i_u const *__p)
{
  return __lasx_xvld(__p, 0);
}

static __m128 _mm_loadu_ps(const float *__p)
{
  __m128i _tmp = __lsx_vld(__p, 0);
  return (__m128)(_tmp);
}

static __m256 _mm256_permutevar8x32_ps(__m256 val, __m256i offsets)
{
  __m256i tmp_ = __lasx_xvperm_w((__m256i)val, offsets);
  return (__m256)tmp_;
}

static __m256 _mm256_moveldup_ps(__m256 _src)
{
  __m256i index = _mm256_set_epi32(6, 6, 4, 4, 2, 2, 0, 0);
  __m256i tmp_ = __lasx_xvperm_w((__m256i)_src, index);
  return (__m256)tmp_;
}

static __m256i _mm256_mullo_epi32 (__m256i __a, __m256i __b)
{
  return (__m256i)((v8i32)__a * (v8i32)__b);
}

static __m256 _mm256_add_ps(__m256 __a, __m256 __b)
{
  return __lasx_xvfadd_s(__a, __b);
}

static __inline __m256 __DEFAULT_FN_ATTRS
_mm256_sub_ps(__m256 __a, __m256 __b)
{
  return __lasx_xvfsub_s(__a, __b);
}

static __m256 _mm256_mul_ps(__m256 __a, __m256 __b)
{
  return __lasx_xvfmul_s(__a, __b);
}

static __m256 _mm256_div_ps(__m256 __a, __m256 __b)
{
  return __lasx_xvfdiv_s(__a, __b);
}

static __m256i _mm256_add_epi32(__m256i _1, __m256i _2)
{
  return (__m256i)__lasx_xvadd_w(_1, _2);
}

static __m256i _mm256_add_epi64(__m256i _a, __m256i _b)
{
  return __lasx_xvadd_d(_a, _b);
}

static __m256i _mm256_sub_epi32(__m256i _a, __m256i _b)
{
  return (__m256i)__lasx_xvsub_w(_a, _b);
}

static __m256i _mm256_mul_epi32(__m256i _a, __m256i _b)
{
  return __lasx_xvmulwev_d_w(_a, _b);
}

static __m256i _mm256_cvtps_epi32(__m256 _src)
{
  return __lasx_xvftintrne_w_s(_src);
}

static __m256i _mm256_packs_epi32(__m256i _1, __m256i _2)
{
  __m256i res = __lasx_xvssrani_h_w(_2, _1, 0);
  return res;
}

static __m256i _mm256_packus_epi32(__m256i _1, __m256i _2)
{
  __m256i res = __lasx_xvssrani_hu_w(_2, _1, 0);
  return res;
}

static __m256i _mm256_packus_epi16(__m256i _1, __m256i _2)
{
  __m256i res = __lasx_xvssrani_bu_h(_2, _1, 0);
  return res;
}

static __m128i _mm_packus_epi16(__m128i _a, __m128i _b)
{
  return __lsx_vssrani_bu_h(_b, _a, 0);
}

static __m128i _mm_packus_epi32(__m128i _a, __m128i _b)
{
  return __lsx_vssrani_hu_w(_b, _a, 0);
}

static __m256i _mm256_adds_epi16(__m256i _1, __m256i _2)
{
  return __lasx_xvsadd_h(_1, _2);
}

static __m256i _mm256_min_epu8(__m256i _1, __m256i _2)
{
  return __lasx_xvmin_bu(_1, _2);
}

static __m256i _mm256_max_epu8(__m256i _1, __m256i _2)
{
  return __lasx_xvmax_bu(_1, _2);
}

static __m256i _mm256_permutevar8x32_epi32(__m256i val, __m256i offsets)
{
  return __lasx_xvperm_w(val, offsets);
}

static __m128i _mm256_castsi256_si128(__m256i _1)
{
  return *(__m128i*)&_1;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storel_epi64(__m128i *__p, __m128i __a)
{
  __lsx_vstelm_d(__a, __p, 0, 0);
}

static __m256i _mm256_maskload_epi32(int const *_X, __m256i _M)
{
  __m256i tmp = *(__m256i*)_X;
  int* a = (int*)&tmp;
  int32_t* __p = (int32_t *) &_M;
  int kk = 0;
  for (int i = 0; i < 8; i++)
  {
    kk = __p[i];
    if (kk < 0)
    {
      a[i] = _X[i];
    }
    else
    {
      a[i] = 0;
    }
  }
  __m256i ret = __lasx_xvld(&tmp, 0);
  return ret;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm256_maskstore_epi32(int *_p, __m256i _m, __m256i _a)
{
  int32_t *m = (int32_t *) &_m;
  int32_t *n = (int32_t *) &_a;
  for (int i = 0; i < 8; i++)
  {
    if ((m[i] >> 31) != 0)
    {
      _p[i] = n[i];
    }
  }
}

static __m128i _mm_maskload_epi32(int const *mem_addr, __m128i mask)
{
  __m128i tmp = __lsx_vld(mem_addr, 0);
  int32_t *p = (int32_t *)&tmp;
  int32_t *m = (int32_t *)&mask;
  for (int i = 0; i < 4; i++)
  {
    int k = m[i] >> 31;
    if (0 == k)
    {
      p[i] = 0;
    }
  }
  return tmp;
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm_maskstore_epi32(int *_p, __m128i _m, __m128i _a)
{
  int32_t *m = (int32_t *) &_m;
  int32_t *n = (int32_t *) &_a;
  for (int i = 0; i < 4; i++)
  {
    if ((m[i] >> 31) != 0)
    {
      _p[i] = n[i];
    }
  }
}

static __m256 _mm256_maskload_ps(float const *_p, __m256i _m)
{
  __m256 tmp = *(__m256*)_p;
  float* a = (float*)&tmp;
  int32_t* __p = (int32_t*) &_m;
  int k = 0;
  for (int i = 0; i < 8; i++)
  {
    k = __p[i];
    if (k < 0)
    {
      a[i] = _p[i];
    }
    else
    {
      a[i] = 0;
    }
  }
  __m256i ret = __lasx_xvld(&tmp, 0);
  return (__m256)(ret);
}

static __inline void __DEFAULT_FN_ATTRS
_mm256_maskstore_ps(float *_p, __m256i _m, __m256 _a)
{
  float *m = (float *) &_a;
  int k = 0;
  for (int i = 0; i < 8; i++)
  {
    k = (((v8i32)_m)[i] >> 31);
    if (k != 0)
    {
      _p[i] = m[i];
    }
  }
}

static __m128 _mm_maskload_ps(float const *_p, __m128i _m)
{
  __m128i tmp = *(__m128i*)_p;
  float * a = (float*)&tmp;
  int32_t * __p =(int32_t*) &_m;
  int k = 0;
  for (int i = 0; i < 4; i++)
  {
    k = __p[i];
    if (k < 0)
    {
      a[i] = _p[i];
    }
    else
    {
      a[i] = 0;
    }
  }
  __m128i ret = __lsx_vld(&tmp, 0);
  return *(__m128*)(&ret);
}

static __inline void __DEFAULT_FN_ATTRS
_mm_maskstore_ps(float *_p, __m128i _m, __m128 _a)
{
  float *m = (float *) &_a;
  int k = 0;
  for (int i = 0; i < 4; i++)
  {
    k = (((v4i32)_m)[i] >> 31);
    if (k != 0)
    {
      _p[i] = m[i];
    }
  }
}

static __m256 _mm256_fmadd_ps(__m256 _1, __m256 _2, __m256 _3)
{
  return __lasx_xvfmadd_s(_1, _2, _3);
}

static __m256i _mm256_shuffle_epi8(__m256i _1, __m256i _2)
{
  char *m = (char *) &_1;
  char *n = (char *) &_2;
  __m256i _tmp;
  int tmp = 0;
  char *h = (char *) &_tmp;
  for (int i = 0; i < 16; i++)
  {
    if ((n[i] >> 7) == 0)
    {
      tmp = (n[i] & 0xF);
      h[i] = m[tmp];
      tmp = 0;
    }
    else
    {
      h[i] = 0;
    }
    if ((n[i + 16] >> 7) == 0)
    {
      tmp = (n[i + 16] & 0xF);
      h[i + 16] = m[tmp + 16];
      tmp = 0;
    }
    else
    {
      h[i + 16] = 0;
    }
  }
  return _tmp;
}

static __m256i _mm256_unpacklo_epi32(__m256i _1, __m256i _2)
{
  return __lasx_xvilvl_w(_2, _1);
}

static __m256i _mm256_unpacklo_epi64(__m256i _1, __m256i _2)
{
  return __lasx_xvilvl_d(_2, _1);
}

static __m256i _mm256_unpackhi_epi32(__m256i _1, __m256i _2)
{
  return __lasx_xvilvh_w(_2, _1);
}

static __m256i _mm256_unpackhi_epi64(__m256i _1, __m256i _2)
{
  return __lasx_xvilvh_d(_2, _1);
}

static __m256i _mm256_permute2x128_si256(__m256i _1, __m256i _2, int8_t _3)
{
  __m256i _tmp = _mm256_set1_epi64(0);
  int64_t *n = (int64_t *) &_tmp;
  int64_t *m = (int64_t *) &_1;
  int64_t *h = (int64_t *) &_2;

  switch(_3 & 0x03){
    case 0:
      n[0] = m[0];
      n[1] = m[1];
      break;
    case 1:
      n[0] = m[2];
      n[1] = m[3];
      break;
    case 2:
      n[0] = h[0];
      n[1] = h[1];
      break;
    case 3:
      n[0] = h[2];
      n[1] = h[3];
      break;
  }
  if (_3 & 0x08)
  {
    n[0] = 0;
    n[1] = 0;
  }

  _3 >>= 4;
  switch(_3 & 0x03){
    case 0:
      n[2] = m[0];
      n[3] = m[1];
      break;
    case 1:
      n[2] = m[2];
      n[3] = m[3];
      break;
    case 2:
      n[2] = h[0];
      n[3] = h[1];
      break;
    case 3:
      n[2] = h[2];
      n[3] = h[3];
      break;
  }
  if (_3 & 0x08)
  {
    n[2] = 0;
    n[3] = 0;
  }

  return _tmp;
}

static void _mm_mfence()
{
  __dbar(0);
}

static void _mm_clflush(void const * _p)
{
  (void)_p;
}

static __m256i _mm256_maddubs_epi16(__m256i _a, __m256i _b)
{
  __m256i _tmp1 = __lasx_xvmulwev_h_bu_b(_a, _b);
  __m256i _tmp2 = __lasx_xvmulwod_h_bu_b(_a, _b);
  _tmp1 = __lasx_xvsadd_h(_tmp1, _tmp2);
  return _tmp1;
}

static __m256i _mm256_madd_epi16(__m256i _a, __m256i _b)
{
  __m256i _tmp1 = _mm256_setzero_si256();
  __m256i _tmp2 = _mm256_setzero_si256();
  _tmp1 = __lasx_xvmaddwev_w_h(_tmp1, _a, _b);
  _tmp2 = __lasx_xvmaddwod_w_h(_tmp2, _a, _b);
  return __lasx_xvadd_w(_tmp1, _tmp2);
}


static __m256i _mm256_lddqu_si256(__m256i const *_p)
{
  return __lasx_xvld(_p, 0);
}

static __m256i _mm256_unpacklo_epi8(__m256i _a, __m256i _b)
{
  return __lasx_xvilvl_b(_b, _a);
}

static __m256i _mm256_unpackhi_epi8(__m256i _a, __m256i _b)
{
  return __lasx_xvilvh_b(_b, _a);
}

static __m256i _mm256_unpacklo_epi16(__m256i _a, __m256i _b)
{
  return __lasx_xvilvl_h(_b, _a);
}

static __m256i _mm256_unpackhi_epi16(__m256i _a, __m256i _b)
{
  return __lasx_xvilvh_h(_b, _a);
}

#define _mm256_extract_epi64(__a, __imm) __lasx_xvpickve2gr_d(__a, (__imm & 0x3))


static __inline __m256i __DEFAULT_FN_ATTRS
_mm256_load_si256(__m256i const *__p)
{
  return __lasx_xvld(__p, 0);
}

static __m256i _mm256_permute2f128_si256(__m256i _a, __m256i _b, int8_t _imm)
{
  __m256i _tmp;
  int64_t *n = (int64_t *) &_tmp;
  int64_t *src1 = (int64_t *) &_a;
  int64_t *src2 = (int64_t *) &_b;

  switch(_imm & 0x03){
    case 0:
      n[0] = src1[0];
      n[1] = src1[1];
      break;
    case 1:
      n[0] = src1[2];
      n[1] = src1[3];
      break;
    case 2:
      n[0] = src2[0];
      n[1] = src2[1];
      break;
    case 3:
      n[0] = src2[2];
      n[1] = src2[3];
      break;
  }
  if (_imm & 0x08)
  {
    n[0] = 0;
    n[1] = 0;
  }

  _imm >>= 4;
  switch(_imm & 0x03){
    case 0:
      n[2] = src1[0];
      n[3] = src1[1];
      break;
    case 1:
      n[2] = src1[2];
      n[3] = src1[3];
      break;
    case 2:
      n[2] = src2[0];
      n[3] = src2[1];
      break;
    case 3:
      n[2] = src2[2];
      n[3] = src2[3];
      break;
  }
  if (_imm & 0x08)
  {
    n[2] = 0;
    n[3] = 0;
  }

  return _tmp;
}

static __inline void __DEFAULT_FN_ATTRS
_mm256_store_si256(__m256i *__p, __m256i __a)
{
  __lasx_xvst(__a, __p, 0);
}

static __inline void __DEFAULT_FN_ATTRS
_mm256_store_ps(float *__p, __m256 __a)
{
  *(__m256 *)__p = __a;
}

static __m128i _mm256_extractf128_si256(__m256i _a, int offset)
{
  __m128i _tmp;
  int64_t *n = (int64_t *) &_tmp;
  int64_t *m = (int64_t *) &_a;
  int kk = offset & 0x1;
  switch(kk){
    case 0:
      n[0] = m[0];
      n[1] = m[1];
      break;
    case 1:
      n[0] = m[2];
      n[1] = m[3];
      break;
  }
  return _tmp;
}

static __m256i _mm256_and_si256(__m256i _a, __m256i _b)
{
  return __lasx_xvand_v(_a, _b);
}

static __m256i _mm256_or_si256(__m256i _a, __m256i _b)
{
  return __lasx_xvor_v(_b, _a);
}

static __m256i _mm256_cvtepu8_epi16(__m128i _a)
{
  return __lasx_vext2xv_hu_bu(*(__m256i*)&_a);
}

static __m256i _mm256_cvtepu8_epi32(__m128i _a)
{
  return __lasx_vext2xv_wu_bu(*(__m256i*)&_a);
}

static __m256i _mm256_cvtepu16_epi32(__m128i _src)
{
  return __lasx_vext2xv_wu_hu(*(__m256i*)&_src);
}

static __m256i _mm256_cvtepi8_epi32(__m128i _a)
{
  return __lasx_vext2xv_w_b(*(__m256i*)&_a);
}

static __m256 _mm256_cvtepi32_ps(__m256i _src)
{
  return __lasx_xvffint_s_w(_src);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_loadl_epi64(__m128i const *__p)
{
  struct __mm_loadl_epi64_struct {
    long long __u;
  } __attribute__((__packed__, __may_alias__));
  return (__m128i) { ((struct __mm_loadl_epi64_struct*)__p)->__u, 0};
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_set1_epi64x(long long __q)
{
  return (__m128i){ __q, __q };
}

static __m256i _mm256_castsi128_si256(__m128i _a)
{
  return *(__m256i*)&_a;
}

static __m256i _mm256_insertf128_si256(__m256i _a, __m128i _b, uint8_t _imm)
{
  __m256i _tmp;
  int64_t *n = (int64_t *) &_a;
  int64_t *m = (int64_t *) &_b;
  int64_t *h = (int64_t *) &_tmp;
  int kk = _imm & 0x1;
  switch(kk){
    case 0:
      h[0] = m[0];
      h[1] = m[1];
      h[2] = n[2];
      h[3] = n[3];
      break;
    case 1:
      h[0] = n[0];
      h[1] = n[1];
      h[2] = m[0];
      h[3] = m[1];
      break;
  }
  return _tmp;
}

static __m128 _mm_set1_ps(float __w)
{
  return (__m128){ __w, __w, __w, __w };
}

static __m256 _mm256_castps128_ps256(__m128 _a)
{
  return *(__m256*)&_a;
}

static __m256 _mm256_insertf128_ps(__m256 _a, __m128 _b, uint8_t _imm)
{
  __m256 _tmp;
  double *n = (double *) &_a;
  double *m = (double *) &_b;
  double *h = (double *) &_tmp;
  int kk = _imm & 0x1;
  switch(kk){
    case 0:
      h[0] = m[0];
      h[1] = m[1];
      h[2] = n[2];
      h[3] = n[3];
      break;
    case 1:
      h[0] = n[0];
      h[1] = n[1];
      h[2] = m[0];
      h[3] = m[1];
      break;
  }
  return _tmp;
}

static __m128 _mm_unpacklo_ps(__m128 _a, __m128 _b)
{
  __m128i _1 = *(__m128i*)(&_a);
  __m128i _2 = *(__m128i*)(&_b);
  __m128i _ret = __lsx_vilvl_w(_2, _1);
  return *(__m128*)&(_ret);
}

static __m128 _mm_unpackhi_ps(__m128 _a, __m128 _b)
{
  __m128i _1 = *(__m128i*)(&_a);
  __m128i _2 = *(__m128i*)(&_b);
  __m128i _ret = __lsx_vilvh_w(_2, _1);
  return *(__m128*)&(_ret);
}

static __m128 _mm_movelh_ps(__m128 _a, __m128 _b)
{
  __m128 _tmp;
  float *h = (float *) &_tmp;
  h[0] = ((v4f32)_a)[0];
  h[1] = ((v4f32)_a)[1];
  h[2] = ((v4f32)_b)[0];
  h[3] = ((v4f32)_b)[1];
  return _tmp;
}

static __m128 _mm_movehl_ps(__m128 _a, __m128 _b)
{
  __m128 _tmp;
  float *h = (float *) &_tmp;
  h[0] = ((v4f32)_b)[2];
  h[1] = ((v4f32)_b)[3];
  h[2] = ((v4f32)_a)[2];
  h[3] = ((v4f32)_a)[3];
  return _tmp;
}

#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do { \
  __m128 tmp3, tmp2, tmp1, tmp0; \
  tmp0 = _mm_unpacklo_ps((row0), (row1)); \
  tmp2 = _mm_unpacklo_ps((row2), (row3)); \
  tmp1 = _mm_unpackhi_ps((row0), (row1)); \
  tmp3 = _mm_unpackhi_ps((row2), (row3)); \
  (row0) = _mm_movelh_ps(tmp0, tmp2); \
  (row1) = _mm_movehl_ps(tmp2, tmp0); \
  (row2) = _mm_movelh_ps(tmp1, tmp3); \
  (row3) = _mm_movehl_ps(tmp3, tmp1); \
} while (0)

static __inline__ void __DEFAULT_FN_ATTRS
_mm_storeu_ps(float *__p, __m128 __a)
{
  __m128i __b = *(__m128i*)(&__a);
  __lsx_vst(__b, __p, 0);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS
_mm_load_si128(__m128i const *__p)
{
  return __lsx_vld(__p, 0);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS
_mm_setzero_ps(void)
{
  return (__m128){ 0, 0, 0, 0 };
}

static __m256 _mm256_unpacklo_ps(__m256 _a, __m256 _b)
{
  __m256 _tmp;
  float *h = (float *) &_tmp;
  h[0] = ((v8f32)_a)[0];
  h[1] = ((v8f32)_b)[0];
  h[2] = ((v8f32)_a)[1];
  h[3] = ((v8f32)_b)[1];
  h[4] = ((v8f32)_a)[4];
  h[5] = ((v8f32)_b)[4];
  h[6] = ((v8f32)_a)[5];
  h[7] = ((v8f32)_b)[5];
  return _tmp;
}

static __m256 _mm256_unpackhi_ps(__m256 _a, __m256 _b)
{
  __m256 _tmp;
  float *h = (float *) &_tmp;
  h[0] = ((v8f32)_a)[2];
  h[1] = ((v8f32)_b)[2];
  h[2] = ((v8f32)_a)[3];
  h[3] = ((v8f32)_b)[3];
  h[4] = ((v8f32)_a)[6];
  h[5] = ((v8f32)_b)[6];
  h[6] = ((v8f32)_a)[7];
  h[7] = ((v8f32)_b)[7];
  return _tmp;
}

static __m256 _mm256_shuffle_ps(__m256 _a, __m256 _b, uint8_t _imm)
{
  __m256 dst;
  uint32_t* p = (uint32_t*)&dst;
  __m128* a = (__m128*)&_a;
  __m128* b = (__m128*)&_b;
  uint32_t* pa1 = (uint32_t*)a;
  a++;
  uint32_t* pa2 = (uint32_t*)a;
  uint32_t* pb1 = (uint32_t*)b;
  b++;
  uint32_t* pb2 = (uint32_t*)b;

  p[0] = pa1[_imm & 0x3];
  p[1] = pa1[((_imm & 0xC) >> 2)];
  p[2] = pb1[((_imm & 0x30) >> 4)];
  p[3] = pb1[((_imm & 0xC0) >> 6)];
  p[4] = pa2[_imm & 0x3];
  p[5] = pa2[((_imm & 0xC) >> 2)];
  p[6] = pb2[((_imm & 0x30) >> 4)];
  p[7] = pb2[((_imm & 0xC0) >> 6)];

  return dst;

}

static __m256 _mm256_permute2f128_ps(__m256 _a, __m256 _b, uint8_t _imm)
{
  __m256 dst;
  __m128* p1 = (__m128*)&_a;
  __m128* p2 = (__m128*)&_b;
  __m128* p3 = (__m128*)&dst;
  if (_imm & 0x8)
  {
    p3[0] = (__m128){0, 0, 0, 0};
  }
  else
  {
    switch(_imm & 0x3)
    {
      case 0:
        p3[0] = p1[0];
        break;
      case 1:
        p3[0] = p1[1];
        break;
      case 2:
        p3[0] = p2[0];
        break;
      case 3:
        p3[0] = p2[1];
        break;
    }
  }
  if (_imm & 0x80)
  {
    p3[1] = (__m128){0, 0, 0, 0};
  }
  else
  {
    switch((_imm & 0x30) >> 4)
    {
      case 0:
        p3[1] = p1[0];
        break;
      case 1:
        p3[1] = p1[1];
        break;
      case 2:
        p3[1] = p2[0];
        break;
      case 3:
        p3[1] = p2[1];
        break;
    }
  }
  return dst;
}

static __m256i _mm256_blend_epi32(__m256i _a, __m256i _b, uint8_t _imm)
{
  __m256i _tmp;
  int32_t *n = (int32_t *) &_tmp;
  for (int i = 0; i < 8; i++)
  {
    switch(i){
      case 7:
	_imm &= 0x80;
	break;
      case 6:
	_imm &= 0x40;
	break;
      case 5:
	_imm &= 0x20;
	break;
      case 4:
	_imm &= 0x10;
	break;
      case 3:
	_imm &= 0x8;
	break;
      case 2:
	_imm &= 0x4;
	break;
      case 1:
	_imm &= 0x2;
	break;
      case 0:
	_imm &= 0x1;
	break;
      }
    switch(_imm){
      case 0:
        n[i] = ((v8i32)_a)[i];
        break;
      case 1:
        n[i] = ((v8i32)_b)[i];
        break;
      }
  }
  return _tmp;
}

static __m256i _mm256_min_epi32(__m256i _a, __m256i _b)
{
  return __lasx_xvmin_w(_a, _b);
}

static __m256i _mm256_max_epi32(__m256i _a, __m256i _b)
{
  return __lasx_xvmax_w(_a, _b);
}

static __m256 _mm256_broadcast_ss (float const * mem_addr)
{
  return (__m256)__lasx_xvldrepl_w(mem_addr, 0);
}

static __m128 _mm_broadcast_ss (float const * mem_addr)
{
  return (__m128)__lsx_vldrepl_w(mem_addr, 0);
}

static __m256 _mm256_broadcastss_ps (__m128 a)
{
  int32_t *p = (int32_t *) &a;
  return (__m256)__lasx_xvreplgr2vr_w(*p);
}

static void _mm_prefetch(char const* p){
  (void)p;
}

#endif /* defined(__loongarch_asx).  */

#endif //INTRIN_LOONGRACH
