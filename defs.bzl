# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def get_fbgemm_base_srcs():
    return [
        "src/GenerateI8Depthwise.cc",
        "src/RefImplementations.cc",
        "src/Utils.cc",
    ]

def get_fbgemm_generic_srcs(with_base = False):
    return [
        "src/EmbeddingSpMDM.cc",
        "src/EmbeddingSpMDMNBit.cc",
        "src/ExecuteKernel.cc",
        "src/ExecuteKernelU8S8.cc",
        "src/Fbgemm.cc",
        "src/FbgemmBfloat16Convert.cc",
        "src/FbgemmConv.cc",
        "src/FbgemmFPCommon.cc",
        "src/FbgemmFP16.cc",
        "src/FbgemmFloat16Convert.cc",
        "src/FbgemmI64.cc",
        "src/FbgemmSparseDense.cc",
        "src/FbgemmI8Spmdm.cc",
        "src/GenerateKernelDirectConvU8S8S32ACC32.cc",
        "src/GenerateKernel.cc",
        "src/GenerateKernelU8S8S32ACC16.cc",
        "src/GenerateKernelU8S8S32ACC32.cc",
        "src/GroupwiseConv.cc",
        "src/GroupwiseConvAcc32Lasx.cc",
        "src/PackAMatrix.cc",
        "src/PackAWithIm2Col.cc",
        "src/PackAWithQuantRowOffset.cc",
        "src/PackAWithRowOffset.cc",
        "src/PackBMatrix.cc",
        "src/PackMatrix.cc",
        "src/PackWeightMatrixForGConv.cc",
        "src/PackWeightsForConv.cc",
        "src/QuantUtils.cc",
        "src/RowWiseSparseAdagradFused.cc",
        "src/SparseAdagrad.cc",
        "src/spmmUtils.cc",
        "src/TransposeUtils.cc",
    ] + (get_fbgemm_base_srcs() if with_base else [])

def get_fbgemm_public_headers():
    return [
        "include/fbgemm/ConvUtils.h",
        "include/fbgemm/Fbgemm.h",
        "include/fbgemm/FbgemmBuild.h",
        "include/fbgemm/FbgemmConvert.h",
        "include/fbgemm/FbgemmEmbedding.h",
        "include/fbgemm/FbgemmFP16.h",
        "include/fbgemm/FbgemmFPCommon.h",
        "include/fbgemm/FbgemmI64.h",
        "include/fbgemm/FbgemmI8DepthwiseLasx.h",
        "include/fbgemm/FbgemmI8Spmdm.h",
        "include/fbgemm/FbgemmPackMatrixB.h",
        "include/fbgemm/FbgemmSparse.h",
        "include/fbgemm/OutputProcessing-inl.h",
        "include/fbgemm/PackingTraits-inl.h",
        "include/fbgemm/QuantUtils.h",
        "include/fbgemm/QuantUtilsLasx.h",
        "include/fbgemm/spmmUtils.h",
        "include/fbgemm/spmmUtilsLasx.h",
        "include/fbgemm/Utils.h",
        "include/fbgemm/UtilsLasx.h",
        "include/fbgemm/Types.h",
    ]

def get_fbgemm_lasx_srcs():
    return [
        #All the source files that either use lasx instructions statically
        "src/EmbeddingSpMDMLasx.cc",
        "src/FbgemmBfloat16ConvertLasx.cc",
        "src/FbgemmFloat16ConvertLasx.cc",
        "src/FbgemmI8Depthwise3DLasx.cc",
        "src/FbgemmI8DepthwiseLasx.cc",
        "src/FbgemmI8DepthwisePerChannelQuantLasx.cc",
        "src/FbgemmSparseDenseLasx.cc",
        "src/FbgemmSparseDenseInt8Lasx.cc",
        "src/OptimizedKernelsLasx.cc",
        "src/PackDepthwiseConvMatrixLasx.cc",
        "src/QuantUtilsLasx.cc",
        "src/spmmUtilsLasx.cc",
        "src/UtilsLasx.cc",
    ]

def get_fbgemm_inline_lasx_srcs():
    return [
        "src/FbgemmFP16UKernelsIntrinsicLasx.cc",
    ]

def get_fbgemm_tests(skip_tests = []):
    return native.glob(["test/*Test.cc"], exclude = skip_tests)
