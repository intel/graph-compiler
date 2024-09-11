#! /bin/bash

export CASE_DIR=$(pwd)/test/benchgc/cases

FAIL=0
set -e

# bf16
python3 -m benchgc --verbose 0 --driver linalg --case matmul --md 0:32x128xbf16 --md 1:128x64xbf16 --md 2:32x64xbf16 --cast cast_signed || FAIL=1

# f32

# reduce

python3 -m benchgc --verbose 0 --driver linalg --case reduce.add --md 0:128x64x8xf32 --md 1:128xf32 --dimensions=1 --dimensions=2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case reduce.mul --md 0:128x8xf32 --md 1:128xf32 --dimensions=1 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case reduce.max --md 0:128x64x8xf32 --md 1:128xf32 --dimensions=1 --dimensions=2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case reduce.min --md 0:128x64x8xf32 --md 1:128xf32 --dimensions=1 --dimensions=2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case reduce.l1 --md 0:128x64x8xf32 --md 1:128xf32 --dimensions=1 --dimensions=2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case reduce.l2_square --md 0:128x64x8xf32 --md 1:128xf32 --dimensions=1 --dimensions=2 || FAIL=1

# misc
python3 -m benchgc --verbose 0 --driver linalg --case fill --md 0:f32 --md 1:32x4096xf32 --cmp 1:P:0:0 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case copy --md 0:1024x1024xf32 --md 1:1024x1024xbf16 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case broadcast --md 0:1024xf32 --md 1:2x32x1024xf32 --dimensions=0 --dimensions=1 || FAIL=1

# matmul
python3 -m benchgc --verbose 0 --driver linalg --case batch_matmul --md 0:16x512x64xf32 --md 1:16x64x32xf32 --md 2:16x512x32xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_matmul_transpose_a --md 0:16x512x64xf32 --md 1:16x512x32xf32 --md 2:16x64x32xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_matmul_transpose_b --md 0:16x512x64xf32 --md 1:16x128x64xf32 --md 2:16x512x128xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_matvec --md 0:16x512x64xf32 --md 1:16x64xf32 --md 2:16x512xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_mmt4d --md 0:4x4x8x4x2xf32 --md 1:4x8x8x4x2xf32 --md 2:4x4x8x4x4xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_reduce_matmul --md 0:16x512x64xf32 --md 1:16x64x32xf32 --md 2:512x32xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case batch_vecmat --md 0:16x64xf32 --md 1:16x64x512xf32 --md 2:16x512xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case dot --md 0:4096xf32 --md 1:4096xf32 --md 2:0xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case matmul --md 0:1024x512xf32 --md 1:512x512xf32 --md 2:1024x512xf32 --cast cast_signed || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case matmul_transpose_a --md 0:1024x512xf32 --md 1:1024x512xf32 --md 2:512x512xf32 --cast cast_signed || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case matmul_transpose_b --md 0:1024x512xf32 --md 1:1024x512xf32 --md 2:1024x1024xf32 --cast cast_signed || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case matvec --md 0:512x64xf32 --md 1:64xf32 --md 2:512xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case mmt4d --md 0:4x8x4x2xf32 --md 1:8x8x4x2xf32 --md 2:4x8x4x4xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case vecmat --md 0:512xf32 --md 1:512x64xf32 --md 2:64xf32 || FAIL=1

# binary
python3 -m benchgc --verbose 0 --driver linalg --case add --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case sub --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case mul --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case div --md 0:1x32x4096xf32 --md 1:1x32x4096xf32 --md 2:1x32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case max --md 0:1024x1024xf32 --md 1:1024x1024xf32 --md 2:1024x1024xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case min --md 0:1024x1024xf32 --md 1:1024x1024xf32 --md 2:1024x1024xf32 || FAIL=1

# element wise
python3 -m benchgc --verbose 0 --driver linalg --case abs --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case ceil --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case erf --md 0:1024x512xf32 --md 1:1024x512xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case floor --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case log --md 0:4096x32xf32 --md 1:4096x32xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case negf --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case exp --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case round --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
# python3 -m benchgc --verbose 0 --driver linalg --case rsqrt --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case sqrt --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case square --md 0:32x4096xf32 --md 1:32x4096xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case tanh --md 0:128x128xf32 --md 1:128x128xf32 || FAIL=1

# conv
python3 -m benchgc --verbose 0 --driver linalg --case conv_1d_ncw_fcw --md 0:4x4x32xf32 --md 1:8x4x4xf32 --md 2:4x8x13xf32 --strides 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_1d_nwc_wcf --md 0:4x32x4xf32 --md 1:4x4x8xf32 --md 2:4x13x8xf32 --strides 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_1d --md 0:32xf32 --md 1:4xf32 --md 2:29xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_2d_nchw_fchw --md 0:4x4x32x32xf32 --md 1:8x4x4x4xf32 --md 2:4x8x13x13xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_2d_ngchw_fgchw --md 0:4x2x2x32x32xf32 --md 1:4x2x2x4x4xf32 --md 2:4x2x4x13x13xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_2d_ngchw_gfchw --md 0:4x2x2x32x32xf32 --md 1:2x4x2x4x4xf32 --md 2:4x2x4x13x13xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_2d_nhwc_fhwc --md 0:4x32x32x4xf32 --md 1:8x4x4x4xf32 --md 2:4x13x13x8xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_2d_nhwc_hwcf --md 0:4x32x32x4xf32 --md 1:4x4x4x8xf32 --md 2:4x13x13x8xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_2d --md 0:32x32xf32 --md 1:4x4xf32 --md 2:29x29xf32 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_3d_ncdhw_fcdhw --md 0:4x4x32x32x32xf32 --md 1:8x4x4x4x4xf32 --md 2:4x8x13x13x13xf32 --strides 2 --strides 2 --strides 2 --dilations 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_3d_ndhwc_dhwcf --md 0:4x32x32x32x4xf32 --md 1:4x4x4x4x8xf32 --md 2:4x13x13x13x8xf32 --strides 2 --strides 2 --strides 2 --dilations 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case conv_3d --md 0:32x32x32xf32 --md 1:4x4x4xf32 --md 2:29x29x29xf32 || FAIL=1

# depthwise conv
python3 -m benchgc --verbose 0 --driver linalg --case depthwise_conv_1d_ncw_cw --md 0:4x4x32xf32 --md 1:4x4xf32 --md 2:4x4x13xf32 --strides 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case depthwise_conv_1d_nwc_wc --md 0:4x32x4xf32 --md 1:4x4xf32 --md 2:4x13x4xf32 --strides 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case depthwise_conv_1d_nwc_wcm --md 0:4x32x4xf32 --md 1:4x4x3xf32 --md 2:4x13x4x3xf32 --strides 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case depthwise_conv_2d_nchw_chw --md 0:4x4x32x32xf32 --md 1:4x4x4xf32 --md 2:4x4x13x13xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case depthwise_conv_2d_nhwc_hwc --md 0:4x32x32x4xf32 --md 1:4x4x4xf32 --md 2:4x13x13x4xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case depthwise_conv_2d_nhwc_hwcm --md 0:4x32x32x4xf32 --md 1:4x4x4x3xf32 --md 2:4x13x13x4x3xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case depthwise_conv_3d_ncdhw_cdhw --md 0:4x4x32x32x32xf32 --md 1:4x4x4x4xf32 --md 2:4x4x13x13x13xf32 --strides 2 --strides 2 --strides 2 --dilations 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case depthwise_conv_3d_ndhwc_dhwc --md 0:4x32x32x32x4xf32 --md 1:4x4x4x4xf32 --md 2:4x13x13x13x4xf32 --strides 2 --strides 2 --strides 2 --dilations 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case depthwise_conv_3d_ndhwc_dhwcm --md 0:4x32x32x32x4xf32 --md 1:4x4x4x4x3xf32 --md 2:4x13x13x13x4x3xf32 --strides 2 --strides 2 --strides 2 --dilations 2 --dilations 2 --dilations 2 || FAIL=1

# pool
python3 -m benchgc --verbose 0 --driver linalg --case pooling_nchw_max --md 0:4x4x32x32xf32 --md 1:4x4xf32 --md 2:4x4x13x13xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_nchw_sum --md 0:4x4x32x32xf32 --md 1:4x4xf32 --md 2:4x4x13x13xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_ncw_max --md 0:4x4x32xf32 --md 1:4xf32 --md 2:4x4x13xf32 --strides 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_ncw_sum --md 0:4x4x32xf32 --md 1:4xf32 --md 2:4x4x13xf32 --strides 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_ndhwc_max --md 0:4x32x32x32x4xf32 --md 1:4x4x4xf32 --md 2:4x13x13x13x4xf32 --strides 2 --strides 2 --strides 2 --dilations 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_ndhwc_sum --md 0:4x32x32x32x4xf32 --md 1:4x4x4xf32 --md 2:4x13x13x13x4xf32 --strides 2 --strides 2 --strides 2 --dilations 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_nhwc_max --md 0:4x32x32x4xf32 --md 1:4x4xf32 --md 2:4x13x13x4xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_nhwc_sum --md 0:4x32x32x4xf32 --md 1:4x4xf32 --md 2:4x13x13x4xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_nhwc_min --md 0:4x32x32x4xf32 --md 1:4x4xf32 --md 2:4x13x13x4xf32 --strides 2 --strides 2 --dilations 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_nwc_max --md 0:4x32x4xf32 --md 1:4xf32 --md 2:4x13x4xf32 --strides 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_nwc_sum --md 0:4x32x4xf32 --md 1:4xf32 --md 2:4x13x4xf32 --strides 2 --dilations 2 || FAIL=1
python3 -m benchgc --verbose 0 --driver linalg --case pooling_nwc_min --md 0:4x32x4xf32 --md 1:4xf32 --md 2:4x13x4xf32 --strides 2 --dilations 2 || FAIL=1

# generic
python3 -m benchgc --verbose 0 --driver mlir --case ${CASE_DIR}/generic.mlir || FAIL=1

# softmax
# python3 -m benchgc --verbose 0 --driver linalg --case softmax --md 0:32x4096xf32 --md 1:32x4096xf32 --dimension 1 || FAIL=1

# mlir
# python3 -m benchgc --verbose 0 --driver mlir --case ${CASE_DIR}/llama2.mlir || FAIL=1

#mlp
python3 -m benchgc --verbose 1  --driver pattern --case mlp --batch_size=32 --hidden_size_list=32x16x64 --has_bias=1x1 --act_type=noop --dtype=f32 

set +e
exit $FAIL