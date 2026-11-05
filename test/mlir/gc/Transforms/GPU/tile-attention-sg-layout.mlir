// RUN: gc-opt %s --gpu-dev-props --tile-attention --split-input-file | FileCheck %s

// Query rows per subgroup shrink with head_dim to keep Q/S/O in registers.

#mapQ = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#mapK = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
#mapV = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
#mapS = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#mapO = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

// CHECK: attention_kernel = {sg_layout = [8, 1], sg_size = 16 : i32, threads = [128, 1, 1]}
func.func @attention(%q: tensor<?x16x?x64xf16>, %k: tensor<?x16x?x64xf16>, %v: tensor<?x16x?x64xf16>, %init: tensor<?x16x?x64xf16>) -> tensor<?x16x?x64xf16> {
  %scale = arith.constant 1.250000e-01 : f16
  %0 = linalgx.attention <{indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO]}>
       ins(%q, %k, %v, %scale : tensor<?x16x?x64xf16>, tensor<?x16x?x64xf16>, tensor<?x16x?x64xf16>, f16)
       outs(%init : tensor<?x16x?x64xf16>) -> tensor<?x16x?x64xf16>
  return %0 : tensor<?x16x?x64xf16>
}

// -----

#mapQ = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#mapK = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
#mapV = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
#mapS = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#mapO = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

// CHECK: attention_kernel = {sg_layout = [16, 1], sg_size = 16 : i32, threads = [256, 1, 1]}
func.func @attention(%q: tensor<?x16x?x256xf16>, %k: tensor<?x16x?x256xf16>, %v: tensor<?x16x?x256xf16>, %init: tensor<?x16x?x256xf16>) -> tensor<?x16x?x256xf16> {
  %scale = arith.constant 6.250000e-02 : f16
  %0 = linalgx.attention <{indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO]}>
       ins(%q, %k, %v, %scale : tensor<?x16x?x256xf16>, tensor<?x16x?x256xf16>, tensor<?x16x?x256xf16>, f16)
       outs(%init : tensor<?x16x?x256xf16>) -> tensor<?x16x?x256xf16>
  return %0 : tensor<?x16x?x256xf16>
}

// -----

#mapQ = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#mapK = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>
#mapV = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
#mapS = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
#mapO = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

// CHECK: attention_kernel = {sg_layout = [32, 1], sg_size = 16 : i32, threads = [512, 1, 1]}
func.func @attention(%q: tensor<?x1x?x512xf16>, %k: tensor<?x1x?x512xf16>, %v: tensor<?x1x?x512xf16>, %init: tensor<?x1x?x512xf16>) -> tensor<?x1x?x512xf16> {
  %scale = arith.constant 4.418950e-02 : f16
  %0 = linalgx.attention <{indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO]}>
       ins(%q, %k, %v, %scale : tensor<?x1x?x512xf16>, tensor<?x1x?x512xf16>, tensor<?x1x?x512xf16>, f16)
       outs(%init : tensor<?x1x?x512xf16>) -> tensor<?x1x?x512xf16>
  return %0 : tensor<?x1x?x512xf16>
}
