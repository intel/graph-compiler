// RUN: gc-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @sigmoid
func.func @sigmoid(%arg0: tensor<4x256x64xbf16>, %arg1: tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16> {
  // CHECK: linalgx.sigmoid
  %0 = linalgx.sigmoid ins(%arg0 : tensor<4x256x64xbf16>) outs(%arg1 : tensor<4x256x64xbf16>) -> tensor<4x256x64xbf16>
  return %0 : tensor<4x256x64xbf16>
}

// CHECK-LABEL: @mm2d_vnni
func.func @mm2d_vnni(%arg0: tensor<256x64xi8>, %arg1: tensor<16x2x8x32x4xi8>, 
                      %arg2: tensor<256x512xi32>) -> tensor<256x512xi32> {
  // CHECK: linalgx.mm2d_vnni
  %0 = linalgx.mm2d_vnni ins(%arg0, %arg1 : tensor<256x64xi8>, tensor<16x2x8x32x4xi8>) 
                          outs(%arg2 : tensor<256x512xi32>) -> tensor<256x512xi32>
  return %0 : tensor<256x512xi32>
}

// CHECK-LABEL: @mm4d_vnni
func.func @mm4d_vnni(%arg0: tensor<2x8x32x32xbf16>, %arg1: tensor<4x8x16x32x2xbf16>, 
                      %arg2: tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  // CHECK: linalgx.mm4d_vnni
  %0 = linalgx.mm4d_vnni ins(%arg0, %arg1 : tensor<2x8x32x32xbf16>, tensor<4x8x16x32x2xbf16>) 
                          outs(%arg2 : tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  return %0 : tensor<2x4x32x32xbf16>
}

// CHECK-LABEL: @packed_matmul
#m_packing_vnni = [#linalgx.packing_map<[0] -> [0]>, #linalgx.packing_map<[2] -> [2]>]
#n_packing_vnni = [#linalgx.packing_map<[0] -> [1]>, #linalgx.packing_map<[3] -> [3]>]
#k_packing_vnni = [#linalgx.packing_map<[1] -> [1]>, #linalgx.packing_map<[3] -> [2, 4]>]
func.func @packed_matmul(%A: tensor<2x8x32x32xbf16>, %B: tensor<4x8x16x32x2xbf16>, 
                      %C: tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  // CHECK: linalgx.packed_matmul
  %0 = linalgx.packed_matmul 
        {m_packing = #m_packing_vnni, n_packing = #n_packing_vnni, k_packing = #k_packing_vnni}
        ins(%A, %B : tensor<2x8x32x32xbf16>, tensor<4x8x16x32x2xbf16>)
        outs(%C : tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  return %0 : tensor<2x4x32x32xbf16>
}

// CHECK-LABEL: @batch_reduce_matmul_vnni
func.func @batch_reduce_matmul_vnni(%arg0: tensor<512x32x64xbf16>, %arg1: tensor<512x32x128x2xbf16>, 
                      %arg2: tensor<32x128xf32>) -> tensor<32x128xf32> {
  // CHECK: linalgx.batch_reduce_matmul_vnni
  %0 = linalgx.batch_reduce_matmul_vnni ins(%arg0, %arg1 : tensor<512x32x64xbf16>, tensor<512x32x128x2xbf16>) 
                          outs(%arg2 : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}

// CHECK-LABEL: @multi_batch_matmul
func.func @multi_batch_matmul(%arg0: tensor<13x5x6x128x512xbf16>, %arg1: tensor<13x5x6x512x256xbf16>, 
                              %arg2: tensor<13x5x6x128x256xbf16>) -> tensor<13x5x6x128x256xbf16> {
  // CHECK: linalgx.multi_batch_matmul
  %0 = linalgx.multi_batch_matmul ins(%arg0, %arg1 : tensor<13x5x6x128x512xbf16>, tensor<13x5x6x512x256xbf16>) 
                                  outs(%arg2 : tensor<13x5x6x128x256xbf16>) -> tensor<13x5x6x128x256xbf16>
  return %0 : tensor<13x5x6x128x256xbf16>
}
