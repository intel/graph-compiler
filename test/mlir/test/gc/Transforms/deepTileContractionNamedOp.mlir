// RUN: gc-opt --split-input-file --deep-tile-contraction-named-op %s | FileCheck %s

// -----

/// CHECK-LABEL: @matmul_2Dx2D_f32
func.func @matmul_2Dx2D_f32(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    // CHECK: scf.forall {{.*}} (0) to (4096) step (1024) {{.*}}  (tensor<4096x4096xf32>) {
    // CHECK: tensor.extract_slice {{.*}} [1024, 4096] [1, 1]
    // CHECK: scf.forall {{.*}} (0) to (4096) step (2048) {{.*}}  (tensor<1024x4096xf32>)
    // CHECK: tensor.extract_slice {{.*}} [1024, 2048] [1, 1]
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [256, 2048] [1, 1]
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [256, 256] [1, 1]
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [32, 256] [1, 1]
    // CHECK: tensor.extract_slice {{.*}} [32, 256] [1, 1]
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [256, 32] [1, 1]
    // CHECK: tensor.extract_slice {{.*}} [32, 32] [1, 1]
    // CHECK: linalg.transpose {{.*}} permutation = [1, 0, 2] 
    // CHECK: tensor.expand_shape {{.*}} output_shape [8, 32, 32] : tensor<256x32xf32> into tensor<8x32x32xf32>
    // CHECK: scf.if
    // CHECK: linalg.fill
    // CHECK: linalg.batch_reduce_matmul
    // CHECK: else
    // CHECK: linalg.batch_reduce_matmul
    // CHECK: tensor.insert_slice {{.*}} [32, 256] [1, 1]
    %2 = linalg.matmul {MThreads = 4 : i32, NThreads = 2 : i32,  KThreads = 1 : i32, MBlock = 256 : i32, NBlock = 256 : i32, KBlock = 256 : i32,innermostMBlock = 32 : i32, innermostNBlock = 32 : i32,  innermostKBlock = 32 : i32 } ins(%arg0, %arg1 : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%1 : tensor<4096x4096xf32>)  -> tensor<4096x4096xf32>
    return %2 : tensor<4096x4096xf32>
}

// -----

/// CHECK-LABEL: @matmul_4Dx4D_bf16
func.func @matmul_4Dx4D_bf16(%arg0: tensor<128x128x32x32xbf16>, %arg1: tensor<128x128x16x32x2xbf16>) -> tensor<128x128x32x32xbf16> {
    %cst_0 = arith.constant 0.000000e+00 : bf16
    // CHECK: tensor.empty() : tensor<128x128x32x32xbf16>
    %0 = tensor.empty() : tensor<128x128x32x32xbf16>
    // CHECK-NOT: linalg.fill
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<128x128x32x32xbf16>) -> tensor<128x128x32x32xbf16>
    // CHECK: scf.forall {{.*}} (0) to (128) step (8) {{.*}} (tensor<128x128x32x32xbf16>)
    // CHECK: tensor.extract_slice {{.*}} [8, 128, 32, 32] [1, 1, 1, 1]
    // CHECK: scf.forall {{.*}} (0) to (128) step (64) {{.*}} (tensor<8x128x32x32xbf16>)
    // CHECK: tensor.extract_slice {{.*}} [8, 64, 32, 32] [1, 1, 1, 1]
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [8, 8, 32, 32] [1, 1, 1, 1]
    // CHECK: tensor.empty() : tensor<8x8x32x32xf32>
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [1, 8, 32, 32] [1, 1, 1, 1]
    // CHECK: tensor.extract_slice {{.*}} [1, 8, 32, 32] [1, 1, 1, 1]
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [1, 8, 32, 32] [1, 1, 1, 1]
    // CHECK: tensor.extract_slice {{.*}} [1, 8, 16, 32, 2] [1, 1, 1, 1, 1]
    // CHECK: tensor.extract_slice {{.*}} [1, 1, 32, 32] [1, 1, 1, 1]
    // CHECK: tensor.extract_slice {{.*}} [1, 1, 32, 32] [1, 1, 1, 1]
    // CHECK: scf.if
    // CHECK: linalg.fill
    // CHECK: linalgx.batch_reduce_matmul_vnni
    // CHECK: else
    // CHECK: linalgx.batch_reduce_matmul_vnni
    // CHECK: scf.if
    // CHECK: linalg.copy
    // CHECK: else
    %2 = linalgx.mm4d_vnni {MThreads = 16 : i32, NThreads = 2 : i32,  KThreads = 1 : i32, MBlock = 256 : i32, NBlock = 256 : i32, KBlock = 256 : i32,innermostMBlock = 32 : i32, innermostNBlock = 32 : i32,  innermostKBlock = 32 : i32 } ins(%arg0, %arg1 : tensor<128x128x32x32xbf16>, tensor<128x128x16x32x2xbf16>) outs(%1 : tensor<128x128x32x32xbf16>)  -> tensor<128x128x32x32xbf16>
    return %2 : tensor<128x128x32x32xbf16>
}

// -----

/// CHECK-LABEL: @matmul_2Dx4D_bf16
func.func @matmul_2Dx4D_bf16(%arg0: tensor<4096x4096xbf16>, %arg1: tensor<128x128x16x32x2xbf16>) -> tensor<4096x4096xbf16> {
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4096x4096xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    // CHECK: scf.forall {{.*}} (2) {{.*}} (tensor<2x1x1x4096x4096xf32>)
    // CHECK: tensor.extract_slice {{.*}} [1, 1, 1, 4096, 4096] [1, 1, 1, 1, 1]
    // CHECK: scf.forall {{.*}} (0) to (4096) step (256) {{.*}} (tensor<4096x4096xf32>)
    // CHECK: tensor.extract_slice {{.*}} [256, 4096] [1, 1]
    // CHECK: scf.forall {{.*}} (0) to (128) step (64) {{.*}} (tensor<256x4096xf32>)
    // CHECK: tensor.extract_slice {{.*}} [256, 2048] [1, 1]
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [256, 256] [1, 1]
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [32, 256] [1, 1]
    // CHECK: tensor.extract_slice {{.*}} [32, 256] [1, 1]
    // CHECK: scf.for
    // CHECK: tensor.extract_slice {{.*}} [1, 8, 16, 32, 2] [1, 1, 1, 1, 1]
    // CHECK: tensor.extract_slice {{.*}} [32, 32] [1, 1]
    // CHECK: linalg.transpose {{.*}} permutation = [1, 0, 2]
    // CHECK: scf.if
    // CHECK: linalg.fill
    // CHECK: linalgx.batch_reduce_matmul_vnni
    // CHECK: else
    // CHECK: linalgx.batch_reduce_matmul_vnni
    // CHECK: scf.forall.in_parallel
    // CHECK: scf.forall.in_parallel
    // CHECK: scf.forall.in_parallel
    // CHECK: linalg.reduce {{.*}} dimensions = [0, 1, 2] 
    // CHECK: linalg.copy
    %2 = linalgx.mm2d_vnni {MThreads = 32 : i32, NThreads = 2 : i32,  KThreads = 2 : i32, MBlock = 256 : i32, NBlock = 256 : i32, KBlock = 256 : i32,innermostMBlock = 32 : i32, innermostNBlock = 32 : i32,  innermostKBlock = 32 : i32 } ins(%arg0, %arg1 : tensor<4096x4096xbf16>, tensor<128x128x16x32x2xbf16>) outs(%1 : tensor<4096x4096xbf16>)  -> tensor<4096x4096xbf16>
    return %2 : tensor<4096x4096xbf16>
}

// -----

module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 49152 : i32>,
      #dlti.dl_entry<"L2_cache_size_in_bytes", 2097152 : i32>,
      #dlti.dl_entry<"L3_cache_size_in_bytes", 110100480 : i32>,
      #dlti.dl_entry<"num_threads", 56 : i32>,
      #dlti.dl_entry<"max_vector_width", 512 : i32>>
  >} {
    /// CHECK-LABEL: @matmul_2Dx4D_bf16_with_dlti
func.func @matmul_2Dx4D_bf16_with_dlti(%arg0: tensor<4096x4096xbf16>, %arg1: tensor<128x128x16x32x2xbf16>) -> tensor<4096x4096xbf16> {
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4096x4096xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    // CHECK: scf.forall
    // CHECK: tensor.extract_slice
    // CHECK: scf.forall
    // CHECK: tensor.extract_slice
    // CHECK: scf.for
    // CHECK: tensor.extract_slice
    // CHECK: scf.for
    // CHECK: tensor.extract_slice
    // CHECK: tensor.extract_slice
    // CHECK: scf.for
    // CHECK: tensor.extract_slice
    // CHECK: tensor.extract_slice
    // CHECK: linalg.transpose
    // CHECK: scf.if
    // CHECK: linalg.fill
    // CHECK: linalgx.batch_reduce_matmul_vnni
    // CHECK: else
    // CHECK: linalgx.batch_reduce_matmul_vnni
    // CHECK: scf.forall.in_parallel
    // CHECK: scf.forall.in_parallel
    %2 = linalgx.mm2d_vnni ins(%arg0, %arg1 : tensor<4096x4096xbf16>, tensor<128x128x16x32x2xbf16>) outs(%1 : tensor<4096x4096xbf16>)  -> tensor<4096x4096xbf16>
    return %2 : tensor<4096x4096xbf16>
}

}
