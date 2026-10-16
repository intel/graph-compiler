// RUN: gc-opt %s --set-attention-layouts | FileCheck %s
// RUN: gc-opt %s --set-attention-layouts --set-layouts | FileCheck %s

// With a static seq_len that is not a multiple of the k-tile, the canonicalizer
// folds away the scf.if guarding the peeled remainder, so the tail iteration is
// inlined after the k-loop. It must get the same layouts as the loop body, and
// set-layouts must not override them with its defaults.

// CHECK-LABEL: gpu.func @sdpa_kernel
// CHECK:         %[[Q:.+]] = xegpu.load_nd {{.*}}sg_layout = [8, 1], sg_data = [16, 64]
// CHECK:         %[[LOOP:.+]] = scf.for
// CHECK:           xegpu.dpas %[[Q]], {{.*}}layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>
// CHECK:           xegpu.dpas {{.*}}layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>
// CHECK:         }
// CHECK:         xegpu.load_nd {{.*}}sg_layout = [1, 1], sg_data = [64, 64], order = [0, 1]
// CHECK:         xegpu.dpas %[[Q]], {{.*}}layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>, layout_b = #xegpu.layout<sg_layout = [1, 1], sg_data = [64, 64]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>
// CHECK:         xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [64, 64]>}>
// CHECK:         xegpu.dpas {{.*}}%[[LOOP]] {{.*}}layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>, layout_b = #xegpu.layout<sg_layout = [1, 1], sg_data = [64, 64]>, layout_cd = #xegpu.layout<sg_layout = [8, 1], sg_data = [16, 64]>
// CHECK:         xegpu.store_nd {{.*}}sg_layout = [8, 1], sg_data = [16, 64]

module attributes {gc.module = {kernels = {sdpa_kernel = {sg_layout = [4, 2]}}}, gpu.container_module} {
  gpu.module @sdpa_kernel {
    gpu.func @sdpa_kernel(%q: memref<128x64xf16>, %k: memref<1178x64xf16>, %v: memref<1178x64xf16>, %out: memref<128x64xf16>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1152 = arith.constant 1152 : index
      %zero = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %q_td = xegpu.create_nd_tdesc %q : memref<128x64xf16> -> !xegpu.tensor_desc<128x64xf16>
      %q_val = xegpu.load_nd %q_td[0, 0] : !xegpu.tensor_desc<128x64xf16> -> vector<128x64xf16>
      %k_td = xegpu.create_nd_tdesc %k : memref<1178x64xf16> -> !xegpu.tensor_desc<64x64xf16>
      %v_td = xegpu.create_nd_tdesc %v : memref<1178x64xf16> -> !xegpu.tensor_desc<64x64xf16>
      %loop = scf.for %iv = %c0 to %c1152 step %c64 iter_args(%acc = %zero) -> (vector<128x64xf32>) {
        %k_val = xegpu.load_nd %k_td[%iv, 0] : !xegpu.tensor_desc<64x64xf16> -> vector<64x64xf16>
        %k_t = vector.transpose %k_val, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %s = xegpu.dpas %q_val, %k_t, %zero : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %p = arith.truncf %s : vector<128x64xf32> to vector<128x64xf16>
        %v_val = xegpu.load_nd %v_td[%iv, 0] : !xegpu.tensor_desc<64x64xf16> -> vector<64x64xf16>
        %o = xegpu.dpas %p, %v_val, %acc : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %o : vector<128x64xf32>
      }
      // Peeled tail (rows 1152..1177), inlined after the loop.
      %k_tail = xegpu.load_nd %k_td[%c1152, 0] : !xegpu.tensor_desc<64x64xf16> -> vector<64x64xf16>
      %k_tail_t = vector.transpose %k_tail, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
      %s_tail = xegpu.dpas %q_val, %k_tail_t, %zero : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %p_tail = arith.truncf %s_tail : vector<128x64xf32> to vector<128x64xf16>
      %v_tail = xegpu.load_nd %v_td[%c1152, 0] : !xegpu.tensor_desc<64x64xf16> -> vector<64x64xf16>
      %o_tail = xegpu.dpas %p_tail, %v_tail, %loop : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %res = arith.truncf %o_tail : vector<128x64xf32> to vector<128x64xf16>
      %out_td = xegpu.create_nd_tdesc %out : memref<128x64xf16> -> !xegpu.tensor_desc<128x64xf16>
      xegpu.store_nd %res, %out_td[0, 0] : vector<128x64xf16>, !xegpu.tensor_desc<128x64xf16>
      gpu.return
    }
  }
}
