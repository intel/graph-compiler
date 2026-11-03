// RUN: gc-opt %s --set-attention-layouts --split-input-file | FileCheck %s
// RUN: gc-opt %s --set-attention-layouts --set-layouts --split-input-file | FileCheck %s
// RUN: gc-opt %s --hoist-attention-v-load --set-attention-prefetch --set-attention-layouts --split-input-file | FileCheck %s --check-prefix=INDEPENDENT

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

// -----

// Two independent DPAS operations sharing an input in one K loop are not
// attention. Do not assign attention layouts or introduce prefetches. The
// IV-dependent A descriptor makes the load a prefetch candidate, and B1 starts
// after the first DPAS so an incorrect V-load hoist would change its position.
// INDEPENDENT-LABEL: gpu.func @horizontal
// INDEPENDENT-DAG:     %[[ZERO:.+]] = arith.constant 0 : index
// INDEPENDENT-DAG:     %[[STEP:.+]] = arith.constant 64 : index
// INDEPENDENT-DAG:     %[[KEND:.+]] = arith.constant 256 : index
// INDEPENDENT-DAG:     %[[INIT:.+]] = arith.constant dense<{{.*}}> : vector<32x64xf32>
// INDEPENDENT-DAG:     %[[BD0:.+]] = xegpu.create_nd_tdesc %arg1
// INDEPENDENT-DAG:     %[[BD1:.+]] = xegpu.create_nd_tdesc %arg2
// INDEPENDENT-NOT:     xegpu.prefetch_nd
// INDEPENDENT-NEXT:    %[[LOOP:.+]]:2 = scf.for %[[K:[^ ]+]] = %[[ZERO]] to %[[KEND]] step %[[STEP]]
// INDEPENDENT-SAME:      iter_args(%[[ACC0:[^ ]+]] = %[[INIT]], %[[ACC1:[^ ]+]] = %[[INIT]])
// INDEPENDENT-NOT:       xegpu.prefetch_nd
// INDEPENDENT-NEXT:      %[[AT:.+]] = memref.subview %arg0[0, %[[K]]]
// INDEPENDENT-NEXT:      %[[AD:.+]] = xegpu.create_nd_tdesc %[[AT]]
// INDEPENDENT-NEXT:      %[[A:.+]] = xegpu.load_nd %[[AD]][0, 0] :
// INDEPENDENT-NEXT:      %[[B0:.+]] = xegpu.load_nd %[[BD0]][%[[K]], 0] :
// INDEPENDENT-NEXT:      %[[R0:.+]] = xegpu.dpas %[[A]], %[[B0]], %[[ACC0]] :
// INDEPENDENT-NEXT:      %[[B1:.+]] = xegpu.load_nd %[[BD1]][%[[K]], 0] :
// INDEPENDENT-NEXT:      %[[R1:.+]] = xegpu.dpas %[[A]], %[[B1]], %[[ACC1]] :
// INDEPENDENT-NEXT:      scf.yield %[[R0]], %[[R1]] : vector<32x64xf32>, vector<32x64xf32>
// INDEPENDENT-NEXT:    }
// INDEPENDENT-NOT:     xegpu.prefetch_nd
// INDEPENDENT-NEXT:    gpu.return
module attributes {gpu.container_module} {
  gpu.module @horizontal {
    gpu.func @horizontal(%a: memref<32x256xf16>, %b0: memref<256x64xf16>,
                         %b1: memref<256x64xf16>) kernel
        attributes {known_block_size = array<i32: 256, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %zero = arith.constant dense<0.0> : vector<32x64xf32>
      %b0d = xegpu.create_nd_tdesc %b0 : memref<256x64xf16> -> !xegpu.tensor_desc<64x64xf16>
      %b1d = xegpu.create_nd_tdesc %b1 : memref<256x64xf16> -> !xegpu.tensor_desc<64x64xf16>
      %r:2 = scf.for %k = %c0 to %c256 step %c64
          iter_args(%acc0 = %zero, %acc1 = %zero) -> (vector<32x64xf32>, vector<32x64xf32>) {
        %a_tile = memref.subview %a[0, %k] [32, 64] [1, 1]
            : memref<32x256xf16> to memref<32x64xf16, strided<[256, 1], offset: ?>>
        %ad = xegpu.create_nd_tdesc %a_tile : memref<32x64xf16, strided<[256, 1], offset: ?>> -> !xegpu.tensor_desc<32x64xf16>
        %av = xegpu.load_nd %ad[0, 0] : !xegpu.tensor_desc<32x64xf16> -> vector<32x64xf16>
        %b0v = xegpu.load_nd %b0d[%k, 0] : !xegpu.tensor_desc<64x64xf16> -> vector<64x64xf16>
        %r0 = xegpu.dpas %av, %b0v, %acc0 : vector<32x64xf16>, vector<64x64xf16>, vector<32x64xf32> -> vector<32x64xf32>
        %b1v = xegpu.load_nd %b1d[%k, 0] : !xegpu.tensor_desc<64x64xf16> -> vector<64x64xf16>
        %r1 = xegpu.dpas %av, %b1v, %acc1 : vector<32x64xf16>, vector<64x64xf16>, vector<32x64xf32> -> vector<32x64xf32>
        scf.yield %r0, %r1 : vector<32x64xf32>, vector<32x64xf32>
      }
      gpu.return
    }
  }
}
