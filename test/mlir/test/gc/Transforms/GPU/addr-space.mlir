// RUN: gc-opt %s -gc-attach-addr-spaces | FileCheck %s

gpu.module @add_kernel {
// CHECK: gpu.func @add_kernel(%arg0: index, %arg1: index, %arg2: memref<32xf32, #gpu.address_space<global>>, %arg3: memref<32xf32, #gpu.address_space<global>>, %arg4: memref<32xf32, #gpu.address_space<global>>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 32, 1, 1>} {
gpu.func @add_kernel(%arg0: index, %arg1: index, %arg2: memref<32xf32>, %arg3: memref<32xf32>, %arg4: memref<32xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 32, 1, 1>} {
    gpu.return
}
}
