// RUN: gc-opt --split-input-file -pass-pipeline="builtin.module(csa,cst)" %s | FileCheck %s

// CHECK-LABEL: func.func @entry
module {
    func.func @entry(%a: tensor<128xf32>, %b: tensor<128xf32>, %c: tensor<128xf32>) -> (tensor<128xf32>) attributes { llvm.emit_c_interface, onednn_graph.const_args = [0 : i32, 1 : i32] } {
        %c0 = arith.constant 0 : index
        cpuruntime.printf "HI%zu\n" %c0 : index
        %ax2 = tensor.empty() : tensor<128xf32>
        %2 = linalg.add ins(%a, %a : tensor<128xf32>,tensor<128xf32>) outs(%ax2 : tensor<128xf32>) -> tensor<128xf32>
        %bx2 = tensor.empty() : tensor<128xf32>
        %3 = linalg.add ins(%b, %b : tensor<128xf32>,tensor<128xf32>) outs(%bx2 : tensor<128xf32>) -> tensor<128xf32>
        %ax2pbx2 = tensor.empty() : tensor<128xf32>
        %4 = linalg.add ins(%2, %3 : tensor<128xf32>,tensor<128xf32>) outs(%ax2pbx2 : tensor<128xf32>) -> tensor<128xf32>
        %ax2pbx2pc = tensor.empty() : tensor<128xf32>
        %d = linalg.add ins(%4, %c : tensor<128xf32>,tensor<128xf32>) outs(%ax2pbx2pc : tensor<128xf32>) -> tensor<128xf32>
        return %d : tensor<128xf32>
    }
}

// CHECK: cpuruntime.printf
// CHECK: linalg.add
// CHECK: linalg.add
// CHECK: func.func @fold
// CHECK: linalg.add
// CHECK: linalg.add

// COM: expected output:
// COM: module {
// COM:     llvm.mlir.global constant @__num_orig_num_args(4 : i32) : i32
// COM:     llvm.mlir.global constant @__fold_buffer_ids(dense<[2, 114514, 1919810]> : tensor<3 x i64>) : !llvm.array<3 x i64>
// COM:     // a,b, foldedA,foldedB
// COM:     llvm.mlir.global constant @__fold_args(dense<[4, 0, 1, 4, 5]> : tensor<5xi32>) : !llvm.array<5 x i32>
// COM:     // foldedA, foldedB, c, d
// COM:     llvm.mlir.global constant @__compute_args(dense<[4, 4, 5, 2, 3]> : tensor<5xi32>) : !llvm.array<5 x i32>
// COM:     func.func @fold(%a: tensor<128xf32>, %b: tensor<128xf32>) -> (tensor<128xf32>, tensor<128xf32>) attributes { llvm.emit_c_interface } {
// COM:         %c0 = arith.constant 0 : index
// COM:         cpuruntime.printf "HI%zu\n" %c0 : index
// COM:         %out = tensor.empty() : tensor<128xf32>
// COM:         %2 = linalg.add ins(%a, %a : tensor<128xf32>,tensor<128xf32>) outs(%out : tensor<128xf32>) -> tensor<128xf32>
// COM:         %out2 = tensor.empty() : tensor<128xf32>
// COM:         %3 = linalg.add ins(%b, %b : tensor<128xf32>,tensor<128xf32>) outs(%out2 : tensor<128xf32>) -> tensor<128xf32>
// COM:         return %2, %3 : tensor<128xf32>, tensor<128xf32>
// COM:     }
// COM:     func.func @compute(%ax2: tensor<128xf32>, %bx2: tensor<128xf32>, %c: tensor<128xf32>) -> tensor<128xf32> attributes { llvm.emit_c_interface } {
// COM:         %out = tensor.empty() : tensor<128xf32>
// COM:         %2 = linalg.add ins(%ax2, %bx2 : tensor<128xf32>,tensor<128xf32>) outs(%out : tensor<128xf32>) -> tensor<128xf32>
// COM:         %d = linalg.add ins(%2, %c : tensor<128xf32>,tensor<128xf32>) outs(%2 : tensor<128xf32>) -> tensor<128xf32>
// COM:         return %d : tensor<128xf32>
// COM:     }
// COM: }