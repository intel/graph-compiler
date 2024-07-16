# Graph Compiler CPU Compilation Flow Overview

Graph Compiler is an MLIR based end-to-end DL compiler. The entire compilation process is divided into front-end, middle-end and back-end. Different compilation stages will use different combinations of dialects, and together with various transformation passes to perform various optimizations and graph lowering transformations. The entire process will transform IR from hardware-independent abstract expression to hardware-related concrete expression, and finally generate an executable kernel.

Meanwhile, as an MLIR down-stream project, Graph Compiler's implementation not only uses the existing dialects and passes from MLIR up-stream, but also defines new dialects and passes. Most of the new implementations are upstream-able, and we will do so in the future.

The content introduced in this document does not represent the current implemented status, but the target status after the implementation is completed.

### Front-End

The Graph Compiler front-end takes OneDNN Graph dialect as input. oneDNN Graph dialect is a newly defined dialect, which aims to describe the computation graph defined by oneDNN Graph. The ops in Dialect follow the [oneDNN Graph specification](https://oneapi-src.github.io/oneDNN/graph_supported_operations.html).

oneDNN graph dialect example:

```mlir
func.func @mlp(%in: tensor<128x512xbf16>,
               %weight0: tensor<512x256xbf16>, %bias0: tensor<256xbf16>) -> tensor<128x256xbf16> {
  // layer 0
  %0 = onednn_graph.matmul %in, %weight0, %bias0 : (tensor<128x512xbf16>, tensor<512x256xbf16>, tensor<256xbf16>) -> tensor<128x256xbf16>
  %1 = onednn_graph.relu %0 : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %1 : tensor<128x256xbf16>
}
```

There's no planned optimization passe in front-end. The only transformation pass is to lowering OneDNN Graph dialect into Linalg dialect.

### Middle-End

Middle-end is mainly responsible for general optimizations that are independent of the target hardware, and most of the transformations apply to both CPU and GPU. Some of the transformations need to query the target hardware information, such as cache level and capacity. The Hardware abstract layer(HAL) is the interface for abstracting and describing the target hardware information. Therefore, the same pass can generate different optimization results for different hardware under the guidance of HAL.

According to the different dialect combinations used, middle-end is divided into the following stages:

#### Linalg on Tensor

This is the intermediate representation closest to the framework calculation graph. The example IR looks like:

```mlir
func.func @mlp(%in: tensor<128x512xbf16>,
               %weight0: tensor<512x256xbf16>, %bias0: tensor<256xbf16>) -> tensor<128x256xbf16> {
  %0 = tensor.empty() : tensor<128x256xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %2 = linalg.matmul ins(%in, %weight0 : tensor<128x512xbf16>, tensor<512x256xbf16>) outs(%1 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %3 = tensor.empty() : tensor<128x256xbf16>
  %broadcasted = linalg.broadcast ins(%bias0 : tensor<256xbf16>) outs(%3 : tensor<128x256xbf16>) dimensions = [0]
  %4 = tensor.empty() : tensor<128x256xbf16>
  %5 = linalg.add ins(%2, %broadcasted : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%4: tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %6 = tensor.empty() : tensor<128x256xbf16>
  %7 = linalgx.relu ins(%5 : tensor<128x256xbf16>) outs(%6 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %7 : tensor<128x256xbf16>
}
```

In this stage, GC will perform some analysis and transformation related to the whole graph. The main transformations include:

* Padding propagation : insert tensor.pad op to adjust tensor shape if the shape is not divisible for target tiling size.
* Layout propagation : insert tensor.pack and tensor.unpack to adjust tensor layout if blocking layout is preferred.
* Tensor constant propagation : identify folding with constant tensor and build folding block.
* Matmul lowering : lower Linalg.matmul into scf.forall with linalg.batch_reduce_matmul.
* Fine-grain fusion: fuse element-wise/broadcast/reduce/movement ops into base op(e.g. matmul).
* Lower linalg to arith/math on virtual vector : lower Linalg to Arith/Math and tiling tensor to virtual vector.

### Tensor and scf loop with arith/math on virtual vector

In this stage, most of the Linalg ops are lowered to Scf loops with Arith and Math ops. Both Arith and Math ops use tile tensor as input and output. The tile tensor here can be multi-dimensional tensor in any shape, regardless of the hardware register width. The tile size is chosen based on L1 cache capacity, that is, it is a good abstraction to partition the problem size to this granularity, since the microkernel, pre-op, and post-op, works at the tensor size fitting within l1 cache size. Meanwhile, converting Linalg into Arith and Math can further expose the implementation details of Linalg ops, which allow us to further simplify the computation after fusion.

IR example:

```mlir
func.func @add_tensor(%arg0: tensor<4x8x31xf32>, %arg1: tensor<4x8x31xf32>) -> tensor<4x8x31xf32> {
  %0 = tensor.empty() : tensor<4x8x31xf32>
  %init = arith.constant 0: index
  %c1 = arith.constant 1: index
  %first_dim = arith.constant 4: index
  %second_dim = arith.constant 8: index
  // assume our tile shape is [31]
  %third_dim = arith.constant 31: index
  scf.for %c5 = %init to %first_dim step %c1 {
    scf.for %c6 = %init to %second_dim step %c1 {
        scf.for %c7 = %init to %third_dim step %c1 {
          %1 =  vector.transfer_read %args0[%c5,%c6,%c7] {permutation_map = affine_map<() -> ()>} : tensor<31xf32>, vector<31xf32>
          %2 =  vector.transfer_read %args0[%c5,%c6,%c7] {permutation_map = affine_map<() -> ()>} : tensor<31xf32>, vector<31xf32>
          %3 = arith.add %1, %2 : vector<31xf32>
          vector.transfer_write %3, %0[%c5, %c6, %c7] : vector<31xf32>, tensor<31xf32>
        }
    }
  }
  return %0: tensor<4x8x31xf32>
}
```

The main transformations in this stage include:
* Bfloat16 promotion and cast eliminatation : legalize the Arith and Math ops by inserting `arith.extf` and `arith.truncf` pairs if target device doesn't support, remove pair of redundant `arith.extf` and `arith.truncf` pairs to improve performance and accuracy.
* Lower to physical vector : Lower virtual vector to physical vector based on physical register width of target device.

### Backend-End

Back-end is responsible for device dependent optimization. The use of dialect will vary with the target device. This document will focus on the backend implementation for CPU.

The implementation of BRGEMM is the key to CPU performance.In GC we plan to introduce two different implementations:

* The BRGEMM provided by the library, such as onednn. In order to better abstract and describe the kernel provided by the library, we introduced the microkernel dialect.

* The BRGEMM generated by MLIR. In this approach, The AMX dialect will be used to simplify tile config processing and optimization.

By default GC will use openmp dialect to handle task parallelism. But for better performance and support for non-openmp threadpools, we also introduced the CPURuntime dialect. This dialect also introduces some runtime function calls specifically designed for the CPU, such as thread-local memory allocator, which can improve performance on the CPU.

The main transformations are:
* Memref lowering and scheduling : lower tensor dialect to memref dialect and perform memory related optimization including memory hoist and rescheduling.
* Microkernel dialect and lowering : lower linalg.batch_reduce_matmul to microkernel dialect and further lower to a function call to dnnl brgemm, or an MLIR-based brgemm implementation.
* Parallelcpu dialect and lowering : lower to parallelcpu dialect for Nested parallel loop support and other CPU runtime calls.

In the last step, everything will lower to LLVM dialect. We don't plan to introduce any transformation on LLVM dialect, just leverage the upstream implementation for this.
