# Constant tensors folding pass

## 1 Motivation
Some tensors of a machine learning model are constant during inference, such as the weights of filters of convolution 
layers. There are two types of representation in terms of these constant tensors:

- Type 1: they appear as literal values within the model, such as `arith.constant` operations in MLIR. 
OpenVINO can use this way. The IR of a OpenVINO model consists of its topology and constant values, like weights, 
in memory. These values are available in compile time. When transforming OpenVINO IR to MLIR, the constants can be 
lowered into `arith.constant` operations.

- Type 2: they appear as the arguments of the MLIR module entry function and are marked as constant explicitly. 
OneDNN Graph needs to adopt this way. According to the specification of oneDNN Graph, AOT compiling is not supported 
and the kernel compilation happens with logical tensors instead of real tensors. The literal values of these constant 
tensors are available at execution time. OpenVINO can also use this way, by excluding the constant `Input` operations 
when integrating.

Within the IR, there are operations that take the constant tensors as parameters and process them, such as 
reordering or packing. Outputs of such operations are also constants. However, these operations will run every time 
the kernel being executed, which causes redundant memory and computation consumptions. 
This pass modifies the IR so that these operations will only run once. For Type 1 constants, these operations will only 
run once in compile time. For Type 2 constants, these operations will only run once in the first execution time.

## 2 Background
These is no similar pass in the MLIR community currently. 

### 2.1 Constant folding
A related pass is constant folding, which processes **explicit** constants at compile time. But in machine learning, 
the tensors are usually high-dimensional and the operations that process the constant tensors are complex and require 
compiled kernels. So traditional constant folding cannot handle them well. 

Our pass can be thought of enhanced constant folding. It makes the constant tensors be processed only once and the 
processed tensors are cached to buffers in compile time to reuse in runtime.

### 2.2 Constant tensors caching in OpenVINO
There are already some similar transformations in OpenVINO. For each `Graph`, there is a `GraphContext` member. 
A `GraphContext` holds a `WeightsSharing`, which is basically a `std::unordered_map<std::string, MemoryInfo::Ptr>` 
that stores the memory of cached tensors. In compile stage, the operations (for example, type casting ops) that 
follow the constant `Input` operations (weights, bias or others) will be executed and the results are cached in the 
`unordered_map` of the `GraphContext`.

For each `FullyConnected` (`FC` for short) operation with DNNL primitive implementation, there is a `DnnlFCExecutor`, 
which has an attribute of type `ExecutorContext`. The `ExecutorContext` holds an `unordered_map<string, MemoryPtr>` 
to store the memory of its private cached weights. When the `FC` has dynamic shape inputs, which is the case for 
llama2, these is nothing to do with the weights in compile stage. Actually, there is no explicit `Reorder` operation 
in the graph after the constant `Input` operation which holds the weight of a `FC`. In the first execution, all the 
input shapes are defined so the `DnnlFCExecutor` is constructed. During the construction, the weight is packed to 
the blocking format that the DNNL primitive requires, and the memory is stored in the `unordered_map` of 
the `ExecutorContext`. In later executions, the packed weight can be used directly. When the `FC` has static shape, 
the `DnnlFCExecutor` is constructed in compile stage, so the above packing and caching process can be done in compile 
time. All the executions directly use the cached weight.

We can not utilize the work in OpenVINO because 
- it happens far later than the transformation that replaces subgraphs with MLIR-world operations;
- it is deeply coupled with DNNL related data structures.

## 3 Algorithm
There are two steps to complete the pass: an analysis step and a transform step. 
These two steps will be implemented as `Pass`es in the MLIR world.

### 3.1 Analysis step
The analysis step is mainly to identify the operations that take the constant tensors as inputs and output constant 
tensors. They will be marked as interested ops. 

The main work of analysis step will be implemented as a 
[DataFlow Analysis](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/) pass. In the input MLIR graph,
the constant tensors will appear as outputs of `arith.constant` operations (Type 1) or arguments of the MLIR module 
entry function marked with constant attributes (Type 2). The constantness starts its propagation 
from these tensors to the output tensors of operations that process them. Eventually, these operations
will form a subgraph, which is named as 'constant subgraph'. Another subgraph, which contains non-constant operations, 
consumes the outputs of constant subgraph and the parameters to the graph.

Because the constantness information is carried by tensors, the analysis step is on linalg-on-tensor level. 
The interested ops are most likely `reorder`, `pack` or `broadcast` ops, so the analysis step should be after the 
layout propagation pass.

### 3.2 Transform step
Take the following IR as an example (To make the IR short and easy to understand, only the important information is 
shown):
```mlir
module {
    // %weight0 is Type 2 constant, %weight1 is Type 1 constant.
    main(%feature0: tensor<*xbf16>, %weight0: tensor<*xbf16>) 
            -> %feature2: tensor<*xbf16> attributes {const_args_index = [1 : i32]} {
        %weight1 = arith.constant dense<"0x01234567..."> : tensor<*xbf16>
        %packedWeight0 = tensor.pack(%weight0, ...)
        %feature1 = linalg.matmul(%feature0, %packedWeight0)
        %packedWeight1 = tensor.pack(%weight1, ...)
        %feature2 = linalg.matmul(%feature1, %packedWeight1)
        return %feature2
    }
}
```

After transformation, there will be three functions in the module, one for compile time folding, one for runtime 
folding and one for computing. The compile time folding function contains the operations that consume and produce 
constants of Type 1. The runtime folding function contains the operations that consume and produce 
constants of Type 2. The computing function will take all folded tensors as inputs. The expected output IR will be like:
```mlir
module {
    compute(%feature0: tensor<*xbf16>, %foldedWeight0: tensor<*xbf16>, %foldedWeight1: tensor<*xbf16>) 
            -> %feature2: tensor<*xbf16> {
        %feature1 = linalg.matmul(%feature0, %foldedWeight0)
        %feature2 = linalg.matmul(%feature1, %foldedWeight1)
        return %feature2
    }
    compile_time_fold() -> %foldedWeight1: tensor<*xbf16> {
        %weight1 = arith.constant dense<"0x01234567..."> : tensor<*xbf16>
        %foldedWeight1 = tensor.pack(%weight1, ...)
        return %foldedWeight1
    }
    runtime_fold(%weight0: tensor<*xbf16>) -> %foldedWeight0: tensor<*xbf16>{
        %foldedWeight0 = tensor.pack(%weight0, ...)
        return %foldedWeight0
    }
}
```
We place this transformation at linalg-on-tensor level, right after the analysis step.

### 3.3 Management of cached tensors
Later after compiled to executable, the compile time folding function will be executed to generate folded tensors, 
which need to be cached into buffers for future use. These buffers will be under management of the runtime context. 
The execution of the compile time folding function will be the final step of the compile stage. 

An example implementation will be a map which stores pairs of a global index and an allocated buffer:
```c++
// shared by MlirOps
std::unordered_map<int64_t, void *> id2Buffers;

std::vector<void *> getCachedBufers(const std::unordered_map<int64_t, void *> &id2Buffers, 
                                    const std::vector<int64_t> &indexes);
```

During the execution of compile time folding function, when a buffer is allocated for the folded tensor,  
an index will be assigned to the buffer. The map stores these pairs. This map is shared by all `MlirOp`s. 
Each `MlirOp` holds the indexes of buffers it uses.

```C++
// Last step of the compile stage
void executeCompileTimeFoldingFunc(ov::MlirOp op) {
    std::vector<std::pair<int64_t, void *>> allocatedBuffers = allocateBuffers(op, COMPILE_TIME, id2Buffers);
    std::vector<void *> cachedBuffers;
    for (auto p : allocatedBuffers) {
        op.cachedBuffersIndexes.push_back(p.first);
        cachedBuffers.push_back(p.second);
    }
    op.compileTimeFold(cachedBuffers);
}

// MlirOp corresponds to a subgraph from OpenVINO model.
void compile(ov::MlirOp op, mlir::ModuleOp module) {
    ...
    constantSubgraphAnalysis(); // analysis step
    constantSubgraphTransform(); // transform step
    ...
    executeCompileTimeFoldingFunc(op); // execute the folding function
}
```

In the first execution, both the runtime folding function and the computing function will be executed. 
Runtime folding function will also allocate buffers and add them to `id2Buffers`. 
In later executions, only the computing function will be executed.
```C++
void execute(ov::MlirOp op) {
    if (op.executionCount == 0) {
        std::vector<std::pair<int64_t, void *>> allocatedBuffers = allocateBuffers(op, RUNTIME, id2Buffers);
        std::vector<void *> cachedBuffers;
        for (auto p : allocatedBuffers) {
            op.cachedBuffersIndexes.push_back(p.first);
            cachedBuffers.push_back(p.second);
        }
        op.runtimeFold(cachedBuffers);
    }

    std::vector<int64_t> indexes = op.cachedBuffersIndexes;
    std::vector<void *> cachedBuffers = getCachedBufers(id2Buffers, indexes);
    op.compute(feature0, cachedBuffers);

    op.executionCount += 1;
    ......
}
```

### 3.5 Postpone expanding size ops
There is another optimization during the transform. Some operations, such as `Broadcast`, will expand the tensor's 
size dramatically. Folding these operations is not profitable. However, this makes folding their children operations 
not possible. If we can change the order of the expanding-size op and its children ops, the children ops 
can be folded. Take the following IR as example:
```mlir
%15 = tensor.empty() : tensor<8x32xbf16>
%packed_arg2 = tensor.pack %arg2 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %15 : tensor<256xbf16> -> tensor<8x32xbf16>
%bc_arg2_init = tensor.empty() : tensor<2x8x32x32xbf16>
%bc_arg2 = linalg.broadcast ins(%packed_arg2 : tensor<8x32xbf16>) outs(%bc_arg2_init : tensor<2x8x32x32xbf16>) dimensions = [0, 2]
%extf32 = arith.extf %bc_arg2 : tensor<2x8x32x32xbf16> to tensor<2x8x32x32xf32>
%cst_2 = arith.constant 2.000000e+00 : f32
%extf32_mul2_init = tensor.empty() : tensor<2x8x32x32xf32>
%extf32_mul2 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extf32 : tensor<2x8x32x32xf32>) outs(%extf32_mul2_init : tensor<2x8x32x32xf32>) {
^bb0(%in: f32, %out: f32):
    %8 = arith.mulf %in, %cst_2 : f32
    linalg.yield %8 : f32
} -> tensor<2x8x32x32xf32>
%truncbf16 = arith.truncf %extf32_mul2 : tensor<2x8x32x32xf32> to tensor<2x8x32x32xbf16>
```
`%arg2` is processed sequentially by `pack`, `broadcast`, `extf`, `mulf` and `truncf`. The `broadcast` stops 
the folding on `extf`, `mulf` and `truncf`. Then the pass moves the `broadcast` after `truncf` and transforms the IR to:
```mlir
%2 = tensor.empty() : tensor<8x32xbf16>
%pack_1 = tensor.pack %arg2 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %2 : tensor<256xbf16> -> tensor<8x32xbf16>
%3 = arith.extf %pack_1 : tensor<8x32xbf16> to tensor<8x32xf32>
%4 = tensor.empty() : tensor<8x32xf32>
%5 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<8x32xf32>) outs(%4 : tensor<8x32xf32>) {
^bb0(%in: f32, %out: f32):
    %10 = arith.mulf %in, %cst : f32
    linalg.yield %10 : f32
} -> tensor<8x32xf32>
%6 = arith.truncf %5 : tensor<8x32xf32> to tensor<8x32xbf16>
%7 = tensor.empty() : tensor<2x8x32x32xbf16>
%broadcasted = linalg.broadcast ins(%6 : tensor<8x32xbf16>) outs(%7 : tensor<2x8x32x32xbf16>) dimensions = [0, 2]
```
Then the `extf`, `mulf` and `truncf` can be folded, and the `broadcast` is still not folded.
Strict constraints have to be applied to this optimization to ensure the semantic correctness. The children ops 
should be element-wise operations from `linalg`, `arith` or `math` dialects.

## 4 Concerns
1. Two MlirOps with same topology and different Type 1 constant values need to be compiled individually.
