# DOC: deep-Tiled Matmul

## Introduction

Tiling and parallelization are important for the performance of a computation intensitive workload (matmul, convolution, and e.t.c). Modern hardware is often equipped with multiple cores and multiple levels of cache, each with different characteristics in terms of size, latency, and bandwidth. To achieve good performance, it is important to utilize the parallelism of the underlying hardware and minimize the number of cache misses to improve the performance of the generated code. The goal of this document is to provide a design overview of the deep-tiled matmul in the graph compiler and its current situation in the community.

## Current Situation in the MLIR Community

According to the last section, tiling and parallelization are two important optimization techniques used in compilers to improve the performance of the generated code(matmul, convolution, and e.t.c). The code template could allow some complex optimization(some nontrivial memory copy/reuse to maximize the hardware efficiency), which is hard to write a unified pass in the compiler.

In the upstream MLIR, there is already some support for tiling and parallelization optimization. The `Linalg` dialect provides a tiling interface to support tiling optimization. Besides, for better representing the concept of schedule, it also introduces the `Transform` dialect to declare the `schedule` in an IR form(vertical to the `payload`).

This section will introduce the current situation in the MLIR community about the tiling interface, `Transform` dialect, hardware abstration layer and what is missing in the current upstream MLIR.

### Tiling Interface And the Related Pass

The MLIR provides the tiling interface as follows to support some simple tiling optimization.

The tiling interface is a set of methods that an operation can implement to provide information about its iteration space and how it can be tiled. The tiling interface is used by the tiling pass to generate a tiled implementation of the operation. It could easily transform the operation like:

```MLIR
%0 = linalg.generic ins(%in) outs(%out) {indexing_maps = [affine_map<(d0) -> (d0)>],
   iterator_types = ["parallel"]}
   : tensor<?xf32> -> tensor<?xf32>
```

into:

```MLIR
%1 = scf.for %iv = %c0 to %dim_0 step %c4 iter_args(%arg3 = %out) -> (tensor<?xf32>) {
  %2 = tensor.extract_slice %in[%iv] [%c4] [1] : tensor<?xf32> to tensor<?xf32>
  %3 = tensor.extract_slice %out[%iv] [%c4] [1] : tensor<?xf32> to tensor<?xf32>
  %4 = linalg.generic ins(%2) outs(%3) ["parallel"] : tensor<?xf32> -> tensor<?xf32>
  %5 = tensor.insert_slice %4, %arg3 : tensor<?xf32>
  scf.yield %5
}
```

The tiling interface further provides several functions like `tileUsingSCF(RewriterBase &rewriter, TilingInterface op, const SCFTilingOptions &options)` to support tile an op inherited the tiling interface, where the SCFTilingOption contains the loop type(scf::For or scf::Forall), interchange vector, mapping vector, and tile size. Through this function, the user could easily generate a tiled implementation of the operation on the parallel axis.

```c++
class TilingInterface : public ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TilingInterfaceTrait<ConcreteOp> {};
  /// Returns a list of iterator types that describe the number of loops.
  SmallVector<utils::IteratorType> getLoopIteratorTypes();
  /// Returns a list of ranges that describe the loop bounds and
  /// step for the loops of the operation.
  SmallVector<Range> getIterationDomain(OpBuilder & b);
  /// Method to generate the tiled implementation of an operation.
  /// 
  /// The iteration space of the operation is returned by
  /// `getIterationDomain`. The caller provides the information of the
  /// tile within this iteration space whose implementation the
  /// caller needs.
  /// - `offsets` provides the offset of the tile in the coordinate system
  ///   of the original iteration space, i.e., if an iteration space
  ///   dimension had non-zero offset, it must be included in the offset
  ///   provided here (as opposed to zero-based offset "relative" to the
  ///   iteration space).
  /// - `sizes` provides the size of the tile.
  /// 
  /// The method returns the operation that is the tiled
  /// implementation.
  FailureOr<TilingResult> getTiledImplementation(OpBuilder & b, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes);
  /// Method to return the position of the result tile computed by the tiled operation.
  /// 
  /// Specifies what tile of the result of the original tensor is computed
  /// by the tiled implementation. Expects the same `offsets` and `sizes` as
  /// used to obtain the tiled implementation of the operation.
  LogicalResult getResultTilePosition(OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, SmallVector<OpFoldResult> & resultOffsets, SmallVector<OpFoldResult> & resultSizes);
  /// Method to generate the code that produces a tile of the result.
  /// 
  /// Generates the IR that computes the tile of a result of the
  /// operation.  The `offsets` and `sizes` describe the tile of
  /// the output required. This is different from
  /// `getTiledImplementation` which generates the tiled
  /// implementation of the operation given a tile of the
  /// iteration space. This method generates a tiled
  /// implementation of the operation based on the tile of the
  /// result required. This method enables fusion by using tile
  /// and fuse. The method returns failure if the operation can't be
  /// tiled to generate the result tile. In practical terms this
  /// implies it cannot be tiled and fused with its consumers.
  /// 
  /// - `offsets` provides the offset of the tile in the coordinate system
  ///   of the original iteration space, i.e., if an iteration space
  ///   dimension had non-zero offset, it must be included in the offset
  ///   provided here (as opposed to zero-based offset "relative" to the
  ///   iteration space).
  /// - `sizes` provides the size of the tile.
  FailureOr<TilingResult> generateResultTileValue(OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
  /// Generates the scalar implementation of the operation.
  /// 
  /// Given the list `ivs` that represent points in the iteration space
  /// (as specified by `getIterationDomain()`) returns the scalar operations
  /// that represent the computation at that point in the iteration space.
  /// This method is typically used as the "exit path", i.e. once all
  /// transformations are done, this method can be used to lower to scalar
  /// code that can then be lowered to LLVM or SPIR-V dialects.
  LogicalResult generateScalarImplementation(OpBuilder & b, Location  loc, ValueRange  ivs);
};

struct SCFTilingOptions {
  /// Computation function that returns the tile sizes for each operation.
  /// Delayed construction of constant tile sizes should occur to interoperate
  /// with folding.
  SCFTileSizeComputationFunction tileSizeComputationFunction = nullptr;

  /// The interchange vector to reorder the tiled loops.
  SmallVector<int64_t> interchangeVector = {};

  /// Specify which loop construct to use for tile and fuse.
  enum class LoopType { ForOp, ForallOp };
  LoopType loopType = LoopType::ForOp;

  /// Specify mapping of loops to devices. This is only respected when the loop
  /// constructs support such a mapping (like `scf.forall`). Will be ignored
  /// when using loop constructs that dont support such a mapping (like
  /// `scf.for`)
  SmallVector<Attribute> mappingVector = {};
};
FailureOr<SCFTilingResult> tileUsingSCF(RewriterBase &rewriter,
                                        TilingInterface op,
                                        const SCFTilingOptions &options);

/// Rewrite a TilingInterface `op` to a tiled `scf.forall`, applying
/// tiling by `numThreads`.
/// If non-empty, the `mapping` is added as an attribute to the
/// resulting `scf.forall`.
/// Zero tile sizes indicate that the dimension is not tiled, and can be
/// thought of as tiling by the full size of data. It is the user's
/// responsibility to ensure that `numThreads` is a valid tiling specification
/// (i.e. that only tiles parallel dimensions, e.g. in the Linalg case).
struct ForallTilingResult {
  Operation *tileOp;
  Operation *tiledOp;
};
FailureOr<ForallTilingResult> tileToForallOp(RewriterBase &builder,
                                             TilingInterface op,
                                             ArrayRef<OpFoldResult> numThreads,
                                             std::optional<ArrayAttr> mapping);

/// Same as `tileToForallOp`, but calculate the number of threads
/// required using the given tileSizes.
FailureOr<ForallTilingResult>
tileToForallOpUsingTileSizes(RewriterBase &builder, TilingInterface op,
                             ArrayRef<OpFoldResult> tileSizes,
                             std::optional<ArrayAttr> mapping);
```

The above tiling interface only supports the tiling on the parallel axis. But in a workload like matmul, it is often required to do a tiling on the reduction axis for better performance considering the size of available memory/cache, computation intensity, cache communication, etc.

```MLIR
%red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0)>],
  iterator_types = ["parallel", "reduction"]}
  ins(%arg0 : tensor<?x?xf32>)
  outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
    %1 = arith.addf %arg7, %arg9 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
```

into:

```MLIR
%0 = tensor.empty(%dim_1) : tensor<?x5xf32>
%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x5xf32>) -> tensor<?x5xf32>
%2 = scf.for %arg2 = %c0 to %dim_0 step %c5 iter_args(%arg3 = %1) -> (tensor<?x5xf32>) {
  %extracted_slice = tensor.extract_slice %1[0, 0] [%dim, 5] [1, 1] : tensor<?x5xf32> to tensor<?x5xf32>
  %extracted_slice_2 = tensor.extract_slice %arg0[0, %arg2] [%dim, 5] [1, 1] : tensor<?x?xf32> to tensor<?x5xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]}
  ins(%extracted_slice_2 : tensor<?x5xf32>)
  outs(%extracted_slice : tensor<?x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.addf %in, %out : f32
    linalg.yield %5 : f32
  } -> tensor<?x5xf32>
  %dim_3 = tensor.dim %1, %c0 : tensor<?x5xf32>
  %inserted_slice = tensor.insert_slice %4 into %arg3[0, 0] [%dim_3, 5] [1, 1] : tensor<?x5xf32> into tensor<?x5xf32>
  scf.yield %inserted_slice : tensor<?x5xf32>
}
%3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                      affine_map<(d0, d1) -> (d0)>],
  iterator_types = ["parallel", "reduction"]}
  ins(%2 : tensor<?x5xf32>)
  outs(%arg1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.addf %in, %out : f32
    linalg.yield %4 : f32
  } -> tensor<?xf32>
```

To support this kind of tiling, the MLIR also provide a `PartialReductionOpInterface` based on TilingInterface. The `PartialReductionOpInterface` is an interface with a set of methods that provide information about its partial reduction and how it can be tiled. Based on the `PartialReductionOpInterface`, it further provides a function `tileReductionUsingScf(RewriterBase &b, PartialReductionOpInterface op, ArrayRef<OpFoldResult> tileSize)` to support tile an op inherited the `PartialReductionOpInterface`, where the `tileSize` is the tile size for the reduction axis.

```c++
class PartialReductionOpInterface : public ::mlir::OpInterface<PartialReductionOpInterface, detail::PartialReductionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PartialReductionOpInterface, detail::PartialReductionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PartialReductionOpInterfaceTrait<ConcreteOp> {};
  /// Method to generate a tensor initalized with the identity value of the
  /// operation reduction. The tensor shape is equal to operation result
  /// shape with new dimension for each non zero tile size.
  FailureOr<Operation*> generateInitialTensorForPartialReduction(OpBuilder & b, Location  loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim);
  /// Method to generate a tiled version of the operation where the tiled
  /// reduction dimension are converted to parallel dimensions with a size
  /// less or equal to the tile size. This is meant to be used with
  /// `mergeReductions` method which will combine the partial reductions.
  Operation*tileToPartialReduction(OpBuilder & b, Location  loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims);
  /// Method to merge partial reductions for an operation that has been
  /// tiled along the reduction dimensions. This will only apply the
  /// reduction the operation.
  Operation*mergeReductions(OpBuilder & b, Location  loc, ValueRange partialReduce, ArrayRef<int> reductionDim);

/// Method to tile a reduction and generate a parallel op within a serial loop.
/// Each of the partial reductions are calculated in parallel. Then after the
/// loop all the partial reduction are merged into a final reduction.
/// For example for the following sequence
///
/// ```mlir
/// %0 = linalg.generic %in ["parallel", "reduction"]
///   : tensor<7x9xf32> -> tensor<7xf32>
/// ```
///
/// into:
///
/// ```mlir
/// %0 = linalg.fill ... : tensor<7x4xf32>
/// %1 = scf.for ... iter_args(%arg0 = %0)
///   %2 = tensor.extract_slice %arg0 : tensor<7x4xf32> -> tensor<7x?xf32>
///   %3 = tensor.extract_slice %in : tensor<7x9xf32> -> tensor<7x?xf32>
///   %4 = linalg.generic %2, %3 ["parallel", "parallel"]
///     : tensor<7x?xf32> -> tensor<7x?xf32>
///   %5 = tensor.insert_slice %3, %0[0, 0] : tensor<7x4xf32>
/// }
/// %6 = linalg.generic %1 ["parallel", "reduction"]
///   : tensor<7x4xf32> -> tensor<7xf32>
/// ```
FailureOr<scf::SCFReductionTilingResult>
tileReductionUsingScf(RewriterBase &b, PartialReductionOpInterface op,
                      ArrayRef<OpFoldResult> tileSize);

/// Method to tile a reduction to parallel iterations computing partial
/// reductions. After the loop all the partial reduction are merged into a final
/// reduction. For example for the following sequence
///
/// ```mlir
/// %0 = linalg.generic %in ["parallel", "reduction"]
///   : tensor<7x9xf32> -> tensor<7xf32>
/// ```
///
/// into:
///
/// ```mlir
/// %0 = linalg.fill ... : tensor<7x4xf32>
/// %1 = scf.forall (%iv) in (%c4) shared_outs(%arg0 = %0)
///   -> (tensor<7x4xf32>) {
///   %2 = tensor.extract_slice %arg3 : tensor<7x4xf32> to tensor<7xf32>
///   %3 = tensor.extract_slice %in : tensor<7x9xf32> -> tensor<7x?xf32>
///   %4 = linalg.generic %2, %3 ["parallel", "reduction"]
///     : tensor<7x?xf32> -> tensor<7xf32>
///   %5 = tensor.insert_slice %3, %arg0[0, %iv] : tensor<7x4xf32>
/// }
/// %6 = linalg.generic %1 ["parallel", "reduction"]
///   : tensor<7x
FailureOr<ForallReductionTilingResult>
tileReductionUsingForall(RewriterBase &b, PartialReductionOpInterface op,
                         ArrayRef<OpFoldResult> numThreads,
                         ArrayRef<OpFoldResult> tileSizes = {},
                         std::optional<ArrayAttr> mapping = std::nullopt);
};
```

### Hardware Abstraction Layer(HAL)

To achieve the best performance, a good schedule requires the hardware information as a reference. Hardware information like cache size, thread number, etc. is often needed to generate the best schedule. Hardware Abstraction Layer(HAL) is a layer of software that provides a hardware-independent interface to the underlying hardware. The mainstream dl compiler or performance library has a way to get the hardware information to guide the schedule like [IREE](https://iree.dev/developers/design-docs/design-roadmap/#hal-hardware-abstraction-layer-and-multi-architecture-executables), [TVM](https://tvm.apache.org/docs/arch/device_target_interactions.html#tvm-target-specific-overview), [onednn](https://github.com/oneapi-src/oneDNN), etc. However, the MLIR doesn't have such a hardware abstraction layer(HAL) to provide the hardware information.

## Deep-Tiled Matmul Introduction

This section will introduce the concept of the deep-tiled matmul optimization(nested matmul/managed_matmul in graph compiler v1) and how it could improve the performance.

Deep-tiled matmul originally is a [matmul code template](https://github.com/oneapi-src/oneDNN/blob/main/src/graph/backend/graph_compiler/core/src/ops/templates/managed_matmul_core.cpp) in the [onednn graph compiler v1](https://arxiv.org/ftp/arxiv/papers/2301/2301.01333.pdf) with well-tuned default parameters to deliver good performance in the e2e model. The basic idea of the deep-tiled matmul is to partition the iteration space of the matmul into 9 loops as the pseudocode shown below. The outermost 3 loops(`Mthreads, NThreads, KThreads`) are used to partition the iteration space of the matmul according to the number of threads, which is used to balance the workload distribution among the threads and minimize the cache synchronization/communication overhead. The middle 3 loops(`MBlock, NBlock, KBlock`) are used to partition the iteration space of the matmul and control the loop order according to the L2 cache size in the CPU, which is used to improve the data locality of the generated code. The innermost 3 loops(`innermostMBlock, innermostNBlock, innermostKBlock`) are used to partition the iteration space of the matmul and control the loop order according to the L1 cache size in CPU, which could further improve the data locality of the generated code. At this level, the matmul will be converted to the micro-kernel call [*brgemm*](https://arxiv.org/pdf/2104.05755.pdf) which is a highly optimized vectorized kernel(appling the optimiztion like unroll, operation interleave, prefetch, nt load/store, particularly tuned memory accessing pattern, carefully handcrafted register allocation). Though the tiling strategy above is based on the CPU model, it could be easily extended to the concept of the other hardware like GPU, FPGA, etc.(`global/shared memory`, `L1/2 cache size`, `execution model(threads, warp, block, grid, etc)`, etc)

```c++
parameter M, N, K, MBlock, NBlock, KBlock, MThreads, NThreads, KThreads, innermostMBlock, innermostNBlock, innermostKBlock
tensor A, B, C
tempC = create_tensor for C -> tensor([KThreads, M, N])
parallel_for([PM, PN, PK]: [MThreads, NThreads, KThreads]) {
  ASlice = extract_slice from A -> tensor([MOuterBlock, KOuterBlock])
  BSlice = extract_slice from B -> tensor([KOuterBlock, NOuterBlock])
  CSlice = extract_slice from C -> tensor([MOuterBlock, NOuterBlock])
  MNumBlock = MOuterBlock / MBlock
  NNumBlock = NOuterBlock / NBlock
  KNumBlock = KOuterBlock / KBlovk
  for([om, on, ok]: [MNumBlock, NNumBlock, KNumBlock]) {
    ASlice2 = extract_slice from ASlice -> tensor([MBlock, KBlock])
    BSlice2 = extract_slice from BSlice -> tensor([KBlock, NBlock])
    CSlice2 = extract_slice from CSlice -> tensor([1, MBlock, NBlock])
    MNumInnerBlock = MBlock / innermostMBlock
    NNumInnerBlock = NBlock / innermostNBlock
    KNumInnerBlock = KBlock / innermostKBlock
    for([im, in]: [MNumInnerBlock, NNumInnerBlock]) {
      ASlice3 = extract_slice from ASlice2 -> tensor([innermostMBlock, KBlock])
      BSlice3 = extract_slice from BSlice2 -> tensor([KBlock, innermostNBlock])
      CSlice3 = extract_slice from CSlice2 -> tensor([innermostMBlock, innermostNBlock])
      if(ok == 0) {
        init CSlice3 with 0 (could use init_brgemm when it is avaliable)
      }
      brgemm(bs=KNumInnerBlock, M=innermostMBlock, N=innermostNBlock, K=innermostKBlock,
A=ASlice3, B=BSlice3, C=CSlice4, onlyUpdate=(ok!=0));
    }
  }
}
C = final_reduce(tempC) -> [M, N]
```

## Proposal

This section will present a proposal based on the [Option 4](#option-4---outer-loop-based-on-tiling-interface--inner-loop-through-a-predefined-template-with-ir-builder) above to implement the deep-tiled matmul in the graph compiler v2. According to the discussion above, option 4 could deliver high performance and maximally reuse the current existing work MLIR, which minimizes the difficulty of acceptance by the community. In the meantime, future optimizations like `loop reorder`, and `axis split` could be easily added by changing the parameter. So this is the recommended way in this document and the detail will be introduced in the following.

### Position

> The transformation control infrastructure provided by this dialect is positioned roughly between rewrite patterns and passes. A transformation that is executed by a transform operation is likely to be sufficiently complex to require at least a set of patterns to be implemented. It is also expected to be more focused than a pass: a pass typically applies identical transformations everywhere in the IR, a transform dialect-controlled transformation would apply to a small subset of operations selected, e.g., by a pattern-matching operation or generated by a previous transformation. It is discouraged, although technically possible, to run a pass pipeline as part of the transform op implementation. *From [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Transform/)*

As MLIR mentions in the documentation, the scope order from largest to smallest is `pass > Transform dialect > rewrite patterns`. The deep-tiled matmul only applies to the operation `matmul` and `batch_matmul`. So it is better to implement it as a rewrite pattern. To better meet the upstream's need, it could be warped into an operation of the `Transform` dialect so that it could become a part of the `Transform` schedule.

In the graph compiler v2, this could be further warped in a pass `deepTilingRewriteForContractionOperation`, which could also contain other deep-tiling rewrite patterns in the future(`paddedConvolution`, `reduceLoweringConvolution`, `depthwiseConvolution`, etc). This pass is expected to be executed after the `padding/layout propagation`-related pass and before the `fusion` '-related pass. `Layout` related pass could convert the input/output tensor to the required blocked layout to achieve better performance. And fusion-related pass may depend on the tiled matmul's `insert_slice/extract_slice` as the anchor to do fusion.

```MLIR
...
layout propogation related pass(pack/unpack, pad, propogation, etc)

deepTilingRewriteForContractionOperation(deep-tiled matmul, deep-tiled padded conv, conv1x1, depthwise conv, etc)

fusion related pass
...
```

Besides, this rewrite pattern is expected to be a part of the linalg dialect. This is similar to the existing rewrite `ConvertConv2DToImg2Col` in MLIR. In graph compiler v2, it could be a part of `linalgX` before upstream.

### Outer Loop Generation

For outer loop generation, we will generate the loop step by step according to the parameters/config(`outermost loop for multicore ->  loop for L2 cache -> loop for L1 cache`). This part would be implemented based on the tiling interface and its related utility function, which could maximally reuse the existing work in the MLIR and decrease the difficulty of the maintenance. Besides, function like `tileToForallOp` provides an `interchange` parameter which makes it easy to change the loop order according to the workload characteristics. This way could be also easily reused by other operations like `convolution`, `depthwise convolution`, etc because they have a similar structure in this part.

The expected implementation in pseudocode code is as follows

```c++
// generate outer loop with MThreads, NThreads
linalg::tileToForallOp(rewriter, cast<TilingInterface>(matmul), {MThreads, NThreads});
// generate outer reduction loop with KThreads
linalg::tileReductionUsingForall(rewriter, cast<PartialReductionOpInterface>(matmul), KThreads, tileSizes);
// generate the middle three loops(MBlock, NBlock, KBlock)
scf::tileUsingSCF(rewriter, cast<TilingInterface>(matmul),tileOption);
// generate the inner loops(innerMostMBlock, innerMostNBlock, innerMostKBlock)
scf::tileUsingSCF(rewriter, cast<TilingInterface>(matmul),tileOption);
```

As mentioned in the [Current Situation in the MLIR Community](#current-situation-in-the-mlir-community), there are still some missing things in the current MLIR like the lack of balance211 for not perfectly divisible cases, inefficient partial K threads position for cpu, etc. These should be further enhanced in future work.

### Inner Loop Body Generation

Compared to outer loop generation, the inner loop body generation is sometimes op-specific. For example, the `squeeze stride` optimization for convolution doesn't make any sense for `matmul`. Besides, this part is possibly more complex than the outer-loop(may have tail processing, non-trivial memory copy/init) and hard to unify a pass to do it. So it is better to implement it as a predefined template through IR builder which could make the code more flexible. We could also add easy builder/util support to make it more readable.

Below is the expected pseudocode of the inner loop body for the deep-tiled matmul in the graph compiler v2.

```c++
A = tensor.extract_slice
B = tensor.extract_slice
C = tensor.extract_slice
D3 = scf.if(ok == 0) {
  D1 = init_brgemm(A,B,C) tensor<...>, tensor<...>, tensor<...> -> tensor<...>
} else {
  D2 = brgemm(A,B,C) tensor<...>, tensor<...>, tensor<...> -> tensor<...>
} -> tensor<...>
tensor.insert_slice D3
```

The inner loop body will convert the `matmul` to the `batch_reduce_gemm`, which will be finally converted to the microkernel [`brgemm`](https://github.com/oneapi-src/oneDNN/pull/1852) call.

### Config/Schedule

```c++
struct MatmulConfig {
  int MThreads, NThreads, KThreads;
  int MBlock, NBlock, KBlock;
  int innerMostMBlock, innerMostNBlock, innerMostKBlock;
  int loopOrder;
};
```

The above is the expected config for the deep-tiled matmul. The `MThreads, NThreads, KThreads` is used to partition the iteration space of the matmul according to the number of threads. The `MBlock, NBlock, KBlock` is used to partition the iteration space of the matmul and control the loop order according to the L2 cache size in the CPU. The `innerMostMBlock, innerMostNBlock, innerMostKBlock` is used to partition the iteration space of the matmul and control the loop order according to the L1 cache size in the CPU. The `loopOrder` is used to control the loop order/iterate order according to the workload characteristics.

A default heuristic config corresponding to these items will be tuned for the performance.

1. For `MThreads, NThreads, KThreads`, we should rely on the available threads, required memory for the input/output/temp buffer, and the L2/L3 cache size to build a cost model that maximizes the workload balance, threads utilization and minimize the cache synchronization. But the threads on the K axis should be set carefully as it may hurt performance in most cases (performance gain on large K but small M, N).
2. For `MBlock, NBlock, KBlock`, the L2 cache size and the required memory for every core are needed to build a cost model so that the L2 cache misses would be minimized.
3. For `innerMostMBlock, innerMostNBlock, innerMostKBlock`, we need to know the L1 cache size, the size of available registers and vector/matrix-vector (amx-like) length to decide the innermost block size so that the hardware efficiency can be maximized. Besides, if we convert the brgemm to an external library function call, the cost of the function call is also needed to be considered. In the case that M/N/K is not divisible by vector length, we usually will choose a factor of the M/N/K as the innermost block size or do the packing/unpacking to make it divisible in advance(a tradeoff between reducing memory copy and maximize hardware efficiency).
4. The `loopOrder` is mainly related to the workload characteristics(data, weight, output size), the cache size and where the actual data/weight is located at L1/L2/L3/memory. This will have an impact on the visit order of the memory and finally impact the cache data locality.

The description above shows what should be considered from the horizontal view(`[M/N/K]threads`, `[M/N/K]block`, `innermost[M/N/K]Block`, `loop order`) in the config. However, in the vertical view(`MThreads, MBlock, innermostMBlock`, `N...`, `K...`), they will have some interdependence that will also impact the performance, and the order to decide them will matter. The breakdown of how to decide is as follows.

1. Firstly, we need to decide the `innerMostBlock[M/N/K]` which will impact the maximum hardware efficiency we can achieve, especially for the machine with a specialized matrix computation unit(amx-like). For example, if the physical matrix vector size is 16x64 and we choose the innermost block size as 8x32, then the theoretical efficiency will be a quarter of the maximum. Even for the vector instruction set like `avx512, avx2, etc`, the `innermostBlock` still matters because they still require the `innermostBlock` to align with the vector length(64/32/...). So the priority of the `innermostBlock` is the highest.
2. After the `innermostBlock` is decided, the input and output matrix will be divided into `[M/N/K]NumBlock` blocks with block size `[M/N/K]innermostBlock`. Then we will decide what `[M/N/K]Threads` should use to distribute these blocks so that the best workload balance, compute intensity and cache utilization can be achieved.
3. After step 2, the number of innermost blocks for every thread has been decided. Then we will decide the `[M/N/K]Block` to further partition the iteration space of the matmul so that the L2 cache misses in a single core would be minimized. This should be the multiples of the `innermost[M/N/K]Block`.
4. After above steps, all tile size is decided and we have enough infomation about where the data is located(L1/L2/L3 and their size). The `loopOrder` could be decided to maximize the data locality/data reuse. What it decides is the order of these loops(`pmpnpkomonokiminik`, `pnpmpkokonominimik`, etc where `p` is the outermost parallel loop, `o` is the middle outer loop, `i` is the innermost loop).

**Note**: In the graph compiler v1, we also consider the impact of the previous matmul as this will decide where the output of the previous matmul is located (3rd core's l2 cache or 4th core's). This could be also further enhanced in the future.

The heuristic default config will be implemented as an [analysis pass](https://mlir.llvm.org/docs/PassManagement/#analysis-management). In this way, the heuristic is maximally isolated from the real IR transformation and easier to be accepted by the upstream community(who want to separate the heuristics from passes as much as possible). By the way, other passes like layout/padding propagation could also know which tile size is preferable by the matmul and will not have a dependence cycle among these passes.

All choices above need to be under the guidance of HAL. But the HAL support(multi-level cache size, machine kind, available threads, register vector length) is not fully ready in the MLIR now. So there is a risk here to tune a good performance for general.

### Expected IR Change

Below is a matmul example(`M=256, K=128, N=512`) of the expected IR change after applying the deep-tiled matmul rewrite pattern(with config `MThreads=2, NThreads=2, KThreads=1, MBlock=128, NBlock=256, KBlock=128, innerMostMBlock=32, innerMostKBlock=32, loopOrder=0`).

```MLIR
%0 = linalg.matmul ins(%cst_0, %cst_1 : tensor<256x128xf32>, tensor<128x512xf32>) outs(%cst_2 : tensor<256x512xf32>) -> tensor<256x512xf32>
```

into:

```MLIR
%0 = scf.forall (%arg0, %arg1) in (2, 2) shared_outs(%arg2 = %cst_3) -> (tensor<256x512xf32>) {
  %1 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
  %2 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg1)
  %3 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
  %4 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg1)
  %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
  %6 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg1)
  %extracted_slice = tensor.extract_slice %cst_1[%3, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
  %extracted_slice_5 = tensor.extract_slice %cst_2[0, %4] [128, 256] [1, 1] : tensor<128x512xf32> to tensor<128x256xf32>
  %extracted_slice_6 = tensor.extract_slice %arg2[%5, %6] [128, 256] [1, 1] : tensor<256x512xf32> to tensor<128x256xf32>
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c128_7 = arith.constant 128 : index
  %7 = scf.for %arg3 = %c0 to %c128 step %c128_7 iter_args(%arg4 = %extracted_slice_6) -> (tensor<128x256xf32>) {
    %c0_8 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c256_9 = arith.constant 256 : index
    %10 = scf.for %arg5 = %c0_8 to %c256 step %c256_9 iter_args(%arg6 = %arg4) -> (tensor<128x256xf32>) {
      %c0_10 = arith.constant 0 : index
      %c128_11 = arith.constant 128 : index
      %c128_12 = arith.constant 128 : index
      %11 = scf.for %arg7 = %c0_10 to %c128_11 step %c128_12 iter_args(%arg8 = %arg6) -> (tensor<128x256xf32>) {
        %extracted_slice_13 = tensor.extract_slice %extracted_slice[%arg3, %arg7] [128, 128] [1, 1] : tensor<128x128xf32> to tensor<128x128xf32>
        %extracted_slice_14 = tensor.extract_slice %extracted_slice_5[%arg7, %arg5] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
        %extracted_slice_15 = tensor.extract_slice %arg8[%arg3, %arg5] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
        %c0_16 = arith.constant 0 : index
        %c128_17 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %12 = scf.for %arg9 = %c0_16 to %c128_17 step %c32 iter_args(%arg10 = %extracted_slice_15) -> (tensor<128x256xf32>) {
          %c0_18 = arith.constant 0 : index
          %c256_19 = arith.constant 256 : index
          %c32_20 = arith.constant 32 : index
          %13 = scf.for %arg11 = %c0_18 to %c256_19 step %c32_20 iter_args(%arg12 = %arg10) -> (tensor<128x256xf32>) {
            %c0_21 = arith.constant 0 : index
            %c128_22 = arith.constant 128 : index
            %c128_23 = arith.constant 128 : index
            %14 = scf.for %arg13 = %c0_21 to %c128_22 step %c128_23 iter_args(%arg14 = %arg12) -> (tensor<128x256xf32>) {
              %extracted_slice_24 = tensor.extract_slice %extracted_slice_13[%arg9, %arg13] [32, 128] [1, 1] : tensor<128x128xf32> to tensor<32x128xf32>
              %extracted_slice_25 = tensor.extract_slice %extracted_slice_14[%arg13, %arg11] [128, 32] [1, 1] : tensor<128x256xf32> to tensor<128x32xf32>
              %extracted_slice_26 = tensor.extract_slice %arg14[%arg9, %arg11] [32, 32] [1, 1] : tensor<128x256xf32> to tensor<32x32xf32>
              %expanded = tensor.expand_shape %extracted_slice_24 [[0, 1], [2]] : tensor<32x128xf32> into tensor<1x32x128xf32>
              %expanded_27 = tensor.expand_shape %extracted_slice_25 [[0, 1], [2]] : tensor<128x32xf32> into tensor<1x128x32xf32>
              %15 = linalg.batch_reduce_matmul ins(%expanded, %expanded_27 : tensor<1x32x128xf32>, tensor<1x128x32xf32>) outs(%extracted_slice_26 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %inserted_slice_28 = tensor.insert_slice %15 into %arg14[%arg9, %arg11] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x256xf32>
              scf.yield %inserted_slice_28 : tensor<128x256xf32>
            }
            scf.yield %14 : tensor<128x256xf32>
          }
          scf.yield %13 : tensor<128x256xf32>
        }
        %inserted_slice = tensor.insert_slice %12 into %arg8[%arg3, %arg5] [128, 256] [1, 1] : tensor<128x256xf32> into tensor<128x256xf32>
        scf.yield %inserted_slice : tensor<128x256xf32>
      }
      scf.yield %11 : tensor<128x256xf32>
    }
    scf.yield %10 : tensor<128x256xf32>
  }
  %8 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
  %9 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg1)
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %7 into %arg2[%8, %9] [128, 256] [1, 1] : tensor<128x256xf32> into tensor<256x512xf32>
  }
}
```

When the `KThreads=2`, there will be partial reduction in the loop

```MLIR
%0 = scf.forall (%arg0, %arg1) in (2, 2) shared_outs(%arg2 = %cst_3) -> (tensor<256x512xf32>) {
  %1 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
  %2 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg1)
  %3 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
  %4 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg1)
  %5 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
  %6 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg1)
  %extracted_slice = tensor.extract_slice %cst_1[%3, 0] [128, 128] [1, 1] : tensor<256x128xf32> to tensor<128x128xf32>
  %extracted_slice_5 = tensor.extract_slice %cst_2[0, %4] [128, 256] [1, 1] : tensor<128x512xf32> to tensor<128x256xf32>
  %extracted_slice_6 = tensor.extract_slice %arg2[%5, %6] [128, 256] [1, 1] : tensor<256x512xf32> to tensor<128x256xf32>
  %c0 = arith.constant 0 : index
  %c0_7 = arith.constant 0 : index
  %c2_8 = arith.constant 2 : index
  %7 = tensor.empty() : tensor<128x256x2xf32>
  %cst_9 = arith.constant 0.000000e+00 : f32
  %8 = linalg.fill ins(%cst_9 : f32) outs(%7 : tensor<128x256x2xf32>) -> tensor<128x256x2xf32>
  %c2_10 = arith.constant 2 : index
  %9 = scf.forall (%arg3) in (2) shared_outs(%arg4 = %8) -> (tensor<128x256x2xf32>) {
    %13 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg3)
    %extracted_slice_11 = tensor.extract_slice %arg4[0, 0, %arg3] [128, 256, 1] [1, 1, 1] : tensor<128x256x2xf32> to tensor<128x256xf32>
    %c0_12 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c128_13 = arith.constant 128 : index
    %14 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%arg3, %c128_13]
    %15 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%14, %c0_12]
    %16 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%c2_10, %c128_13]
    %17 = scf.for %arg5 = %15 to %c128 step %16 iter_args(%arg6 = %extracted_slice_11) -> (tensor<128x256xf32>) {
      %extracted_slice_14 = tensor.extract_slice %extracted_slice[0, %arg5] [128, 128] [1, 1] : tensor<128x128xf32> to tensor<128x128xf32>
      %extracted_slice_15 = tensor.extract_slice %extracted_slice_5[%arg5, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
      %extracted_slice_16 = tensor.extract_slice %arg6[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
      %c0_17 = arith.constant 0 : index
      %c128_18 = arith.constant 128 : index
      %c128_19 = arith.constant 128 : index
      %18 = scf.for %arg7 = %c0_17 to %c128_18 step %c128_19 iter_args(%arg8 = %extracted_slice_16) -> (tensor<128x256xf32>) {
        %c0_20 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c256_21 = arith.constant 256 : index
        %19 = scf.for %arg9 = %c0_20 to %c256 step %c256_21 iter_args(%arg10 = %arg8) -> (tensor<128x256xf32>) {
          %c0_22 = arith.constant 0 : index
          %c128_23 = arith.constant 128 : index
          %c64 = arith.constant 64 : index
          %20 = scf.for %arg11 = %c0_22 to %c128_23 step %c64 iter_args(%arg12 = %arg10) -> (tensor<128x256xf32>) {
            %extracted_slice_24 = tensor.extract_slice %extracted_slice_14[%arg7, %arg11] [128, 64] [1, 1] : tensor<128x128xf32> to tensor<128x64xf32>
            %extracted_slice_25 = tensor.extract_slice %extracted_slice_15[%arg11, %arg9] [64, 256] [1, 1] : tensor<128x256xf32> to tensor<64x256xf32>
            %extracted_slice_26 = tensor.extract_slice %arg12[%arg7, %arg9] [128, 256] [1, 1] : tensor<128x256xf32> to tensor<128x256xf32>
            %c0_27 = arith.constant 0 : index
            %c128_28 = arith.constant 128 : index
            %c32 = arith.constant 32 : index
            %21 = scf.for %arg13 = %c0_27 to %c128_28 step %c32 iter_args(%arg14 = %extracted_slice_26) -> (tensor<128x256xf32>) {
              %c0_30 = arith.constant 0 : index
              %c256_31 = arith.constant 256 : index
              %c32_32 = arith.constant 32 : index
              %22 = scf.for %arg15 = %c0_30 to %c256_31 step %c32_32 iter_args(%arg16 = %arg14) -> (tensor<128x256xf32>) {
                %c0_33 = arith.constant 0 : index
                %c64_34 = arith.constant 64 : index
                %c64_35 = arith.constant 64 : index
                %23 = scf.for %arg17 = %c0_33 to %c64_34 step %c64_35 iter_args(%arg18 = %arg16) -> (tensor<128x256xf32>) {
                  %extracted_slice_36 = tensor.extract_slice %extracted_slice_24[%arg13, %arg17] [32, 64] [1, 1] : tensor<128x64xf32> to tensor<32x64xf32>
                  %extracted_slice_37 = tensor.extract_slice %extracted_slice_25[%arg17, %arg15] [64, 32] [1, 1] : tensor<64x256xf32> to tensor<64x32xf32>
                  %extracted_slice_38 = tensor.extract_slice %arg18[%arg13, %arg15] [32, 32] [1, 1] : tensor<128x256xf32> to tensor<32x32xf32>
                  %expanded = tensor.expand_shape %extracted_slice_36 [[0, 1], [2]] : tensor<32x64xf32> into tensor<1x32x64xf32>
                  %expanded_39 = tensor.expand_shape %extracted_slice_37 [[0, 1], [2]] : tensor<64x32xf32> into tensor<1x64x32xf32>
                  %24 = linalg.batch_reduce_matmul ins(%expanded, %expanded_39 : tensor<1x32x64xf32>, tensor<1x64x32xf32>) outs(%extracted_slice_38 : tensor<32x32xf32>) -> tensor<32x32xf32>
                  %inserted_slice_40 = tensor.insert_slice %24 into %arg18[%arg13, %arg15] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x256xf32>
                  scf.yield %inserted_slice_40 : tensor<128x256xf32>
                }
                scf.yield %23 : tensor<128x256xf32>
              }
              scf.yield %22 : tensor<128x256xf32>
            }
            %inserted_slice_29 = tensor.insert_slice %21 into %arg12[%arg7, %arg9] [128, 256] [1, 1] : tensor<128x256xf32> into tensor<128x256xf32>
            scf.yield %inserted_slice_29 : tensor<128x256xf32>
          }
          scf.yield %20 : tensor<128x256xf32>
        }
        scf.yield %19 : tensor<128x256xf32>
      }
      %inserted_slice = tensor.insert_slice %18 into %arg6[0, 0] [128, 256] [1, 1] : tensor<128x256xf32> into tensor<128x256xf32>
      scf.yield %inserted_slice : tensor<128x256xf32>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %17 into %arg4[0, 0, %arg3] [128, 256, 1] [1, 1, 1] : tensor<128x256xf32> into tensor<128x256x2xf32>
    }
  }
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%9 : tensor<128x256x2xf32>) outs(%extracted_slice_6 : tensor<128x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    %13 = arith.addf %in, %out : f32
    linalg.yield %13 : f32
  } -> tensor<128x256xf32>
  %11 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg0)
  %12 = affine.apply affine_map<(d0) -> (d0 * 256)>(%arg1)
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %10 into %arg2[%11, %12] [128, 256] [1, 1] : tensor<128x256xf32> into tensor<256x512xf32>
  }
}
```