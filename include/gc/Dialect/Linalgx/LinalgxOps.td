//===- LinalgxOps.td - linalgx dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALGX_OPS
#define LINALGX_OPS

include "LinalgxDialect.td"

include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for Linalg dialect ops that do not correspond to library calls.
class Linalgx_Op<string mnemonic, list<Trait> traits = []> :
    Op<LinalgxDialect, mnemonic, traits>;

def Linalgx_AttentionOp : Linalgx_Op<"attention",
    [
     DeclareOpInterfaceMethods<IndexingMapOpInterface, ["getMatchingIndexingMap"]>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation",
       "generateResultTileValue"]>]> {
  let summary = [{Attention operator.}];
  let description = [{
    Computes the scaled dot product attention function:

    attention(Q, K, V, scale) = softmax(Q @ K.T * scale) @ V

    Here Q, K, V are given tensors and scale is a scalar value specifying
    the scale to use.

    If an additional mask argument M is included, the result of the first matmul is modified according to:

    Q @ K.T += M
  }];

  let arguments = (ins AnyShaped:$query,
                       AnyShaped:$key,
                       AnyShaped:$value,
                       AnyFloat:$scale,
                       Optional<AnyShaped>:$mask,
                       AnyShaped:$output,
                       AffineMapArrayAttr:$indexing_maps,
                       OptionalAttr<DictionaryAttr>:$decomposition_config
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    attr-dict
    prop-dict
    `ins` `(` $query `,` $key `,` $value `,` $scale (`,` $mask^)?  `:` type($query) `,` type($key) `,` type($value) `,` type($scale) (`,` type($mask)^ )?`)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results,
      "Value":$query,
      "Value":$key,
      "Value":$value,
      "Value":$scale,
      "Value":$output,
      "ArrayAttr":$indexing_maps,
      CArg<"std::optional<Value>", "std::nullopt">:$mask)>,
    OpBuilder<(ins "TypeRange":$results,
      "ValueRange":$inputOperands,
      "ValueRange":$initOperands,
      "ArrayAttr":$indexing_maps)>
  ];

  let extraClassDeclaration = [{
    SmallVector<AffineMap> getIndexingMapsArray();

    ArrayRef<int64_t> getShape(OpOperand* opOperand);

    AffineMap getQueryMap() {
      return cast<AffineMap>(getIndexingMapsArray()[0]);
    }
    AffineMap getKeyMap() {
      return cast<AffineMap>(getIndexingMapsArray()[1]);
    }
    AffineMap getValueMap() {
      return cast<AffineMap>(getIndexingMapsArray()[2]);
    }
    AffineMap getScaleMap() {
      return cast<AffineMap>(getIndexingMapsArray()[3]);
    }
    std::optional<AffineMap> getMaskMap() {
      if (getMask()) {
        return cast<AffineMap>(getIndexingMapsArray()[4]);
      }
      return std::nullopt;
    }
    AffineMap getOutputMap() {
      return cast<AffineMap>(getIndexingMapsArray()[getNumDpsInputs()]);
    }
    int64_t getIterationDomainRank() {
      return getQueryMap().getNumDims();
    }

    /* Decomposition control attributes */

    // Attributes to set on QK and PV matmul after decomposition.
    static StringRef getQKAttrStr() { return "qk_attrs"; }
    static StringRef getPVAttrStr() { return "pv_attrs"; }
  }];

  let hasCanonicalizer = 1;
}

def Linalgx_GroupedMatmulOp : Linalgx_Op<"grouped_matmul",
    [
     DeclareOpInterfaceMethods<IndexingMapOpInterface, ["getMatchingIndexingMap"]>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation",
       "generateResultTileValue"]>]> {
  let summary = [{Grouped matmul with the semantics of PyTorch's grouped_mm.}];
  let description = [{
    Same as `torch._grouped_mm(lhs, weights.transpose(-2, -1), offs=offsets)`:
    the rows of `lhs` are split into consecutive groups, one per expert, and
    each group is multiplied by its own expert's weights.

      lhs      [total, K]
      weights  [E, N, K]   (K innermost, i.e. transposed)
      offsets  [E]         end row of each group (i32/i64)
      output   [total, N]

    Example: 3 experts with 2, 0 and 3 rows, so `total = 5` and
    `offsets = [2, 2, 5]`:

      rows [0, 2) -> out = lhs[0:2] * weights[0]^T
      rows [2, 2) -> expert 1 has no rows
      rows [2, 5) -> out = lhs[2:5] * weights[2]^T

    `max_group_size` is an optional performance hint: an upper bound on the
    number of rows in any group. It defaults to `total` and only shrinks the
    loop over rows, but it must be a real bound - rows of a group longer than
    `max_group_size` are not computed.
  }];

  let arguments = (ins AnyShaped:$lhs,
                       AnyShaped:$weights,
                       AnyShaped:$offsets,
                       Optional<Index>:$max_group_size,
                       AnyShaped:$output,
                       AffineMapArrayAttr:$indexing_maps
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict
    prop-dict
    `ins` `(` $lhs `,` $weights `,` $offsets `:`
              type($lhs) `,` type($weights) `,` type($offsets) `)`
    (`max_group_size` `(` $max_group_size^ `)`)?
    `outs` `(` $output `:` type($output) `)`
    (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    SmallVector<AffineMap> getIndexingMapsArray();

    ArrayRef<int64_t> getShape(OpOperand* opOperand);

    /* `indexing_maps` holds one map per *shaped* operand, in operand order;
       `max_group_size` is a scalar and has no map. */
    AffineMap getLhsMap() { return getIndexingMapsArray()[0]; }
    AffineMap getWeightsMap() { return getIndexingMapsArray()[1]; }
    AffineMap getOffsetsMap() { return getIndexingMapsArray()[2]; }
    AffineMap getOutputMap() { return getIndexingMapsArray()[3]; }

    int64_t getIterationDomainRank() { return getLhsMap().getNumDims(); }

    /* Iteration domain dim positions. */
    static constexpr int64_t kExpertDim = 0;
    static constexpr int64_t kMDim = 1;
    static constexpr int64_t kNDim = 2;
    static constexpr int64_t kKDim = 3;
  }];
}

#endif // LINALGX_OPS
