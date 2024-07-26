//===-- TilingUtil.hpp - Tile op using tiling interface ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEMPORARY_TILEUSINGINTERFACE_X_H
#define TEMPORARY_TILEUSINGINTERFACE_X_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/TilingInterface.h"
#include <optional>
namespace mlir {
namespace linalgX {

// An enahncement for the upstream pass to support tiling reduction for MKmk
// like cases(with multiple reduction iterators).
FailureOr<linalg::ForallReductionTilingResult> tileReductionUsingForall(
    RewriterBase &b, PartialReductionOpInterface op,
    ArrayRef<OpFoldResult> threadNums, ArrayRef<OpFoldResult> tileSizes,
    ArrayRef<OpFoldResult> newParallelDims, std::optional<ArrayAttr> mapping);
} // namespace linalgX
} // namespace mlir

#endif