//===- MatcherUtils.h - Matcher utils ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_MATCHERUTILS_H
#define GC_MATCHERUTILS_H

namespace mlir {
class Value;
namespace linalg {
class LinalgOp;
class GenericOp;
} // namespace linalg
namespace structured_match {
namespace utils {

// Returns true if the linalg operation is a 2d eltwsie floating point addition.
bool isTwoDAddOp(linalg::LinalgOp linalgOp,
                 SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic is a 2d eltwise floating point relu
// operation.
bool isTwoDReluOp(linalg::LinalgOp linalgOp,
                  SmallVectorImpl<Value> *capturedOperands = nullptr);

} // namespace utils
} // namespace structured_match
} // namespace mlir

#endif // GC_MATCHERUTILS_H
