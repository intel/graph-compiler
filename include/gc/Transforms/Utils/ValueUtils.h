//===-- ValueUtils.h - Zero-checking utilities ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORMS_UTILS_VALUEUTILS_H
#define GC_TRANSFORMS_UTILS_VALUEUTILS_H

namespace mlir {
class Value;
class OpBuilder;
namespace utils {

// Returns true if the value is a constant float or integer.
bool isValConstZero(Value val);

// Returns true if the op defining `val` represents a zero filled tensor.
bool isZeroTensor(Value val);

// Returns the strides of `val`. The method returns something useful
// only if the `val` type is a strided memref.
FailureOr<SmallVector<int64_t>> getStrides(Value val);

// Returns the strides of `val`. The method returns something useful
// only if the `val` type is a strided memref and the strides are statically
// known.
FailureOr<SmallVector<int64_t>> getStaticStrides(Value val);

// Return the offset and ptr for `val`. Assert if `val`
// is not a memref.
std::pair<Value, Value> getPtrAndOffset(OpBuilder &builder, Value operand);

template <typename T>
Value createTypedVector(PatternRewriter &rewriter, Location loc,
                        ArrayRef<T> values, Type elementType) {
  mlir::VectorType vectorType =
      mlir::VectorType::get({static_cast<int64_t>(values.size())}, elementType);
  mlir::DenseElementsAttr denseAttr =
      mlir::DenseElementsAttr::get(vectorType, values);
  auto vector =
      rewriter.create<mlir::arith::ConstantOp>(loc, vectorType, denseAttr)
          .getResult();
  return vector;
}

Value flattenMemref(PatternRewriter &rewriter, Location loc, Value srcMemref);

bool hasSharedMemSpace(mlir::Value memref);

} // namespace utils
} // namespace mlir

#endif // GC_TRANSFORMS_UTILS_VALUEUTILS_H
