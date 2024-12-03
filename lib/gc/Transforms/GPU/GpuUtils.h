//===-- GpuUtils.h - DESC ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GPUUTILS_H
#define GPUUTILS_H

#include "gc/Utils/Log.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <numeric>

using namespace mlir;

namespace mlir::gc {
template <typename DerivedT> struct GpuPass {

  int64_t getGpuPropertyAsInt(Builder &builder, StringRef name,
                              int64_t defaultValue) {
    if (auto mod = static_cast<DerivedT *>(this)
                       ->getOperation()
                       ->template getParentOfType<ModuleOp>()) {
      DataLayout layout(mod);
      if (auto value = layout.getDevicePropertyValue(
              builder.getStringAttr("GPU" /* device ID*/),
              builder.getStringAttr(name))) {
        if (auto attr = dyn_cast<IntegerAttr>(*value)) {
          return attr.getInt();
        }
      }
    }
    return defaultValue;
  }

  int64_t getNumEus(Builder &builder) {
    return getGpuPropertyAsInt(builder, "num_exec_units",
                               static_cast<DerivedT *>(this)->numEus);
  }

  int64_t getNumEusPerSlice(Builder &builder) {
    return getGpuPropertyAsInt(builder, "num_exec_units_per_slice",
                               static_cast<DerivedT *>(this)->numEusPerSlice);
  }

  int64_t getNumThreadsPerEu(Builder &builder) {
    return getGpuPropertyAsInt(builder, "num_threads_per_eu",
                               static_cast<DerivedT *>(this)->numThreadsPerEu);
  }

  int64_t getLocalMemSize(Builder &builder) {
    return getGpuPropertyAsInt(builder, "local_mem_size",
                               static_cast<DerivedT *>(this)->localMemSize);
  }

  int64_t getVectorWidth(Builder &builder) {
    return getGpuPropertyAsInt(builder, "max_vector_op_width",
                               static_cast<DerivedT *>(this)->vectorWidth);
  }

  int64_t getWorkGroupSize(Builder &builder) {
    return getGpuPropertyAsInt(builder, "max_work_group_size",
                               static_cast<DerivedT *>(this)->workGroupSize);
  }
};

// This class is a placeholder for the rewriter-related boilerplate code.
struct OpRewriter final : IRRewriter {
  Location loc;

  explicit OpRewriter(func::FuncOp &func)
      : IRRewriter(func.getContext()), loc(func.getLoc()) {}

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    return RewriterBase::create<OpTy>(loc, std::forward<Args>(args)...);
  }

  arith::ConstantIndexOp createConstant(int64_t v) {
    return create<arith::ConstantIndexOp>(v);
  }

  arith::ConstantFloatOp createConstant(double v) {
    return create<arith::ConstantFloatOp>(APFloat(v), getF64Type());
  }
};

template <typename T> static T isPow2(T value) {
  assert(value > 0);
  return (value & (value - 1)) == 0;
}

// Round to the largest power of 2 that is <= value.
template <typename T> static T floorPow2(T value) {
  auto v = static_cast<std::make_unsigned_t<T>>(value);
  return T(1) << (llvm::bit_width(v) - 1);
}

// Round to the smallest power of 2 that is >= value.
template <typename T> static T ceilPow2(T value) {
  auto v = static_cast<std::make_unsigned_t<T>>(value);
  return llvm::bit_ceil(v);
}

// Find a factor of the number that is close to the given value and, if
// possible, is a power of 2.
template <typename T> T findFactor(T number, T closeTo) {
  closeTo = std::max(T(1), std::min(closeTo, number));

  for (T max = number - closeTo + 1, i = 0; i < max; ++i) {
    T up = closeTo + i;
    if (auto pow2 = ceilPow2(up); number % pow2 == 0) {
      return pow2;
    }
    if (i < closeTo - 1) {
      T down = closeTo - i;
      if (auto pow2 = floorPow2(down); pow2 != 1 && number % pow2 == 0) {
        return pow2;
      }
      if (number % down == 0) {
        return down;
      }
    }
    if (number % up == 0) {
      return up;
    }
  }

  return closeTo;
}

namespace impl {
// Controls the adjustment in case of more than 2 tiles.
enum class AdjustTilesMode {
  // Sort the input and switch to the First mode.
  Sort,
  // Adjust the first tile and call adjustTiles() recursively for the rest.
  First,
  // To allow for squeezing, set 1's for all tiles except the last 2.
  XeGpu,
};

template <typename T>
static void adjustTwoTiles(T totalSize, T *aPtr, T *bPtr,
                           AdjustTilesMode mode) {
  T a = *aPtr;
  T b = *bPtr;
  assert(a >= b);

  if (a * b <= totalSize) {
    return;
  }

  T minSize = static_cast<T>(mode == AdjustTilesMode::XeGpu ? 8 : 1);
  bool aPow2 = isPow2(a);
  bool bPow2 = isPow2(b);
  double ratio = static_cast<double>(a) / static_cast<double>(b);
  T x = static_cast<T>(std::sqrt(totalSize)) * static_cast<T>(std::sqrt(ratio));
  T y;

  if (aPow2) {
    x = std::min(ceilPow2(x), std::min(a, floorPow2(totalSize)));
  } else {
    x = std::min(findFactor(a, x), std::min(a, totalSize));
  }
  x = std::max(x, minSize);
  if (bPow2) {
    y = std::min(floorPow2(totalSize / x), b);
  } else {
    y = std::min(findFactor(b, totalSize / x), b);
  }
  if (y < minSize && a >= minSize && b >= minSize) {
    if (auto newX = ceilPow2(totalSize / minSize); newX >= minSize) {
      x = std::min(newX, a);
      y = minSize;
    }
  }

  // Adjust x and y to get the closest ratio
  auto distance =
      std::abs(ratio - static_cast<double>(x) / static_cast<double>(y));
  auto ax = aPow2 ? x * 2 : findFactor(a, x * 2);
  auto ay = std::max(bPow2 ? y / 2 : findFactor(b, y / 2), minSize);

  if (ax * ay <= totalSize &&
      std::abs(ratio - static_cast<double>(ax) / static_cast<double>(ay)) <
          distance) {
    x = ax;
    y = ay;
  } else {
    ax = std::max(aPow2 ? x / 2 : findFactor(a, x / 2), minSize);
    ay = bPow2 ? y * 2 : findFactor(b, y * 2);
    if (ax * ay <= totalSize &&
        std::abs(ratio - static_cast<double>(ax) / static_cast<double>(ay)) <
            distance) {
      x = ax;
      y = ay;
    }
  }

  *aPtr = x;
  *bPtr = y;
}

// Adjust tile sizes that meet the following conditions:
// 1. The product of all tiles is as close to totalSize as possible.
// 2. The new sizes are proportional to the initial sizes.
// 3. If the initial size is a power of 2, then the resulting size is a power of
//    2 either. Otherwise, the resulting size is a factor of the initial size
//    and, if possible, is a power of 2.
template <typename T>
static void adjustTiles(T totalSize, T *begin, T *end,
                        AdjustTilesMode mode = AdjustTilesMode::Sort) {
  auto count = end - begin;
  if (count == 0) {
    return;
  }

  if (count == 1) {
    T minSize = static_cast<T>(mode == AdjustTilesMode::XeGpu ? 8 : 1);
    if (T a = *begin; isPow2(a)) {
      *begin = std::min(std::max(ceilPow2(a), minSize), floorPow2(totalSize));
    } else {
      *begin = std::min(findFactor(a, totalSize), minSize);
    }
    return;
  }

  if (count > 2) {
    if (mode == AdjustTilesMode::XeGpu) {
      for (unsigned i = 0; i < count - 2; ++i) {
        *(begin + i) = 1;
      }
      T *aPtr = end - 2;
      T *bPtr = end - 1;
      if (*aPtr < *bPtr) {
        std::swap(aPtr, bPtr);
      }
      adjustTwoTiles(totalSize, aPtr, bPtr, mode);
      return;
    }

    SmallVector<T> sorted;
    SmallVector<unsigned> indices;
    T *head;
    T *tail;

    if (mode == AdjustTilesMode::First) {
      head = begin;
      tail = end;
    } else {
      assert(mode == AdjustTilesMode::Sort);
      SmallVector<std::pair<T, unsigned>> pairs;
      pairs.reserve(count);
      for (unsigned i = 0; i < count; ++i) {
        pairs.emplace_back(*(begin + i), i);
      }
      llvm::sort(pairs);
      sorted.reserve(count);
      indices.reserve(count);
      for (auto &p : pairs) {
        sorted.push_back(p.first);
        indices.push_back(p.second);
      }
      head = sorted.data();
      tail = head + count;
    }

    // Split the array in two. The first one consists of the 2 elements - the
    // first one and the product of the rest. The second one is the rest.
    T first[] = {*head, std::accumulate(head + 2, tail, *(head + 1),
                                        std::multiplies<>())};
    adjustTiles(totalSize, first, first + 2, AdjustTilesMode::First);
    adjustTiles(totalSize / *first, head + 1, tail, AdjustTilesMode::First);
    *head = *first;

    if (mode == AdjustTilesMode::Sort) {
      for (unsigned i = 0; i < count; ++i) {
        *(begin + indices[i]) = sorted[i];
      }
    }
  } else if (*begin >= *(end - 1)) {
    adjustTwoTiles(totalSize, begin, end - 1, mode);
  } else {
    adjustTwoTiles(totalSize, end - 1, begin, mode);
  }
}
} // namespace impl

template <typename T, unsigned N>
static void adjustTiles(T totalSize, SmallVector<T, N> &tiles,
                        bool xeGpuMode = false) {
  impl::adjustTiles(totalSize, tiles.begin(), tiles.end(),
                    xeGpuMode ? impl::AdjustTilesMode::XeGpu
                              : impl::AdjustTilesMode::Sort);
}

// Check recursively if the specified operation has an operand that
// depends on a result of a previous operation, matching the predicate.
template <unsigned MaxDepth = std::numeric_limits<unsigned>::max()>
bool isOperandDependsOnOp(bool (*predicate)(Operation *), Operation *operation,
                          unsigned depth = 0) {
  for (auto operand : operation->getOperands()) {
    if (auto op = operand.getDefiningOp();
        op &&
        (predicate(op) || (depth < MaxDepth &&
                           isOperandDependsOnOp(predicate, op, depth + 1)))) {
      return true;
    }
  }
  return false;
}

// Check recursively if there are any operation, matching the predicate, that
// depends on the result of the specified operation.
template <unsigned MaxDepth = std::numeric_limits<unsigned>::max()>
bool isOpDependsOnResult(bool (*predicate)(Operation *), Operation *operation,
                         unsigned depth = 0) {
  for (auto res : operation->getResults()) {
    for (auto u : res.getUsers()) {
      if (predicate(u) ||
          (depth < MaxDepth && isOpDependsOnResult(predicate, u, depth + 1))) {
        return true;
      }
    }
  }
  return false;
}
} // namespace mlir::gc
#endif
