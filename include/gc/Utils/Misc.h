//===-- Misc.h - Miscellaneous utilities -------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MISCUTILS_H
#define MISCUTILS_H

#include <algorithm>
#include <cassert>
#include <numeric>

#include "llvm/ADT/bit.h"

template <typename T> static T isPow2(T value) {
  assert(value > 0);
  return (value & (value - 1)) == 0;
}

// Round to the largest power of 2 that is <= value.
template <typename T> static T floorPow2(T value) {
  assert(value > 0);
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

template <typename L, typename T> static T findClosestDiv(L &sorted, T value) {
  for (int i = sorted.size() - 1; i >= 0; --i) {
    if (value % sorted[i] == 0) {
      return static_cast<T>(sorted[i]);
    }
  }
  return static_cast<T>(1);
}
#endif // MISCUTILS_H