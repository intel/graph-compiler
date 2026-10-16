//===-- Misc.h - Miscellaneous utilities ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <cstdlib>
#include <type_traits>

#include <llvm/ADT/StringRef.h>

template <typename T = const char *>
T gcGetEnv(const char *name, T defaultValue = T()) {
  auto env = std::getenv(name);
  if (!env) return defaultValue;

  if constexpr (std::is_same_v<T, const char *>) {
    return env;
  } else if constexpr (std::is_same_v<T, bool>) {
    return env[0] != '\0' && env[0] != '0';
  } else if constexpr (std::is_convertible_v<llvm::StringRef, T>) {
    return (T)llvm::StringRef(env);
  } else if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(std::atoi(env));
  } else if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(std::atof(env));
  } else {
    static_assert(std::is_same_v<T, void>, "Unsupported type");
  }
}
