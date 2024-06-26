//===-- DnnlTestUtils.h - Test utils ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#if __cplusplus > 202002L
#include <stdfloat>
#else
namespace std {
#if defined(__SIZEOF_FLOAT__) && __SIZEOF_FLOAT__ == 4
using float32_t = float;
#elif defined(__SIZEOF_DOUBLE__) && __SIZEOF_DOUBLE__ == 4
using float32_t = double;
#else
static_assert(false, "No 32-bit floating point type available");
#endif
} // namespace std
#endif

static std::string readStrResource(const std::string &name) {
  std::filesystem::path res_dir{"resources"};
  auto path = std::filesystem::absolute(res_dir / name);
  std::ifstream file(path);

  if (!file) {
    throw std::runtime_error("Unable to open file " + path.string());
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  return buffer.str();
}
