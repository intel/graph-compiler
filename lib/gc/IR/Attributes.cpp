//===-- Attributes.cpp - Graph Compiler IR attribute utilities -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/IR/Attributes.h"
#include "gc/Runtime/GpuRuntime.h"
#include "mlir/Dialect/XeGPU/uArch/uArchCommon.h"

namespace mlir::gc {

const xegpu::uArch::uArch *DevAttrs::getUarch() {
  if (!arch) arch = xegpu::uArch::getUArch(getArch().value_or("bmg"));
  return arch;
}

const std::optional<const char *> DevAttrs::getDeviceArch(int id) {
  auto arch = ::gc::gpu::DeviceInfo::idToArch(id);
  return arch[0] == '\0' ? std::nullopt : std::optional<const char *>(arch);
}

} // namespace mlir::gc
