//===--------- SetGpuDeviceProps.cpp - Set GPU device properties ----------*-
// C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Utils/Transform.h"
#include "mlir/Conversion/Passes.h"

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUDEVICEPROPS
#define GEN_PASS_DEF_GPUDEVICEPROPS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct GpuDeviceProps final
    : gc::impl::GpuDevicePropsBase<GpuDeviceProps> {
  explicit GpuDeviceProps()
      : GpuDeviceProps(GpuDevicePropsOptions{}) {}
  explicit GpuDeviceProps(const GpuDevicePropsOptions &opts)
      : GpuDevicePropsBase(opts) {}

  void runOnOperation() override {
    DevAttrs dev(getOperation());

    if (!dev.getId() && id != 0) {
      dev.setId(id);
    }
    if (!dev.getName() && !name.empty()) {
      dev.setName(name);
    }
    if (!dev.getArch()) {
      auto a = arch.c_str();
      if (auto id = dev.getId())
        a = dev.getDeviceArch(*id).value_or(a);
      dev.setArch(a);
    }
    if (!dev.getMaxWgSize())
      dev.setMaxWgSize(maxWgSize);
    if (!dev.getVectorWidth())
      dev.setVectorWidth(vectorWidth);
    if (!dev.getSgSizes()) {
      if (sgSizes.empty())
        dev.setSgSizes({16, 32});
      else
        dev.setSgSizes(sgSizes);
    }
  }
};

} // namespace
