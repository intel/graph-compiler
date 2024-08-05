//===- GPULegalizeModule.cpp - Legalize target for gpu module ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

using namespace mlir;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_GPULEGALIZEMODULE
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

struct GpuLegalizeModule
    : public gc::impl::GpuLegalizeModuleBase<GpuLegalizeModule> {
  using GpuLegalizeModuleBase::GpuLegalizeModuleBase;

  void runOnOperation() override;
};

void GpuLegalizeModule::runOnOperation() {
  OpBuilder builder(&getContext());
  using namespace mlir::spirv;

  auto version = Version::V_1_0;
  SmallVector<Capability, 4> capabilities = {
      Capability::Addresses, Capability::Int64, Capability::Kernel};
  SmallVector<Extension> extensions{};

  auto caps = ArrayRef<Capability>(capabilities);
  auto exts = ArrayRef<Extension>(extensions);
  VerCapExtAttr vce = VerCapExtAttr::get(version, caps, exts, &getContext());

  auto limits = ResourceLimitsAttr::get(
      &getContext(), /*max_compute_shared_memory_size=*/16384,
      /*max_compute_workgroup_invocations=*/128,
      /*max_compute_workgroup_size=*/builder.getI32ArrayAttr({128, 128, 64}),
      /*subgroup_size=*/16,
      /*min_subgroup_size=*/std::nullopt,
      /*max_subgroup_size=*/std::nullopt,
      /*cooperative_matrix_properties_khr=*/ArrayAttr{},
      /*cooperative_matrix_properties_nv=*/ArrayAttr{});

  auto target = spirv::TargetEnvAttr::get(
      vce, limits, ClientAPI::OpenCL, Vendor::Intel, DeviceType::DiscreteGPU,
      TargetEnvAttr::kUnknownDeviceID);

  getOperation()->walk([&](gpu::GPUModuleOp gpuModule) {
    SmallVector<Attribute> targets;
    if (std::optional<ArrayAttr> attrs = gpuModule.getTargets())
      targets.append(attrs->getValue().begin(), attrs->getValue().end());
    targets.push_back(target);
    // Remove any duplicate targets.
    targets.erase(llvm::unique(targets), targets.end());
    gpuModule.setTargetsAttr(builder.getArrayAttr(targets));
  });
}
