// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef GC_OCL_SOURCE_MODULE_H
#define GC_OCL_SOURCE_MODULE_H

#include <memory>
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace mlir::gc::gpu {

class OclSourceModule {
public:
  static llvm::Expected<OclSourceModule>
  build(ModuleOp module, cl_device_id device, cl_context context);
  llvm::Error invoke(cl_command_queue queue, llvm::ArrayRef<cl_mem> args,
                     llvm::ArrayRef<cl_event> waitList,
                     cl_event *completion) const;
  llvm::Error invoke(cl_command_queue queue, llvm::ArrayRef<cl_mem> args,
                     llvm::ArrayRef<llvm::ArrayRef<int64_t>> shapes,
                     llvm::ArrayRef<cl_event> waitList,
                     cl_event *completion) const;
  bool isStatic() const;

private:
  struct Impl;
  explicit OclSourceModule(std::shared_ptr<Impl> impl)
      : impl(std::move(impl)) {}
  std::shared_ptr<Impl> impl;
};

bool useOclSourceBackend(cl_device_id device);

} // namespace mlir::gc::gpu
#endif
