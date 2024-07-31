//===- Passes.h - Graph Compiler passes -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_PASSES_H
#define GC_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace LLVM {
class LLVMDialect;
}

namespace scf {
class SCFDialect;
}

namespace openmp {
class OpenMPDialect;
}

namespace linalg {
class LinalgDialect;
}
namespace linalgx {
class LinalgxDialect;
}

namespace MemRef {
class MemRefDialect;
}

namespace xegpu {
class XeGPUDialect;
}

class OpPassManager;

namespace bufferization {
class BufferizationDialect;
}

namespace gc {

void populateFrontendPasses(mlir::OpPassManager &);
void populateCPUPipeline(mlir::OpPassManager &);

std::unique_ptr<Pass> createBufferNestedParallelLoopHoistingPass();

#define GEN_PASS_DECL
#include "gc/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

#endif // GC_PASSES_H
