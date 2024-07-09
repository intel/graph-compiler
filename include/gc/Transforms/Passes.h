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

class PassManager;

namespace gc {

void populateFrontendPasses(mlir::PassManager &);
void populateCPUPipeline(mlir::PassManager &);

/// Creates a pass that finds two consecutive matmuls, tiles them and fuses them.
std::unique_ptr<Pass> createMatmulSpecialTileAndFusePass();

#define GEN_PASS_DECL
#include "gc/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

#endif // GC_PASSES_H
