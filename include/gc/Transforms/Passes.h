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
class OpBuilder;
class SymbolTable;
class ModuleOp;

namespace func {
class FuncOp;
} // namespace func


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
namespace xegpu {
class XeGPUDialect;
}

class OpPassManager;
class BufferViewFlowAnalysis;

namespace gc {

/// abstract base class for lifetime of buffers in the same "allocation scope".
/// It should hold the lifetime informantion of buffers that are to be merged in
/// the same allocation in an "allocation scope". TraceCollectorFunc decides
/// which buffers are put into which "allocation scope".
class LifetimeTrace {
public:
  enum TraceKind { TK_TICK };
  virtual ~LifetimeTrace() = default;
  LifetimeTrace(TraceKind kind) : kind{kind} {}
  TraceKind getKind() const { return kind; }
  virtual Block *getAllocScope() const = 0;
  virtual Attribute getMemorySpace() const = 0;

private:
  TraceKind kind;
};

/// top level memory trace info for multiple scopes. Each element of scopeTraces
/// should contain an "allocation scope" and the implementation-defined lifetime
/// data
struct MemoryTraceScopes {
  llvm::SmallVector<std::unique_ptr<LifetimeTrace>> scopeTraces;
  MemoryTraceScopes() = default;
};

/// the memory scheduling result for allocations in the same allocation scope.
/// allocation => offset map. All Operation* in the map should be
/// memref::AllocOp which are in the same LifetimeTrace.
struct MemorySchedule {
  size_t totalSize;
  Attribute memorySpace;
  llvm::DenseMap<Operation *, int64_t> allocToOffset;
  MemorySchedule() : totalSize{0} {}
};

struct MergeAllocationOptions;
using TraceCollectorFunc = std::function<FailureOr<MemoryTraceScopes>(
    Operation *, const BufferViewFlowAnalysis &,
    const MergeAllocationOptions &)>;
using MemoryPlannerFunc = std::function<FailureOr<MemorySchedule>(
    Operation *, const LifetimeTrace &, const MergeAllocationOptions &)>;
using MemoryMergeMutatorFunc = std::function<LogicalResult(
    Operation *toplevel, Block *scope, const MemorySchedule &,
    const MergeAllocationOptions &)>;

struct MergeAllocationOptions {
  bool checkOnly = false;
  std::string plannerOptions;
  int64_t alignment = 64;
  TraceCollectorFunc tracer;
  MemoryPlannerFunc planner;
  MemoryMergeMutatorFunc mutator;
};

/// Creates an operation pass to merge the local memref allocations
std::unique_ptr<Pass> createMergeAllocPass(const MergeAllocationOptions &o);
std::unique_ptr<Pass> createMergeAllocPass();

void populateFrontendPasses(mlir::OpPassManager &);
void populateCPUPipeline(mlir::OpPassManager &);

#ifdef GC_USE_IMEX
void populateGPUPipeline(mlir::OpPassManager &);
#endif

#define GEN_PASS_DECL
#include "gc/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

#endif // GC_PASSES_H
