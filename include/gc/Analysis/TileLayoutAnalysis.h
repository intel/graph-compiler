#ifndef MLIR_ANALYSIS_TILELAYOUT_ANALYSIS_H
#define MLIR_ANALYSIS_TILELAYOUT_ANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

namespace mlir {

class CallOpInterface;
class CallableOpInterface;
class BranchOpInterface;
class RegionBranchOpInterface;
class RegionBranchTerminatorOpInterface;

namespace gc {

//===----------------------------------------------------------------------===//
// DeadCodeAnalysis
//===----------------------------------------------------------------------===//

/// Dead code analysis analyzes control-flow, as understood by
/// `RegionBranchOpInterface` and `BranchOpInterface`, and the callgraph, as
/// understood by `CallableOpInterface` and `CallOpInterface`.
///
/// This analysis uses known constant values of operands to determine the
/// liveness of each block and each edge between a block and its predecessors.
/// For region control-flow, this analysis determines the predecessor operations
/// for region entry blocks and region control-flow operations. For the
/// callgraph, this analysis determines the callsites and live returns of every
/// function.
class TileLayoutAnalysis {
public:
  explicit TileLayoutAnalysis(mlir::Operation *op);

  // /// Initialize the analysis by visiting every operation with potential
  // /// control-flow semantics.
  // LogicalResult initialize(Operation *top) override;

  /// Visit an operation with control-flow semantics and deduce which of its
  /// successors are live.
  LogicalResult visit(Operation *op);

private:
  // /// Find and mark symbol callables with potentially unknown callsites as
  // /// having overdefined predecessors. `top` is the top-level operation that
  // the
  // /// analysis is operating on.
  // void initializeSymbolCallables(Operation *top);

  // /// Recursively Initialize the analysis on nested regions.
  // LogicalResult initializeRecursively(Operation *op);

  // /// Visit the given call operation and compute any necessary lattice state.
  // void visitCallOperation(CallOpInterface call);

  // /// Visit the given branch operation with successors and try to determine
  // /// which are live from the current block.
  // void visitBranchOperation(BranchOpInterface branch);

  // /// Visit the given region branch operation, which defines regions, and
  // /// compute any necessary lattice state. This also resolves the lattice
  // state
  // /// of both the operation results and any nested regions.
  // void visitRegionBranchOperation(RegionBranchOpInterface branch);

  // /// Visit the given terminator operation that exits a region under an
  // /// operation with control-flow semantics. These are terminators with no
  // CFG
  // /// successors.
  // void visitRegionTerminator(Operation *op, RegionBranchOpInterface branch);

  // /// Visit the given terminator operation that exits a callable region.
  // These
  // /// are terminators with no CFG successors.
  // void visitCallableTerminator(Operation *op, CallableOpInterface callable);

  // /// Mark the edge between `from` and `to` as executable.
  // void markEdgeLive(Block *from, Block *to);

  // /// Mark the entry blocks of the operation as executable.
  // void markEntryBlocksLive(Operation *op);

  // /// Get the constant values of the operands of the operation. Returns
  // /// std::nullopt if any of the operand lattices are uninitialized.
  // std::optional<SmallVector<Attribute>> getOperandValues(Operation *op);

  // /// The top-level operation the analysis is running on. This is used to
  // detect
  // /// if a callable is outside the scope of the analysis and thus must be
  // /// considered an external callable.
  // Operation *analysisScope;

  // /// A symbol table used for O(1) symbol lookups during simplification.
  // SymbolTableCollection symbolTable;
};

} // end namespace gc
} // end namespace mlir

#endif // MLIR_ANALYSIS_TILELAYOUT_ANALYSIS_H