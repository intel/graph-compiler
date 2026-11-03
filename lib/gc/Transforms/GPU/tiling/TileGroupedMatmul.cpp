#include "TilingUtils.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TILEGROUPEDMATMUL
#define GEN_PASS_DEF_TILEGROUPEDMATMUL
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Tiles 'linalgx.grouped_matmul' at the work-group level and its tile-local
// contraction (or fused multi-output generic) at the reduction level.
//
// The pass must run *after* TileContraction: by then every plain contraction is
// already marked with 'gc.tiling.level', so the WG search only picks up
// grouped_matmul ops, while the sub-group search still finds the contraction or
// fused generic emitted by getTiledImplementation() inside the new WG loop.
struct TileGroupedMatmul final
    : TilingPass<::mlir::gc::impl::TileGroupedMatmulBase<TileGroupedMatmul>> {
  // Size of the K loop of the contraction emitted by getTiledImplementation().
  static constexpr size_t kReductionTile = 64;

  bool isSupportedOp(TilingInterface ti) override {
    return isa<linalgx::GroupedMatmulOp>(ti) || isMatmulOp(ti) ||
           isFusedGroupedMatmulOp(ti);
  }

  void computeWgTiles(Target &tg) override {
    // The generic search handles the (expert, m, n) parallel domain as-is: it
    // sets the leading dims to 1 and picks the M/N tiles from the target's
    // block sizes, so the 2-D matmul emitted by getTiledImplementation() gets a
    // real 2-D sub-group layout. Unlike a block-aligned formulation, the M tile
    // is free here - getTiledImplementation() clamps each tile to the group's
    // real length, so no tile can cross a group boundary.
    if (!isa<linalgx::GroupedMatmulOp>(tg.op.getOperation())) {
      TilingPass::computeWgTiles(tg);
      return;
    }
    // The real row extent is always clamped to the expert's group, so it need
    // not divide the hardware tile. In particular, a bound of 35 must not force
    // one-row tiles; padding handles the partial final tile. Round up to a
    // power of two rather than dropping the bound, so small groups (e.g. 1 or 4
    // rows) still get small tiles.
    size_t rowBound = tg.sizes[linalgx::GroupedMatmulOp::kMDim];
    tg.sizes[linalgx::GroupedMatmulOp::kMDim] = llvm::PowerOf2Ceil(rowBound);
    TilingPass::computeWgTiles(tg);
    tg.sizes[linalgx::GroupedMatmulOp::kMDim] = rowBound;
    // One expert per tile; the search does this already, enforce it anyway
    // because getTiledImplementation() rejects anything else.
    using GMM = linalgx::GroupedMatmulOp;
    tg.tiles[GMM::kExpertDim] = 1;
    tg.sgTiles[GMM::kExpertDim] = 1;
  }

  Type getTilingElementType(Target &tg) override {
    auto grouped = dyn_cast<linalgx::GroupedMatmulOp>(tg.op.getOperation());
    if (!grouped) return TilingPass::getTilingElementType(tg);
    // Offsets are metadata, not matrix elements. Their i32/i64 width must not
    // change the matrix load/store tile sizes.
    auto matrixOperands = llvm::to_vector(llvm::concat<Value>(
        grouped.getLhs(), grouped.getWeights(), grouped.getOutputs()));
    return getWidestElementType(matrixOperands);
  }

  void computeSgTiles(Target &tg) override {
    if (isa<linalgx::GroupedMatmulOp>(tg.op.getOperation())) {
      // Nothing to do: the reduction loop is created when the contraction
      // emitted by getTiledImplementation() is tiled.
      std::fill(tg.tiles.begin(), tg.tiles.end(), 0);
      return;
    }
    TilingPass::computeSgTiles(tg);
    for (auto [i, r] : llvm::enumerate(tg.reductions)) {
      if (r) {
        tg.tiles[i] = kReductionTile;
        tg.sgTiles[i] = kReductionTile;
      }
    }
  }
};

} // namespace
