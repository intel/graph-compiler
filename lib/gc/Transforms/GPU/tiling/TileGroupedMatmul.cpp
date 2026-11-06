#include "TilingUtils.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TILEGROUPEDMATMUL
#define GEN_PASS_DEF_TILEGROUPEDMATMUL
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Tiles 'linalgx.grouped_matmul' at the work-group level and the contraction it
// decomposes into at the sub-group level.
//
// The pass must run *after* TileContraction: by then every plain contraction is
// already marked with 'gc.tiling.level', so the WG search only picks up
// grouped_matmul ops, while the sub-group search still finds the contraction
// this op's getTiledImplementation() emits inside the freshly created WG loop.
struct TileGroupedMatmul final
    : TilingPass<::mlir::gc::impl::TileGroupedMatmulBase<TileGroupedMatmul>> {
  // Size of the K loop of the contraction emitted by getTiledImplementation().
  static constexpr size_t kReductionTile = 64;

  bool isSupportedOp(TilingInterface ti) override {
    return isa<linalgx::GroupedMatmulOp>(ti) || isMatmulOp(ti);
  }

  void computeWgTiles(Target &tg) override {
    // The generic search handles the (expert, m, n) parallel domain as-is: it
    // sets the leading dims to 1 and picks the M/N tiles from the target's
    // block sizes, so the 2-D matmul emitted by getTiledImplementation() gets a
    // real 2-D sub-group layout. Unlike a block-aligned formulation, the M tile
    // is free here - getTiledImplementation() clamps each tile to the group's
    // real length, so no tile can cross a group boundary.
    TilingPass::computeWgTiles(tg);
    if (!isa<linalgx::GroupedMatmulOp>(tg.op.getOperation())) return;
    // One expert per tile; the search does this already, enforce it anyway
    // because getTiledImplementation() rejects anything else.
    using GMM = linalgx::GroupedMatmulOp;
    tg.tiles[GMM::kExpertDim] = 1;
    tg.sgTiles[GMM::kExpertDim] = 1;
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
