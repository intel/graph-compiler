#include "TilingUtils.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TILEATTENTION
#define GEN_PASS_DEF_TILEATTENTION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
struct TileAttention final
    : TilingPass<gc::impl::TileAttentionBase<TileAttention>> {

  bool isSupportedOp(TilingInterface ti) override {
    return isa<linalgx::AttentionOp>(ti);
  }

  void computeSgTiles(Target &tg) override {
    std::fill(tg.tiles.begin(), tg.tiles.end(), 0);
    tg.kernelAttrs.setThreads({128, 1, 1});
  }

  void computeWgTiles(Target &tg) override {
    std::fill(tg.tiles.begin(), tg.tiles.end() - 2, 1);
    tg.tiles[tg.tiles.size() - 2] = 128;
    tg.tiles[tg.tiles.size() - 1] = 0;
  }
};
} // namespace
