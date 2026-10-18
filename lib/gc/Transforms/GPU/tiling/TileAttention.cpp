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
    : TilingPass<::mlir::gc::impl::TileAttentionBase<TileAttention>> {

  bool isSupportedOp(TilingInterface ti) override {
    return isa<linalgx::AttentionOp>(ti);
  }

  void computeSgTiles(Target &tg) override {
    std::fill(tg.tiles.begin(), tg.tiles.end(), 0);
  }

  void computeWgTiles(Target &tg) override {
    std::fill(tg.tiles.begin(), tg.tiles.end() - 4, 1);
    tg.tiles[tg.tiles.size() - 4] = 128;
  }

  SmallVector<size_t> computeSgLayout(Target &tg) override {
    return {128 / getSgSize(tg), 1};
  }
};
} // namespace
