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

  // Fewer query rows per subgroup for large head dims: the S/P tiles and the
  // output accumulator stay in registers across the k-loop, so their size per
  // subgroup must not grow with head_dim. The thresholds assume the Q tile is
  // re-loaded inside the loop (SetAttentionLayouts q-in-loop).
  SmallVector<size_t> computeSgLayout(Target &tg) override {
    auto n = tg.sizes.size();
    size_t headDim = std::max(tg.sizes[n - 3], tg.sizes[n - 1]);
    size_t rows = headDim <= 128 ? 16 : headDim <= 256 ? 8 : 4;
    return {128 / rows, 1};
  }
};
} // namespace
