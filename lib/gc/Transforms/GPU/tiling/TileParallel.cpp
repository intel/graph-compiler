#include "TilingUtils.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TILEPARALLEL
#define GEN_PASS_DEF_TILEPARALLEL
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
struct TileParallel final
    : TilingPass<gc::impl::TileParallelBase<TileParallel>> {
  bool isSupportedOp(TilingInterface ti) override { return isParallel(ti); }
};
} // namespace