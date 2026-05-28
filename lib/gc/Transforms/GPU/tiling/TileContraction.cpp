#include "TilingUtils.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TILECONTRACTION
#define GEN_PASS_DEF_TILECONTRACTION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
struct TileContraction final
    : TilingPass<gc::impl::TileContractionBase<TileContraction>> {

  bool isSupportedOp(TilingInterface ti) override { return isMatmulOp(ti); }

  virtual std::tuple<SmallVector<unsigned>, SmallVector<unsigned>,
                     SmallVector<unsigned>, unsigned, unsigned>
  getSupportedBlockSizes(Target &tg, bool reduction, size_t width,
                         size_t height) override {
    auto ua = tg.devAttrs.getUarch();
    auto instr =
        dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(ua->getInstruction(
            xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
    assert(instr);
    auto elType =
        cast<ShapedType>(tg.op.getOperation()->getOperand(0).getType())
            .getElementType();

    if (reduction) {
      auto supportedK =
          llvm::map_to_vector(instr->getSupportedK(elType), [](uint32_t v) {
            return static_cast<unsigned>(v);
          });
      return std::make_tuple(supportedK, SmallVector<unsigned>{1},
                             SmallVector<unsigned>{1}, 1, 1);
    }

    auto [widths, heights, counts, sgMul, wgMul] =
        TilingPass::getSupportedBlockSizes(tg, reduction, width, height);
    auto supportedN = instr->getSupportedN(elType);
    auto supportedM = instr->getSupportedM(elType);
    widths = llvm::filter_to_vector(
        widths, [&](unsigned c) { return llvm::is_contained(supportedN, c); });
    heights = llvm::filter_to_vector(
        heights, [&](unsigned c) { return llvm::is_contained(supportedM, c); });
    counts = {1};
    int sgSize = getSgSize(tg);
    return std::make_tuple(widths, heights, counts, sgSize,
                           getWgSize(tg) / sgSize);
  }
};
} // namespace