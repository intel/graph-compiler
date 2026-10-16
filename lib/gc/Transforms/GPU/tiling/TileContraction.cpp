#include "TilingUtils.h"
#include "mlir/Support/LLVM.h"
#include <algorithm>
#include <cmath>
#include <functional>

namespace mlir::gc {
#define GEN_PASS_DECL_TILECONTRACTION
#define GEN_PASS_DEF_TILECONTRACTION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
struct TileContraction final
    : TilingPass<::mlir::gc::impl::TileContractionBase<TileContraction>> {

  bool isSupportedOp(TilingInterface ti) override { return isMatmulOp(ti); }

  BlockSizes getSupportedBlockSizes(Target &tg, bool reduction,
                                    const DimConstraint &cw,
                                    const DimConstraint &ch) override {
    BlockSizes bs;
    if (reduction) {
      auto tiles = tg.kernelAttrs.getTiles().value();
      bs.widths = {static_cast<unsigned>(tiles.back())};
      bs.heights = {1};
      bs.counts = {1};
      return bs;
    }

    auto ua = tg.devAttrs.getUarch();
    auto instr =
        dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(ua->getInstruction(
            xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
    assert(instr);
    auto elType =
        cast<ShapedType>(tg.op.getOperation()->getOperand(0).getType())
            .getElementType();
    bs = TilingPass::getSupportedBlockSizes(tg, false, cw, ch);
    auto supportedM = instr->getSupportedM(elType);
    auto supportedN = instr->getSupportedN(elType);
    auto supportedK =
        llvm::map_to_vector(instr->getSupportedK(elType), [](uint32_t v) {
          return static_cast<unsigned>(v);
        });
    llvm::sort(supportedK, std::greater<unsigned>());

    bs.widths = llvm::filter_to_vector(bs.widths, [&](unsigned c) {
      return llvm::is_contained(supportedN, c);
    });
    bs.heights = llvm::filter_to_vector(bs.heights, [&](unsigned c) {
      return llvm::is_contained(supportedM, c);
    });
    bs.counts = {1};
    if (bs.widths.empty() || bs.heights.empty() || supportedK.empty()) {
      tg.op->emitWarning("No supported DPAS block size found");
      bs.wgMulW = bs.wgMulH = 1;
      return bs;
    }

    size_t kSize = tg.sizes.back();
    unsigned base = std::sqrt(getSgSize(tg));
    unsigned maxMul = getWgSize(tg) / getSgSize(tg);
    for (auto n : bs.widths)
      for (auto mulN = base; mulN; mulN /= 2) {
        if (!cw.satisfies(n * mulN)) continue;
        for (auto m : bs.heights)
          for (auto mulM = base; mulM; mulM /= 2) {
            if (mulM * mulN > maxMul || !ch.satisfies(m * mulM)) continue;
            for (auto k : supportedK)
              for (auto mulK = mulN; mulK; mulK /= 2)
                if (kSize % (k * mulK) == 0) {
                  tg.kernelAttrs.setTiles(
                      {m * mulM, n * mulN, size_t{k} * mulK});
                  bs.widths = {n};
                  bs.heights = {m};
                  bs.wgMulW = mulN;
                  bs.wgMulH = mulM;
                  return bs;
                }
          }
      }

    // Nothing satisfies the constraints, fall back to a single DPAS block.
    tg.kernelAttrs.setTiles(
        {bs.heights.front(), bs.widths.front(), supportedK.front()});
    bs.widths = {bs.widths.front()};
    bs.heights = {bs.heights.front()};
    bs.wgMulW = bs.wgMulH = 1;
    return bs;
  }

  SmallVector<size_t> computeSgLayout(Target &tg) override {
    auto layout = TilingPass::computeSgLayout(tg);
    auto count = std::accumulate(layout.begin(), layout.end(), size_t{1},
                                 std::multiplies<>());
    size_t m = 1, n = 1;
    for (size_t i = std::max<size_t>(1, count / getSgSize(tg)); i > 1; i /= 2)
      (m <= n ? m : n) *= 2;
    return {m, n};
  }
};
} // namespace