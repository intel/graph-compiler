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

  static const xegpu::uArch::SubgroupMatrixMultiplyAcc *getInstr(Target &tg) {
    auto ua = tg.devAttrs.getUarch();
    auto instr =
        dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(ua->getInstruction(
            xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
    assert(instr);
    return instr;
  }

  void computeWgTiles(Target &tg) override {
    if (auto tiles = tg.kernelAttrs.getTiles(); tiles && tiles->size() == 3) {
      tg.tiles[0] = (*tiles)[0];
      tg.tiles[1] = (*tiles)[1];
      return;
    }

    auto instr = getInstr(tg);
    auto elType =
        cast<ShapedType>(tg.op.getOperation()->getOperand(0).getType())
            .getElementType();
    auto supportedM = instr->getSupportedM(elType);
    auto supportedN = instr->getSupportedN(elType);
    auto closestM = findClosestDiv(supportedM, tg.tiles[0]);
    auto closestN = findClosestDiv(supportedN, tg.tiles[1]);
    auto mul =
        std::max(static_cast<size_t>(1),
                 static_cast<size_t>(std::sqrt(getWgSize(tg) / getSgSize(tg))));
    do {
      tg.tiles[0] = closestM * mul;
      tg.tiles[1] = closestN * mul;
    } while ((tg.tiles[0] >= tg.sizes[0] || tg.tiles[1] >= tg.sizes[1]) &&
             (mul = mul / 2));
  }

  void computeSgTiles(Target &tg) override {
    if (tg.mode == Mode::Parallel) {
      tg.tiles = {0, 0};
      return;
    }

    size_t kTile;
    if (auto tiles = tg.kernelAttrs.getTiles(); tiles && tiles->size() == 3) {
      kTile = (*tiles)[2];
    } else {
      auto instr = getInstr(tg);
      auto elType =
          cast<ShapedType>(tg.op.getOperation()->getOperand(0).getType())
              .getElementType();
      auto supportedK = instr->getSupportedK(elType);
      kTile = findClosestDiv(supportedK, tg.tiles[0]);
    }

    tg.tiles[tg.mode == Mode::Reduction ? 0 : 2] = kTile;
  }

  void computeThreads(Target &tg) override {
    tg.tiles[0] = tg.sizes[0] / tg.tiles[0];
    tg.tiles[1] = tg.sizes[1] / tg.tiles[1];
    auto sgSize = getSgSize(tg);
    auto total = tg.tiles[0] * tg.tiles[1] / sgSize;
    if (total < sgSize) {
      tg.tiles = {sgSize, 1, 1};
    } else {
      adjustTiles(total, tg.tiles, false);
      tg.tiles.emplace_back(1);
    }
  }
};
} // namespace