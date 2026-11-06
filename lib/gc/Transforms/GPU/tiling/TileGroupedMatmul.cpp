#include "TilingUtils.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TILEGROUPEDMATMUL
#define GEN_PASS_DEF_TILEGROUPEDMATMUL
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

size_t getNarrowInputLoadWidth(
    Value value, unsigned dim,
    const xegpu::uArch::Subgroup2DBlockLoadInstruction *load) {
  auto type = cast<RankedTensorType>(value.getType());
  Type element = type.getElementType();
  if (type.getRank() < 2 || !element.isIntOrFloat() ||
      element.getIntOrFloatBitWidth() > 8)
    return 1;
  bool contiguous = llvm::all_of(type.getShape().drop_front(dim + 1),
                                 [](int64_t size) { return size == 1; });
  if (auto tensor = value.getDefiningOp<bufferization::ToTensorOp>()) {
    auto memref = dyn_cast<MemRefType>(tensor.getBuffer().getType());
    SmallVector<int64_t> strides;
    int64_t offset;
    contiguous = memref &&
                 succeeded(memref.getStridesAndOffset(strides, offset)) &&
                 strides[dim] == 1;
  }
  if (!contiguous) return 1;
  auto shapes = load->getBlockWidthHeightCount(element, false, false);
  if (!shapes || std::get<0>(*shapes).empty()) return 1;
  return static_cast<size_t>(*llvm::min_element(std::get<0>(*shapes)));
}

// Follow the weights' N dimension to the narrow inputs loaded along that
// dimension. An i4 input contiguous along K does not constrain the N split.
size_t getWeightLoadWidth(Value weights, const xegpu::uArch::uArch *uarch) {
  auto *load = dyn_cast<xegpu::uArch::Subgroup2DBlockLoadInstruction>(
      uarch->getInstruction(
          xegpu::uArch::InstructionKind::Subgroup2DBlockLoad));
  assert(load);
  SmallVector<std::pair<Value, unsigned>> worklist{{weights, 1}};
  llvm::SmallDenseSet<std::pair<Value, unsigned>, 16> visited;
  size_t width = 1;
  while (!worklist.empty()) {
    auto [value, dim] = worklist.pop_back_val();
    if (!visited.insert({value, dim}).second) continue;
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type) continue;
    Operation *def = value.getDefiningOp();
    if (auto slice = dyn_cast_if_present<tensor::ExtractSliceOp>(def)) {
      auto dropped = slice.getDroppedDims();
      unsigned resultDim = 0;
      for (unsigned sourceDim = 0; sourceDim < dropped.size(); ++sourceDim) {
        if (dropped[sourceDim]) continue;
        if (resultDim++ == dim) {
          if (getConstantIntValue(slice.getMixedStrides()[sourceDim]) == 1)
            worklist.emplace_back(slice.getSource(), sourceDim);
          break;
        }
      }
      continue;
    }
    if (auto cast = dyn_cast_if_present<tensor::CastOp>(def)) {
      worklist.emplace_back(cast.getSource(), dim);
      continue;
    }
    if (auto collapse = dyn_cast_if_present<tensor::CollapseShapeOp>(def)) {
      for (auto sourceDim : collapse.getReassociationIndices()[dim])
        worklist.emplace_back(collapse.getSrc(), sourceDim);
      continue;
    }
    if (auto expand = dyn_cast_if_present<tensor::ExpandShapeOp>(def)) {
      for (auto [sourceDim, group] :
           llvm::enumerate(expand.getReassociationIndices())) {
        auto pos = llvm::find(group, dim);
        if (pos != group.end() &&
            llvm::all_of(llvm::make_range(std::next(pos), group.end()),
                         [&](int64_t d) { return type.getDimSize(d) == 1; }))
          worklist.emplace_back(expand.getSrc(), sourceDim);
      }
      continue;
    }
    if (auto producer = dyn_cast_if_present<linalg::LinalgOp>(def);
        producer && isParallel(def)) {
      auto result = cast<OpResult>(value);
      auto resultMap = getResultIndexingMap(def, result.getResultNumber());
      auto iterDim = dyn_cast<AffineDimExpr>(resultMap.getResult(dim));
      if (!iterDim) continue;
      for (OpOperand *input : producer.getDpsInputOperands()) {
        auto map = producer.getMatchingIndexingMap(input);
        for (auto [inputDim, expr] : llvm::enumerate(map.getResults()))
          if (expr == iterDim) worklist.emplace_back(input->get(), inputDim);
      }
      continue;
    }

    width = std::lcm(width, getNarrowInputLoadWidth(value, dim, load));
  }
  return width;
}

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
    size_t width = getWeightLoadWidth(
        cast<GMM>(tg.op.getOperation()).getWeights(), tg.devAttrs.getUarch());
    // Preserve the generic tile where possible, but never split a narrow
    // block load across N subgroups. Partial WG tiles are padded later.
    tg.sgTiles[GMM::kNDim] = std::lcm(tg.sgTiles[GMM::kNDim], width);
    tg.tiles[GMM::kNDim] =
        std::lcm(tg.tiles[GMM::kNDim], tg.sgTiles[GMM::kNDim]);
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

  SmallVector<size_t> computeSgLayout(Target &tg) override {
    auto layout = TilingPass::computeSgLayout(tg);
    if (isa<linalgx::GroupedMatmulOp>(tg.op.getOperation()) &&
        tg.tiles[linalgx::GroupedMatmulOp::kNDim] > 1)
      layout.resize(3, 1);
    return layout;
  }
};

} // namespace
