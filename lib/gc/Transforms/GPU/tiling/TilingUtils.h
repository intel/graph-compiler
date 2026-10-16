#ifndef TILING_UTILS_H
#define TILING_UTILS_H
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "gc/Utils/Log.h"
#include "gc/Utils/Misc.h"
#include "gc/Utils/Transform.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include <cmath>
#include <numeric>
#include <utility>

using namespace mlir;
using namespace mlir::gc;
using namespace mlir::scf;

// GC_ATTR_LEVEL and GC_ATTR_WG_TILE_SIZES are defined in gc/Utils/Transform.h.
constexpr char GC_ATTR_NUM_KERNELS[] = "gc.num_kernels";

// Indexing map for an op's result tensor (via its DPS init operand).
inline AffineMap getResultIndexingMap(Operation *op, unsigned resultNum) {
  auto dst = dyn_cast<DestinationStyleOpInterface>(op);
  auto idx = dyn_cast<IndexingMapOpInterface>(op);
  if (!dst || !idx) return {};
  return idx.getMatchingIndexingMap(dst.getDpsInitOperand(resultNum));
}

// Remap `srcTiles` (indexed by `srcMap`'s iter dims) into a `dstNumDims`-sized
// tile array indexed by `dstMap`'s iter dims, via the shared tensor dims.
// Both maps must describe the same tensor and be projected permutations.
inline SmallVector<int64_t> remapTiles(ArrayRef<size_t> srcTiles,
                                       AffineMap srcMap, AffineMap dstMap,
                                       unsigned dstNumDims) {
  SmallVector<int64_t> out(dstNumDims, 0);
  if (!srcMap || !dstMap || srcMap.getNumResults() != dstMap.getNumResults())
    return out;
  for (unsigned r = 0, n = srcMap.getNumResults(); r < n; ++r) {
    auto sd = dyn_cast<AffineDimExpr>(srcMap.getResult(r));
    auto dd = dyn_cast<AffineDimExpr>(dstMap.getResult(r));
    if (!sd || !dd) continue;
    if (sd.getPosition() < srcTiles.size() && dd.getPosition() < dstNumDims)
      out[dd.getPosition()] = static_cast<int64_t>(srcTiles[sd.getPosition()]);
  }
  return out;
}

// Set/merge `gc.tiling.wg_tile_sizes` on `op`.
// Is used to "merge" wg (parallel-dim) tiles and sg (reduction-dim) tiles.
inline void mergeWgTileSizesAttr(Operation *op, ArrayRef<int64_t> tiles) {
  if (tiles.empty()) return;
  SmallVector<int64_t> merged(tiles.begin(), tiles.end());
  if (auto existing =
          op->getAttrOfType<DenseI64ArrayAttr>(GC_ATTR_WG_TILE_SIZES)) {
    auto prev = existing.asArrayRef();
    for (size_t i = 0, n = std::min(merged.size(), prev.size()); i < n; ++i)
      if (merged[i] == 0) merged[i] = prev[i];
  }
  op->setDiscardableAttr(GC_ATTR_WG_TILE_SIZES,
                         DenseI64ArrayAttr::get(op->getContext(), merged));
}

// Whether `tiled` iterates over the same domain as `orig`, i.e. tiles indexed
// by `orig`'s loops can be attached to `tiled` as is.
inline bool hasSameIterationDomain(Operation *orig, Operation *tiled) {
  auto origTi = dyn_cast<TilingInterface>(orig);
  auto tiledTi = dyn_cast<TilingInterface>(tiled);
  if (!origTi || !tiledTi ||
      origTi.getLoopIteratorTypes() != tiledTi.getLoopIteratorTypes())
    return false;
  auto origIdx = dyn_cast<IndexingMapOpInterface>(orig);
  auto tiledIdx = dyn_cast<IndexingMapOpInterface>(tiled);
  // Without indexing maps there is nothing more to compare.
  if (!origIdx || !tiledIdx) return true;
  return origIdx.getIndexingMapsArray() == tiledIdx.getIndexingMapsArray();
}

// Tag the tiled consumer and fused producers from an SCFTileAndFuseResult
// with `gc.tiling.wg_tile_sizes` derived from `tiles` (indexed by the original
// consumer's iteration domain).
// Example of input args (linalg.fill + linalg.matmul case):
//    origConsumer: untiled linalg.matmul
//    tiles: [256, 512, 16]
//    result.tiledAndFusedOps: [tiled_matmul, tiled_fill]
//    result.fusedProducers:   [orig_fill]
inline void tagTileAndFuseResult(Operation *origConsumer,
                                 ArrayRef<size_t> tiles,
                                 const scf::SCFTileAndFuseResult &result) {
  if (result.tiledAndFusedOps.empty()) return;
  // TODO: currently we have to cast 'tiles' size_t -> int64_t;
  // we should rework our tiling logic to always use int64_t to
  // avoid casts on the C++/mlir boundary.
  auto consumerTiles = llvm::map_to_vector(
      tiles, [](size_t v) { return static_cast<int64_t>(v); });
  auto it = result.tiledAndFusedOps.begin();
  // Set tile-size attribute for the tiled op itself (e.g. linalg.matmul),
  // it's always the first element in the tiledAndFusedOps list. The tiled op
  // may have a different iteration domain (e.g. linalgx.grouped_matmul is
  // tiled into linalg.matmul), in which case the tiles are remapped through
  // the result indexing maps.
  Operation *tiled = *it++;
  auto tiledTi = dyn_cast<TilingInterface>(tiled);
  if (tiledTi && !hasSameIterationDomain(origConsumer, tiled)) {
    assert(llvm::none_of(
               llvm::zip(
                   cast<TilingInterface>(origConsumer).getLoopIteratorTypes(),
                   tiles),
               [](auto p) {
                 // iterate over pairs of (iterator type, tile size) and check
                 // if any reduction dim has a non-zero tile size. It should
                 // never happen: an op whose tiled op has a different iteration
                 // domain (e.g. linalgx.grouped_matmul -> linalg.matmul) is
                 // only tiled at the WG level, and computeWgTiles() tiles
                 // parallel dims only. The reduction dims are tiled at the SG
                 // level on the resulting op, which keeps its iteration domain
                 // and so never gets here.
                 return std::get<0>(p) == utils::IteratorType::reduction &&
                        std::get<1>(p) != 0;
               }) &&
           "can't remap reduction tiles through the result indexing map");
    mergeWgTileSizesAttr(
        tiled, remapTiles(tiles, getResultIndexingMap(origConsumer, 0),
                          getResultIndexingMap(tiled, 0),
                          tiledTi.getLoopIteratorTypes().size()));
  } else {
    mergeWgTileSizesAttr(tiled, consumerTiles);
  }

  auto consumerIdx = dyn_cast<IndexingMapOpInterface>(origConsumer);
  // If the consumer doesn't have indexing maps, we can't remap the tiles to
  // the fused producers.
  if (!consumerIdx) return;

  // Iterate over the tiled producers and set their tile-size attributes.
  // Example:
  // result.tiledAndFusedOps: [tiled_matmul, tiled_fill]
  //                                         ^--*it
  // result.fusedProducers:   [orig_fill]
  //                          ^--*fp
  auto fp = result.fusedProducers.begin();
  for (;
       it != result.tiledAndFusedOps.end() && fp != result.fusedProducers.end();
       ++it, ++fp) {
    auto *tiledProd = *it;
    auto *origProd = *fp;
    auto prodTi = dyn_cast<TilingInterface>(tiledProd);
    if (!prodTi) continue;
    AffineMap consumerMap, prodMap;
    for (auto &operand : origConsumer->getOpOperands()) {
      auto opRes = dyn_cast<OpResult>(operand.get());
      if (!opRes || opRes.getOwner() != origProd) continue;
      consumerMap = consumerIdx.getMatchingIndexingMap(&operand);
      prodMap = getResultIndexingMap(origProd, opRes.getResultNumber());
      break;
    }
    if (!consumerMap || !prodMap) {
      tiledProd->emitWarning()
          << "unable to find matching indexing maps for remapping tiles from "
             "consumer to producer; skipping tile-size attribute propagation";
      continue;
    }

    mergeWgTileSizesAttr(tiledProd,
                         remapTiles(tiles, consumerMap, prodMap,
                                    prodTi.getLoopIteratorTypes().size()));
  }
}

inline bool hasIterator(Operation *op, utils::IteratorType type) {
  auto ti = dyn_cast<TilingInterface>(op);
  return ti &&
         llvm::any_of(ti.getLoopIteratorTypes(),
                      [type](utils::IteratorType t) { return t == type; });
}

inline bool allIterators(Operation *op, utils::IteratorType type) {
  auto ti = dyn_cast<TilingInterface>(op);
  return ti &&
         llvm::all_of(ti.getLoopIteratorTypes(),
                      [type](utils::IteratorType t) { return t == type; });
}

inline bool isParallel(Operation *op) {
  return allIterators(op, utils::IteratorType::parallel);
}

inline bool isReduction(Operation *op) {
  return allIterators(op, utils::IteratorType::reduction);
}

// Tile size constraints of a single tiled dim, collected from the fusible
// subgraph. The two rules are opposite in direction, thus can't be expressed
// as a single upper bound:
//  * A reshape group can only be fused if the tile covers whole groups or
//    divides one, i.e. the tile must divide the group's inner product.
//  * A pack/unpack inner tile can only be fused if the tile covers it
//    completely, i.e. the tile must be a multiple of the inner tile.
struct DimConstraint {
  size_t divisorOf = 0;  // tile must divide this, 0 = unconstrained
  size_t multipleOf = 1; // tile must be a multiple of this

  bool divides(size_t tile) const {
    return tile && (!divisorOf || divisorOf % tile == 0);
  }

  bool satisfies(size_t tile) const {
    return divides(tile) && tile % multipleOf == 0;
  }

  void narrow(size_t size) {
    if (size) divisorOf = divisorOf ? std::gcd(divisorOf, size) : size;
  }

  void align(size_t size) {
    if (size) multipleOf = std::lcm(multipleOf, size);
  }
};

// Block sizes and tile multipliers supported by the target for a tiled op.
struct BlockSizes {
  SmallVector<unsigned> widths, heights, counts;
  unsigned sgMul = 1;  // SG-tile multiplier
  unsigned wgMulW = 1; // WG-tile multiplier of the width dim
  unsigned wgMulH = 1; // WG-tile multiplier of the height dim
};

enum class Level : char { WG, SG };
struct Target {
private:
  SmallString<64> kernelName;

public:
  func::FuncOp fn;
  OpRewriter rw;
  DevAttrs devAttrs;
  KernelAttrs kernelAttrs;
  TilingInterface op;
  Level level;
  SmallVector<size_t> sizes{};
  SmallVector<size_t> tiles{};
  SmallVector<size_t> sgTiles{};
  SmallVector<bool> reductions{};

  Target(func::FuncOp fn)
      : kernelName(fn.getName()), fn(fn), rw(fn), devAttrs(fn),
        kernelAttrs(fn, kernelName) {
    kernelName.append("_kernel");
    kernelAttrs = KernelAttrs(fn, kernelName);
    rw.setInsertionPointToStart(&fn.getBody().front());
  }

  bool set(TilingInterface &op, Level level) {
    this->op = op;
    this->level = level;
    mark(op.getOperation());

    if (level == Level::WG) {
      unsigned numKernels = 1;
      if (auto numKernelsAttr = fn->getDiscardableAttr(GC_ATTR_NUM_KERNELS)) {
        numKernels = getAttrValue<unsigned>(numKernelsAttr);
        char buffer[8];
        snprintf(buffer, sizeof(buffer), "%u", numKernels);
        kernelAttrs.save();
        kernelName.resize(fn.getName().size() + 7);
        kernelName.append("_");
        kernelName.append(buffer);
        kernelAttrs = KernelAttrs(fn, kernelName);
        ++numKernels;
      }
      if (!kernelAttrs.getSgSize())
        kernelAttrs.setSgSize(devAttrs.getUarch()->getSubgroupSize());
      fn->setDiscardableAttr(
          GC_ATTR_NUM_KERNELS,
          createAttr<unsigned>(fn->getContext(), numKernels));
    }

    sizes.resize(0);
    tiles.resize(0);
    sgTiles.resize(0);
    reductions.resize(0);

    // Set the insertion point before the op so that helper ops emitted for
    // dynamic dims (e.g. tensor.dim) dominate their uses.
    OpBuilder::InsertionGuard guard(rw);
    rw.setInsertionPoint(op);
    for (auto [i, t, r] : llvm::enumerate(op.getLoopIteratorTypes(),
                                          op.getIterationDomain(rw))) {
      tiles.emplace_back(0);
      sgTiles.emplace_back(1);
      reductions.emplace_back(t == utils::IteratorType::reduction);
      // Dynamic dims use 0 as a sentinel: computeTiles treats it as "any
      // size" (0 % block == 0) and selects the largest supported block.
      sizes.emplace_back(getConstantIntValue(r.size).value_or(0));
    }
    return true;
  }

  SmallVector<size_t> getSizes(bool reduction) {
    SmallVector<size_t> filtered;
    for (size_t i = 0, n = sizes.size(); i < n; ++i) {
      if (reductions[i] == reduction) {
        filtered.push_back(sizes[i]);
      }
    }
    return filtered;
  }

  void setTiles(SmallVector<size_t> &wgTiles, SmallVector<size_t> &sgTiles,
                bool reduction) {
    for (size_t i = 0, j = 0, n = this->tiles.size(); i < n; ++i) {
      if (reductions[i] == reduction) {
        this->tiles[i] = wgTiles[j];
        this->sgTiles[i] = sgTiles[j++];
      }
    }
  }

  bool hasTiles() {
    return llvm::any_of(tiles, [](size_t t) { return t != 0; });
  }

  bool hasReductions() {
    return llvm::any_of(reductions, [](bool r) { return r; });
  }

  void mark(Operation *op) {
    if (level == Level::WG && isa<ForallOp>(op)) {
      op->setDiscardableAttr(GC_ATTR_KERNEL_NAME,
                             createAttr(op->getContext(), kernelName));
    } else {
      op->setDiscardableAttr(GC_ATTR_LEVEL,
                             createAttr(op->getContext(), level));
    }
  }
};

template <typename BaseT> class TilingPass : public BaseT {

  void runOnOperation() override {
    auto fn = this->getOperation();
    if (!fn.isExternal()) {
      Target tg(fn);
      tileWg(tg);
    }
  }

protected:
  virtual bool isSupportedOp(TilingInterface ti) = 0;

  virtual bool tileWg(Target &tg) {
    struct Filter {
      bool operator()(Operation &op) const {
        return !op.hasAttr(GC_ATTR_LEVEL) &&
               !(isa<ForallOp>(op) && op.hasAttr(GC_ATTR_KERNEL_NAME));
      }
    };
    std::function<bool(TilingInterface)> predicate = [&](TilingInterface op) {
      return isSupportedOp(op);
    };

    while (auto ti = findLast<TilingInterface, Filter>(tg.fn, predicate)) {
      if (!tg.set(ti, Level::WG)) {
        return false;
      }
      computeWgTiles(tg);
      tg.kernelAttrs.setThreads(computeThreads(tg));
      if (auto loop = apply(tg)) {
        if (!tileSg(tg, loop)) {
          return false;
        }
      } else {
        return false;
      }
    }
    return true;
  }

  virtual bool tileSg(Target &tg, LoopLikeOpInterface wgLoop) {
    struct Filter {
      bool operator()(Operation &op) const {
        return getDiscardableAttr(&op, GC_ATTR_LEVEL, Level::WG) != Level::SG;
      }
    };
    std::function<bool(TilingInterface)> predicate = [&](TilingInterface op) {
      return isSupportedOp(op);
    };
    while (auto ti = findLast<TilingInterface, Filter>(wgLoop, predicate)) {
      if (!tg.set(ti, Level::SG)) {
        return false;
      }
      computeSgTiles(tg);
      if (auto loop = apply(tg); !loop && tg.hasTiles()) {
        return false;
      }
    }
    return true;
  }

  virtual void computeWgTiles(Target &tg) { computeTiles(tg, false); }

  virtual void computeSgTiles(Target &tg) { computeTiles(tg, true); }

  // Tile the innermost 2 non-unit dims and set all leading dims to 1.
  virtual void computeTiles(Target &tg, bool reduction) {
    auto wgTiles = tg.getSizes(reduction);
    if (wgTiles.empty()) return;
    SmallVector<size_t> sgTiles(wgTiles.size(), 1);

    // Skip the inner most unit dims because they can't be mapped to 2D block
    // load/store instructions
    int wDim = wgTiles.size();
    while (wDim-- && wgTiles[wDim] == 1);
    if (wDim == -1) { // all dims are unit
      tg.setTiles(sgTiles, sgTiles, reduction);
      return;
    }

    bool unit = wDim == 0;
    size_t hDim = unit ? wDim : wDim - 1;
    DimConstraint cw, ch;
    cw.narrow(wgTiles[wDim]);
    if (!unit) ch.narrow(wgTiles[hDim]);
    for (auto &t : wgTiles) t = 1;
    collectTileConstraints(tg, cw, ch);
    // TODO: parameterize sgMul and wgMul in kernel attributes so they can
    // be used for auto tuning.
    auto bs = getSupportedBlockSizes(tg, reduction, cw, ch);
    if (unit) bs.heights = {1};
    if (reduction) bs.sgMul = bs.wgMulW = bs.wgMulH = 1;

    auto maxMul = reduction ? 1 : getWgSize(tg) / getSgSize(tg);

    // The second round drops the `multipleOf` constraints: if no tile can
    // satisfy them, the reshape just won't fuse, but tiling must still happen.
    for (bool strict : {true, false})
      for (auto w : bs.widths)
        for (auto h : bs.heights)
          for (auto c : bs.counts)
            for (auto sm = bs.sgMul; sm; sm /= 2)
              for (auto wmw = bs.wgMulW; wmw; wmw /= 2)
                for (auto wmh = unit ? 1u : bs.wgMulH; wmh; wmh /= 2) {
                  if ((unit ? wmw : wmw * wmh) > maxMul) continue;
                  size_t sgw = w * c * sm, sgh = h * sm;
                  size_t wgw = sgw * wmw, wgh = sgh * wmh;
                  if (!(strict ? cw.satisfies(wgw) : cw.divides(wgw))) continue;
                  if (!unit && !(strict ? ch.satisfies(wgh) : ch.divides(wgh)))
                    continue;
                  wgTiles[wDim] = wgw;
                  sgTiles[wDim] = sgw;
                  if (!unit) {
                    wgTiles[hDim] = wgh;
                    sgTiles[hDim] = sgh;
                  }
                  tg.setTiles(wgTiles, sgTiles, reduction);
                  return;
                }

    wgTiles[wDim] = 1;
    wgTiles[hDim] = 1;
    sgTiles[wDim] = 1;
    sgTiles[hDim] = 1;
    tg.setTiles(wgTiles, sgTiles, reduction);
  }

  // If there is a reshape in the fusible subgraph, that reshapes one or both of
  // the last 2 dims (height x width), constrain the width and height tiles, so
  // that the reshape could be fused.
  virtual void collectTileConstraints(Target &tg, DimConstraint &width,
                                      DimConstraint &height) {
    auto canFuse = [&](Operation *user) {
      return user->getBlock() == tg.op->getBlock() && isParallel(user);
    };
    // BFS over fusible consumers; for each reshape op, check if it touches
    // the last two dims and clamp width/height to the innerProd of that group.
    SmallVector<Operation *> stack{tg.op.getOperation()};
    llvm::SmallSet<Operation *, 16> visited;
    while (!stack.empty()) {
      auto *cur = stack.pop_back_val();
      if (!visited.insert(cur).second) continue;
      for (auto res : cur->getResults()) {
        for (auto *user : res.getUsers()) {
          // For expand_shape: inner dims of a multi-dim group give innerProd.
          if (auto expand = dyn_cast<tensor::ExpandShapeOp>(user)) {
            auto dstType = expand.getResultType();
            auto rank = expand.getSrcType().getRank();
            for (auto [gi, group] :
                 llvm::enumerate(expand.getReassociationIndices())) {
              if (group.size() <= 1) continue;
              auto srcDim = expand.getCorrespondingSourceDim(group[0]);
              int64_t ip = 1;
              for (size_t i = 1; i < group.size(); ++i)
                ip *= dstType.getDimSize(group[i]);
              if (srcDim == rank - 1) width.narrow(ip);
              else if (srcDim == rank - 2) height.narrow(ip);
            }
            stack.push_back(user);
          } else if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(user)) {
            auto srcType = collapse.getSrcType();
            size_t rank = collapse.getResultType().getRank();
            for (auto [dstDim, group] :
                 llvm::enumerate(collapse.getReassociationIndices())) {
              if (group.size() <= 1) continue;
              int64_t ip = 1;
              for (auto d : group) ip *= srcType.getDimSize(d);
              if (dstDim == rank - 1) width.narrow(ip);
              else if (dstDim == rank - 2) height.narrow(ip);
            }
            stack.push_back(user);
          } else if (auto pack = dyn_cast<linalg::PackOp>(user)) {
            // The tile must cover whole inner tiles. Narrow it to a single one
            // as well - a bigger tile would fuse, but a multiple of the inner
            // tile is what the consumers of the packed value expect.
            auto tiles = pack.getMixedTiles();
            auto n = tiles.size();
            if (n >= 1)
              if (auto v = getConstantIntValue(tiles[n - 1])) {
                width.align(*v);
                width.narrow(*v);
              }
            if (n >= 2)
              if (auto v = getConstantIntValue(tiles[n - 2])) {
                height.align(*v);
                height.narrow(*v);
              }
            stack.push_back(user);
          } else if (canFuse(user)) {
            stack.push_back(user);
          }
        }
      }
    }
  }

  // Get the supported block sizes, that can be used for tiling of the dims
  // described by the width and height constraints.
  virtual BlockSizes getSupportedBlockSizes(Target &tg, bool reduction,
                                            const DimConstraint &cw,
                                            const DimConstraint &ch) {
    Type elTy;
    // Get the operand with maximum width
    for (auto o : tg.op.getOperation()->getOperands()) {
      if (auto t = dyn_cast<ShapedType>(o.getType())) {
        auto et = t.getElementType();
        if (!et.isIntOrFloat()) continue;
        if (elTy) {
          if (et.getIntOrFloatBitWidth() > elTy.getIntOrFloatBitWidth()) {
            elTy = et;
          }
        } else {
          elTy = et;
        }
      }
    }

    if (!elTy) {
      tg.op->emitError() << "At least one operand must be of ShapedType";
      return {};
    }

    // The block sizes computation is based on the assumption, that the kernel
    // will have at least 2D block load/store instructions.
    auto ua = tg.devAttrs.getUarch();
    auto loadIns = dyn_cast<xegpu::uArch::Subgroup2DBlockLoadInstruction>(
        ua->getInstruction(xegpu::uArch::InstructionKind::Subgroup2DBlockLoad));
    auto storeIns = dyn_cast<xegpu::uArch::Subgroup2DBlockStoreInstruction>(
        ua->getInstruction(
            xegpu::uArch::InstructionKind::Subgroup2DBlockStore));
    assert(loadIns && storeIns);
    auto defaults = std::make_tuple(SmallVector<int>{1}, SmallVector<int>{1},
                                    SmallVector<int>{1});
    auto loadSizes = loadIns->getBlockWidthHeightCount(elTy, false, false)
                         .value_or(defaults);
    auto storeSizes =
        storeIns->getBlockWidthHeightCount(elTy).value_or(defaults);

    // Get only the common sizes from both instructions and filter out those
    // that do not divide the tile sizes.
    BlockSizes bs;
    for (unsigned w : std::get<0>(loadSizes))
      if (cw.divides(w) && llvm::is_contained(std::get<0>(storeSizes), w))
        bs.widths.push_back(w);
    for (unsigned h : std::get<1>(loadSizes))
      if (ch.divides(h) && llvm::is_contained(std::get<1>(storeSizes), h))
        bs.heights.push_back(h);
    for (unsigned c : std::get<2>(loadSizes))
      if (llvm::is_contained(std::get<2>(storeSizes), c))
        bs.counts.push_back(c);
    for (auto l : {&bs.widths, &bs.heights, &bs.counts})
      if (!llvm::is_contained(*l, 1)) l->push_back(1);

    llvm::sort(bs.widths, std::greater<unsigned>());
    llvm::sort(bs.heights, std::greater<unsigned>());
    llvm::sort(bs.counts, std::greater<unsigned>());
    bs.wgMulW = bs.wgMulH = std::sqrt(getSgSize(tg));
    return bs;
  }

  // Number of subgroups the wg tile is distributed over, per non-reduction dim.
  virtual SmallVector<size_t> computeSgLayout(Target &tg) {
    SmallVector<size_t> layout;
    size_t nonUnit = 0;
    for (auto [wg, sg, s, r] :
         llvm::zip(tg.tiles, tg.sgTiles, tg.sizes, tg.reductions)) {
      if (!r) {
        layout.push_back(wg / sg);
        if (s != 1) nonUnit = layout.size();
      }
    }
    // The trailing unit dims are removed by the DropUnitDims pass.
    layout.truncate(nonUnit);
    return layout;
  }

  virtual SmallVector<size_t> computeThreads(Target &tg) {
    auto sgLayout = computeSgLayout(tg);
    auto sgs = [&] {
      return std::accumulate(sgLayout.begin(), sgLayout.end(), size_t{1},
                             std::multiplies<>());
    };
    // Divide by 2 due to -ze-opt-large-register-file.
    auto maxSgs = getWgSize(tg) / 2 / getSgSize(tg);
    for (auto &d : llvm::reverse(sgLayout))
      for (; sgs() > maxSgs && d > 1; d /= 2);
    assert(sgs() <= maxSgs && "wg/sg tiling exceeds max wg size");
    tg.kernelAttrs.setSgLayout(sgLayout);
    return {getSgSize(tg) * sgs(), 1, 1};
  }

  virtual size_t getWgSize(Target &tg) {
    if (auto size = tg.kernelAttrs.getWgSize()) {
      return size.value();
    }
    return tg.devAttrs.getMaxWgSize().value_or(1024);
  }

  virtual size_t getSgSize(Target &tg) {
    if (auto size = tg.kernelAttrs.getSgSize()) {
      return size.value();
    }
    return tg.devAttrs.getUarch()->getSubgroupSize();
  }

  virtual LoopLikeOpInterface apply(Target &tg) {
    if (!tg.hasTiles()) {
      return nullptr;
    }

    SCFTileAndFuseOptions opts;
    opts.tilingOptions.loopType = tg.level == Level::SG
                                      ? SCFTilingOptions::LoopType::ForOp
                                      : SCFTilingOptions::LoopType::ForallOp;
    opts.setFusionControlFn([this, &tg](tensor::ExtractSliceOp candidateSliceOp,
                                        OpResult originalProducer,
                                        bool isDestinationOperand) {
      return this->fusionControl(tg, candidateSliceOp, originalProducer,
                                 isDestinationOperand);
    });

    if (tg.hasReductions()) {
      SmallVector<unsigned> reductionDims;
      for (auto [i, r] : llvm::enumerate(tg.reductions)) {
        if (r && tg.tiles[i] != 0) {
          reductionDims.push_back(i);
        }
      }
      opts.tilingOptions.setReductionDims(reductionDims);
    }
    {
      OpFoldResult zero = tg.rw.getIndexAttr(0);
      opts.tilingOptions.setTileSizes(
          llvm::map_to_vector(tg.tiles, [&](size_t t) {
            return t == 0 ? zero : tg.rw.getIndexAttr(t);
          }));
    }

    // Captured before tiling: tg.op may be erased once its uses are replaced
    // below, so getResultIndexingMap must not be recomputed from it later.
    AffineMap prodMap = getResultIndexingMap(tg.op.getOperation(), 0);

    auto result = tileConsumerAndFuseProducersUsingSCF(tg.rw, tg.op, opts);
    if (failed(result)) {
      tg.op->emitError() << "Failed to tile and fuse using SCF";
      return nullptr;
    }

    tagTileAndFuseResult(tg.op.getOperation(), tg.tiles, *result);

    LoopLikeOpInterface opReplacement = nullptr;
    SmallVector<Operation *> opsToReplace{tg.op.getOperation()};
    append_range(opsToReplace, result->fusedProducers);
    for (auto toReplace : opsToReplace) {
      for (auto res : toReplace->getResults()) {
        if (auto repl = result->replacements.lookup(res)) {
          tg.mark(repl.getDefiningOp());
          tg.rw.replaceAllUsesWith(res, repl);
          if (auto loop = dyn_cast<LoopLikeOpInterface>(repl.getDefiningOp())) {
            if (!opReplacement && tg.op == toReplace) {
              opReplacement = loop;
            }
            replaceEmptySlices(tg.rw, loop);
            tg.mark(loop);
          }
        }
      }
      if (toReplace->use_empty()) {
        tg.rw.eraseOp(toReplace);
      }
    }

    if (!opReplacement) {
      tg.op->emitError() << "Nothing tiled";
      return nullptr;
    }

    if (tg.level == Level::WG) {
      if (!fuseConsumers(tg, opReplacement, prodMap)) return nullptr;
      else fuseProducers(tg, opReplacement);
    }

    canonicalizeLoop(opReplacement);
    if (failed(simplifyRegions(tg.rw, tg.fn->getRegions()))) {
      // Not simplified
    }
    tg.mark(opReplacement);
    return opReplacement;
  }

  virtual std::optional<SCFTileAndFuseOptions::ControlFnResult>
  fusionControl(Target &tg, tensor::ExtractSliceOp candidateSliceOp,
                OpResult originalProducer, bool isDestinationOperand) {
    Operation *op = originalProducer.getOwner();
    if (!op) return std::nullopt;
    if (isDestinationOperand && tg.level == Level::SG) return std::nullopt;

    if (LoopLikeOpInterface loop = llvm::dyn_cast_or_null<LoopLikeOpInterface>(
            candidateSliceOp->getParentOp());
        loop && !canFuse(loop, op, false)) {
      return std::nullopt;
    }

    // If the result of this slice is used by a MatmulOp and the slice
    // has an operand produced by a previous MatmulOp, do not fuse.
    if (isOpDependsOnResult<0>(isMatmulOp, candidateSliceOp) &&
        isOperandDependsOnOp(isMatmulOp, candidateSliceOp)) {
      return std::nullopt;
    }

    return SCFTileAndFuseOptions::ControlFnResult{};
  }

  static bool fuseConsumers(Target &tg, LoopLikeOpInterface &loop,
                            AffineMap prodMap) {
    std::function<bool(Operation *)> isFusible = [&](Operation *op) {
      if (canFuse(loop, op)) return true;
      if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op))
        return false;
      for (auto user : op->getUsers()) {
        if (isFusible(user)) return true;
      }
      return false;
    };

    for (bool fused = true; fused;) {
      fused = false;
      for (auto res : loop->getResults()) {
        if (res.use_empty()) {
          loop->emitWarning()
              << "Result " << res.getResultNumber() << " is unused";
          continue;
        }

        auto forall = dyn_cast<scf::ForallOp>(loop.getOperation());
        Operation *user = nullptr;
        OpOperand *operand = nullptr;
        {
          for (auto &use : res.getUses()) {
            auto u = use.getOwner();
            if (!isFusible(u) || u->hasTrait<OpTrait::ReturnLike>() ||
                hasDiamondDep(res, u))
              continue;
            // We can fuse a single consumer with a single result
            if ((!res.hasNUsesOrMore(2) && u->getNumResults() <= 1) ||
                // or the subgraphs if they have a common intersection.
                (forall && // For WG level only
                 allUsersIntersect(res, isFusible))) {
              user = u;
              operand = &use;
              break;
            }
          }
        }

        if (!user) continue;

        if (forall) {
          if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(user)) {
            bool ok;
            if (auto expand = dyn_cast<tensor::ExpandShapeOp>(user))
              ok = fuseExpandShape(tg.rw, forall, res, expand);
            else
              ok = fuseCollapseShape(tg.rw, forall, res,
                                     cast<tensor::CollapseShapeOp>(user));
            if (!ok) {
              user->emitWarning() << "Failed to fuse reshape op into " << loop;
              continue;
            }
            loop = forall;
            canonicalizeLoop(loop);
            tg.mark(loop);
            fused = true;
            break;
          }
        }

        if (!fuseConsumer(tg, loop, res, user, operand, prodMap)) continue;
        canonicalizeLoop(loop);
        tg.mark(loop);
        fused = true;
        break;
      }
    }
    return true;
  }

  // Fuse producers of extract_slice ops inside the loop
  static void fuseProducers(Target &tg, LoopLikeOpInterface &loop) {
    SmallVector<tensor::ExtractSliceOp> candidates;
    loop->walk([&](tensor::ExtractSliceOp slice) {
      auto producer = slice.getSource().getDefiningOp();
      if (!producer || !canFuse(loop, producer, false)) return;
      if (isOpDependsOnResult<0>(isMatmulOp, slice) &&
          isOperandDependsOnOp(isMatmulOp, slice))
        return;
      candidates.push_back(slice);
    });
    SmallVector<LoopLikeOpInterface> loops = {loop};
    for (auto slice : candidates)
      if (!tileAndFuseProducerOfSlice(tg.rw, slice, loops))
        slice->emitWarning() << "Failed to fuse producer of slice";
    loop = loops[0];
  }

private:
  static bool canFuse(LoopLikeOpInterface loop, Operation *op,
                      bool isConsumer = true) {
    if (op->getBlock() != loop->getBlock()) return false;
    if (containsMatmulOp(op)) return false;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
        linalgOp && !linalgOp.hasOnlyProjectedPermutations()) {
      return false;
    }
    if (isParallel(op)) return true;
    if (isa<linalg::PackOp, linalg::UnPackOp>(loop)) return true;
    if (isa<scf::ForOp>(loop)) return false;
    if (hasIterator(op, utils::IteratorType::reduction)) {
      // We can fuse reductions if the reduction dim is not tiled.
      if (!isConsumer) return true;
      auto forall = dyn_cast<scf::ForallOp>(loop.getOperation());
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!forall || !linalgOp) return false;
      for (auto *operand : linalgOp.getDpsInputOperands()) {
        auto opRes = dyn_cast<OpResult>(operand->get());
        if (!opRes || opRes.getDefiningOp() != loop.getOperation()) continue;
        auto ins = findParallelInsertSlice(forall, opRes);
        if (!ins) return false;
        auto srcType = cast<RankedTensorType>(opRes.getType());
        auto sliceSizes = ins.getMixedSizes();
        auto map = linalgOp.getMatchingIndexingMap(operand);
        for (auto [i, iterType] :
             llvm::enumerate(linalgOp.getIteratorTypesArray())) {
          if (iterType != utils::IteratorType::reduction) continue;
          for (unsigned d = 0, nd = map.getNumResults(); d < nd; ++d) {
            auto dim = dyn_cast<AffineDimExpr>(map.getResult(d));
            if (!dim || dim.getPosition() != i) continue;
            auto sz = getConstantIntValue(sliceSizes[d]);
            if (!sz || *sz != srcType.getDimSize(d)) return false;
          }
        }
        return true;
      }
      return false;
    }
    return false;
  }

  // Fuse `user` into `loop` via tileAndFuseConsumer.
  // If `user` has multiple operands that are loop results (diamond dep from
  // the same loop), all extras are temporarily replaced with tensor.empty so
  // tileAndFuseConsumer sees a single producer. After fusion the
  // extract_slice(empty) operands inside the tiled op are replaced with the
  // actual tile values from the loop's parallel_insert_slices.
  static bool fuseConsumer(Target &tg, LoopLikeOpInterface &loop,
                           OpResult loopRes, Operation *user,
                           OpOperand *operand, AffineMap &prodMap) {
    // (empty, tile, originalLoopRes)
    SmallVector<std::tuple<Value, Value, Value>> emptyToTile;
    auto loopOp = loop.getOperation();
    if (auto forall = dyn_cast<scf::ForallOp>(loopOp)) {
      for (auto &operand : user->getOpOperands()) {
        auto res = dyn_cast<OpResult>(operand.get());
        if (!res || res == loopRes || res.getDefiningOp() != loopOp) continue;
        auto ins = findParallelInsertSlice(forall, res);
        if (!ins) continue;
        auto type = cast<RankedTensorType>(res.getType());
        tg.rw.setInsertionPoint(loop);
        auto empty =
            tg.rw
                .create<tensor::EmptyOp>(type.getShape(), type.getElementType())
                .getResult();
        emptyToTile.push_back({empty, ins.getSource(), res});
        operand.set(empty);
      }
    }

    AffineMap userMap;
    if (auto userIdx = dyn_cast<IndexingMapOpInterface>(user))
      userMap = userIdx.getMatchingIndexingMap(operand);

    SmallVector<LoopLikeOpInterface> loops = {loop};
    auto result = tileAndFuseConsumer(tg.rw, user, loops);
    if (failed(result)) { // Restore original operands
      for (auto &[empty, tile, orig] : emptyToTile) {
        empty.use_begin()->set(orig);
        tg.rw.eraseOp(empty.getDefiningOp());
      }
      loop->emitWarning() << "Failed to tile and fuse consumer: " << *user;
      return false;
    }

    // Fuse producers of slices and replace extract_slice(tensor.empty) with
    // actual tile values.
    for (auto tiled : result->tiledOps) {
      for (auto &operand : tiled->getOpOperands()) {
        auto slice = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
        if (!slice) continue;
        for (auto &[empty, tile, orig] : emptyToTile) {
          if (slice.getSource() == empty) {
            operand.set(tile);
            if (empty.use_empty()) tg.rw.eraseOp(empty.getDefiningOp());
            break;
          }
        }
        if (slice.use_empty()) tg.rw.eraseOp(slice);
      }
      tg.mark(tiled);
      if (auto ti = dyn_cast<TilingInterface>(tiled);
          ti && userMap && prodMap) {
        mergeWgTileSizesAttr(tiled,
                             remapTiles(tg.tiles, prodMap, userMap,
                                        ti.getLoopIteratorTypes().size()));
      }
    }

    loop = loops[0];
    tg.rw.eraseOp(user);
    return true;
  }

  static scf::ForallOp rebuildForall(OpRewriter &rw, scf::ForallOp forall,
                                     unsigned resIdx, Value newOut) {
    SmallVector<Value> newOutputs(forall.getOutputs());
    newOutputs[resIdx] = newOut;
    auto mapping = forall.getMappingAttr();
    auto newForall = rw.create<scf::ForallOp>(
        forall.getMixedLowerBound(), forall.getMixedUpperBound(),
        forall.getMixedStep(), newOutputs,
        mapping ? std::optional(mapping) : std::nullopt);
    newForall.getBody()->erase();
    newForall.getRegion().takeBody(forall.getRegion());
    return newForall;
  }

  static bool fuseCollapseShape(OpRewriter &rw, scf::ForallOp &forall,
                                OpResult loopRes,
                                tensor::CollapseShapeOp collapseOp) {
    tensor::ParallelInsertSliceOp ins =
        findParallelInsertSlice(forall, loopRes);
    if (!ins) return false;

    auto srcType = collapseOp.getSrcType();
    auto dstType = collapseOp.getResultType();
    auto reassoc = collapseOp.getReassociationIndices();
    unsigned resIdx = loopRes.getResultNumber();
    auto insOffsets = ins.getMixedOffsets();
    auto insSizes = ins.getMixedSizes();

    for (auto [gi, group] : llvm::enumerate(reassoc)) {
      if (group.size() <= 1) continue;
      auto sz0 = getConstantIntValue(insSizes[group[0]]);
      if (!sz0) return false;
      if (*sz0 == 1) continue;
      for (size_t i = 1; i < group.size(); ++i) {
        int64_t dimSz = srcType.getDimSize(group[i]);
        if (dimSz == 1) continue;
        auto off = getConstantIntValue(insOffsets[group[i]]);
        auto sz = getConstantIntValue(insSizes[group[i]]);
        if (!off || *off != 0 || !sz || *sz != dimSz) return false;
      }
    }

    // sum_i(off[group[i]] * stride_i), skipping unit dims (they contribute 0).
    auto linearOffset = [&](const ReassociationIndices &group,
                            ArrayRef<OpFoldResult> offs) -> OpFoldResult {
      Value acc = nullptr;
      int64_t stride = 1;
      for (int i = (int)group.size() - 1; i >= 0; --i) {
        int64_t dimSz = srcType.getDimSize(group[i]);
        if (dimSz != 1) {
          if (auto cst = getConstantIntValue(offs[group[i]]);
              !cst || *cst != 0) {
            Value v = dyn_cast<Value>(offs[group[i]]);
            if (!v)
              v = rw.create<arith::ConstantIndexOp>(
                  *getConstantIntValue(offs[group[i]]));
            if (stride != 1)
              v = rw.create<arith::MulIOp>(
                  v, rw.create<arith::ConstantIndexOp>(stride));
            acc = acc ? (Value)rw.create<arith::AddIOp>(acc, v) : v;
          }
        }
        if (i > 0) stride *= dimSz;
      }
      return acc ? (OpFoldResult)acc : rw.getIndexAttr(0);
    };

    // Build collapsed (offsets, sizes) from per-dim offsets/sizes.
    auto colParams = [&](ArrayRef<OpFoldResult> offs,
                         ArrayRef<OpFoldResult> szs)
        -> std::pair<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>> {
      SmallVector<OpFoldResult> colOffs, colSzs;
      for (auto &group : reassoc) {
        colOffs.push_back(linearOffset(group, offs));
        int64_t sz = 1;
        for (auto d : group) sz *= *getConstantIntValue(szs[d]);
        colSzs.push_back(rw.getIndexAttr(sz));
      }
      return {colOffs, colSzs};
    };

    auto colType = [&](ArrayRef<OpFoldResult> szs, Type elem) {
      return RankedTensorType::get(
          llvm::map_to_vector(
              szs, [](OpFoldResult r) { return *getConstantIntValue(r); }),
          elem);
    };

    SmallVector<OpFoldResult> unitStrides(reassoc.size(), rw.getIndexAttr(1));

    rw.setInsertionPoint(forall);
    rw.loc = forall.getLoc();
    Value newOut = rw.create<tensor::CollapseShapeOp>(
        dstType, forall.getOutputs()[resIdx], reassoc);
    auto newForall = rebuildForall(rw, forall, resIdx, newOut);

    BlockArgument newOutArg =
        newForall.getTiedBlockArgument(newForall->getResult(resIdx));
    newOutArg.setType(dstType);

    for (auto *user : llvm::make_early_inc_range(newOutArg.getUsers())) {
      auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!slice) continue;
      rw.setInsertionPoint(slice);
      rw.loc = slice.getLoc();
      auto [colOffs, colSzs] =
          colParams(slice.getMixedOffsets(), slice.getMixedSizes());
      auto colSlice = rw.create<tensor::ExtractSliceOp>(
          colType(colSzs, srcType.getElementType()), newOutArg, colOffs, colSzs,
          unitStrides);
      rw.replaceOp(slice, rw.create<tensor::ExpandShapeOp>(slice.getType(),
                                                           colSlice, reassoc)
                              .getResult());
    }

    rw.setInsertionPoint(newForall.getTerminator());
    rw.loc = ins.getLoc();
    auto [newOffs, newSzs] = colParams(insOffsets, insSizes);
    auto collapsedTile = rw.create<tensor::CollapseShapeOp>(
        colType(
            newSzs,
            cast<RankedTensorType>(ins.getSource().getType()).getElementType()),
        ins.getSource(), reassoc);
    rw.setInsertionPoint(ins);
    rw.create<tensor::ParallelInsertSliceOp>(collapsedTile, newOutArg, newOffs,
                                             newSzs, unitStrides);
    rw.eraseOp(ins);

    // rw.replaceAllUsesWith(loopRes, newForall->getResult(resIdx));
    // rw.replaceAllUsesWith(collapseOp.getResult(),
    // newForall->getResult(resIdx));
    rw.replaceAllUsesWith(collapseOp.getResult(), newForall->getResult(resIdx));
    rw.eraseOp(collapseOp);
    rw.replaceOp(forall, newForall);
    forall = newForall;
    return true;
  }

  static bool fuseExpandShape(OpRewriter &rw, scf::ForallOp &forall,
                              OpResult loopRes,
                              tensor::ExpandShapeOp expandOp) {
    auto dstType = expandOp.getResultType();
    auto reassoc = expandOp.getReassociationIndices();
    unsigned resIdx = loopRes.getResultNumber();
    tensor::ParallelInsertSliceOp ins =
        findParallelInsertSlice(forall, loopRes);
    if (!ins) return false;

    auto insOffsets = ins.getMixedOffsets();
    auto insSizes = ins.getMixedSizes();

    // innerProd[gi] = product of dst dims group[1:] for each reassoc group.
    SmallVector<int64_t> innerProds(reassoc.size(), 1);
    for (auto [gi, group] : llvm::enumerate(reassoc)) {
      if (group.size() <= 1) continue;
      int64_t srcDim = expandOp.getCorrespondingSourceDim(group[0]);
      if (!dyn_cast_if_present<Value>(insOffsets[srcDim])) return false;
      for (size_t i = 1; i < group.size(); ++i)
        innerProds[gi] *= dstType.getDimSize(group[i]);
      auto tileSzOpt = getConstantIntValue(insSizes[srcDim]);
      if (!tileSzOpt) return false;
      int64_t ts = *tileSzOpt, ip = innerProds[gi];
      // Full: tile covers k complete outer rows (ts % ip == 0).
      // Sub-inner: tile is smaller than one inner slice (ip % ts == 0), only
      // 2-dim groups supported so the partial dim is unambiguous.
      if (ts % ip != 0 && (group.size() != 2 || ip % ts != 0)) return false;
    }

    // Compute expanded offsets/sizes from collapsed src coords.
    //   Full:      outer_off = src_off / ip, inner dims fully covered.
    //   Sub-inner: outer_off = src_off / ip, last inner off = src_off % ip.
    auto expandParams = [&](ArrayRef<OpFoldResult> offs,
                            ArrayRef<OpFoldResult> szs)
        -> std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
                      SmallVector<OpFoldResult>> {
      SmallVector<OpFoldResult> expOffs, expSzs, expStrides;
      for (auto [gi, group] : llvm::enumerate(reassoc)) {
        int64_t srcDim = expandOp.getCorrespondingSourceDim(group[0]);
        int64_t ip = innerProds[gi];
        int64_t ts = *getConstantIntValue(szs[srcDim]);
        bool subInner = ip > 1 && ip % ts == 0 && ts % ip != 0;
        // outer dim
        if (ip == 1) {
          expOffs.push_back(offs[srcDim]);
          expSzs.push_back(szs[srcDim]);
        } else {
          Value ip_v = rw.create<arith::ConstantIndexOp>(ip);
          Value srcOff = cast<Value>(offs[srcDim]);
          expOffs.push_back((Value)rw.create<arith::DivUIOp>(srcOff, ip_v));
          expSzs.push_back(rw.getIndexAttr(subInner ? 1 : ts / ip));
        }
        expStrides.push_back(rw.getIndexAttr(1));
        // inner dims
        for (size_t i = 1; i < group.size(); ++i) {
          bool isPartial = subInner && i == group.size() - 1;
          if (isPartial) {
            Value ip_v = rw.create<arith::ConstantIndexOp>(ip);
            expOffs.push_back((Value)rw.create<arith::RemUIOp>(
                cast<Value>(offs[srcDim]), ip_v));
            expSzs.push_back(rw.getIndexAttr(ts));
          } else {
            expOffs.push_back(rw.getIndexAttr(0));
            expSzs.push_back(rw.getIndexAttr(dstType.getDimSize(group[i])));
          }
          expStrides.push_back(rw.getIndexAttr(1));
        }
      }
      return {expOffs, expSzs, expStrides};
    };

    rw.setInsertionPoint(forall);
    rw.loc = forall.getLoc();
    Value newOut = rw.create<tensor::ExpandShapeOp>(
        dstType, forall.getOutputs()[resIdx], reassoc,
        expandOp.getMixedOutputShape());
    auto newForall = rebuildForall(rw, forall, resIdx, newOut);

    BlockArgument newOutArg =
        newForall.getTiedBlockArgument(newForall->getResult(resIdx));
    newOutArg.setType(dstType);

    auto expandedTileType = [&](ArrayRef<OpFoldResult> szs) {
      return RankedTensorType::get(
          llvm::map_to_vector(
              szs, [](OpFoldResult r) { return *getConstantIntValue(r); }),
          dstType.getElementType());
    };

    for (auto *user : llvm::make_early_inc_range(newOutArg.getUsers())) {
      auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!slice) continue;
      rw.setInsertionPoint(slice);
      rw.loc = slice.getLoc();
      auto [offs, szs, strides] =
          expandParams(slice.getMixedOffsets(), slice.getMixedSizes());
      auto extracted = rw.create<tensor::ExtractSliceOp>(
          expandedTileType(szs), newOutArg, offs, szs, strides);
      rw.replaceOp(slice, rw.create<tensor::CollapseShapeOp>(
                              slice.getType(), extracted, reassoc));
    }

    rw.setInsertionPoint(newForall.getTerminator());
    rw.loc = ins.getLoc();
    auto [offs, szs, strides] = expandParams(insOffsets, insSizes);
    auto expandedTile = rw.create<tensor::ExpandShapeOp>(
        expandedTileType(szs), ins.getSource(), reassoc);
    rw.setInsertionPoint(ins);
    rw.create<tensor::ParallelInsertSliceOp>(expandedTile, newOutArg, offs, szs,
                                             strides);
    rw.eraseOp(ins);

    rw.replaceAllUsesWith(expandOp.getResult(), newForall->getResult(resIdx));
    rw.eraseOp(expandOp);
    rw.replaceOp(forall, newForall);
    forall = newForall;
    return true;
  }
};
#endif // TILING_UTILS_H
