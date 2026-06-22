#ifndef TILING_UTILS_H
#define TILING_UTILS_H
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "gc/Utils/Log.h"
#include "gc/Utils/Misc.h"
#include "gc/Utils/Transform.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::gc;
using namespace mlir::scf;

constexpr char GC_ATTR_LEVEL[] = "gc.tiling.level";
constexpr char GC_ATTR_NUM_KERNELS[] = "gc.num_kernels";
constexpr char GC_ATTR_WG_TILE_SIZES[] = "gc.tiling.wg_tile_sizes";

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
  // it's always the first element in the tiledAndFusedOps list.
  mergeWgTileSizesAttr(*it++, consumerTiles);

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

inline bool isParallel(Operation *op) {
  auto ti = dyn_cast<TilingInterface>(op);
  return ti &&
         llvm::all_of(ti.getLoopIteratorTypes(), [](utils::IteratorType t) {
           return t == utils::IteratorType::parallel;
         });
}

inline bool hasIterator(Operation *op, utils::IteratorType type) {
  auto ti = dyn_cast<TilingInterface>(op);
  return ti &&
         llvm::any_of(ti.getLoopIteratorTypes(),
                      [type](utils::IteratorType t) { return t == type; });
}

// If a slice inside the loop is created from an external empty tensor and
// the tensor is not passed to the loop's shared_outs, but referenced
// directly, replace the slice with an empty tensor of the same size.
inline void replaceEmptySlices(OpRewriter &rw, LoopLikeOpInterface loop) {
  loop.walk([&](tensor::ExtractSliceOp slice) {
    if (auto empty = slice.getSource().getDefiningOp<tensor::EmptyOp>();
        empty && empty->getParentOfType<LoopLikeOpInterface>() != loop) {
      auto type = slice.getType();
      rw.setInsertionPointAfter(slice);
      SmallVector<Value> dynDims;
      for (int64_t i = 0, r = type.getRank(); i < r; ++i) {
        if (type.isDynamicDim(i)) {
          dynDims.push_back(rw.create<tensor::DimOp>(slice, i));
        }
      }
      rw.replaceOp(slice, rw.create<tensor::EmptyOp>(
                              type.getShape(), type.getElementType(), dynDims));
    }
  });
}

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
        kernelName.resize(fn.getName().size() + 7);
        kernelName.append("_");
        kernelName.append(buffer);
        kernelAttrs = KernelAttrs(fn, kernelName);
        ++numKernels;
      }
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

  // Tile the last 2 dims and set all leading dims to 1.
  virtual void computeTiles(Target &tg, bool reduction) {
    auto wgTiles = tg.getSizes(reduction);
    if (wgTiles.empty()) return;
    SmallVector<size_t> sgTiles(wgTiles.size(), 1);

    bool unit = wgTiles.size() == 1;
    for (auto &t :
         llvm::make_range(wgTiles.begin(), wgTiles.end() - (unit ? 1 : 2)))
      t = 1;

    size_t dummy = 1;
    auto &wTile = wgTiles.back();
    auto &hTile = unit ? dummy : wgTiles[wgTiles.size() - 2];

    // TODO: parameterize sgMul and wgMul in kernel attributes so they can
    // be used for auto tuning.
    auto [widths, heights, counts, sgMul, wgMul] =
        getSupportedBlockSizes(tg, reduction, wTile, hTile);
    if (unit) {
      heights = {1};
    } else if (reduction) {
      sgMul = wgMul = 1;
    } else {
      sgMul = std::sqrt(sgMul);
      wgMul = std::sqrt(wgMul);
    }

    auto sgSize = getSgSize(tg);
    auto wgSize = getWgSize(tg);
    auto maxMul = reduction ? 1 : wgSize / sgSize;

    for (auto w : widths)
      for (auto h : heights)
        for (auto c : counts)
          for (auto sm = sgMul; sm; sm /= 2)
            for (auto wm = wgMul; wm; wm /= 2) {
              if ((unit ? wm : wm * wm) > maxMul) continue;
              auto sgw = w * c * sm, sgh = h * sm;
              auto wgw = sgw * wm, wgh = sgh * wm;
              if (wTile % wgw || (!unit && hTile % wgh)) continue;
              if (wTile == wgw && (unit || hTile == wgh)) continue;
              wTile = wgw;
              hTile = wgh;
              sgTiles.back() = sgw;
              if (!unit) sgTiles[wgTiles.size() - 2] = sgh;
              tg.setTiles(wgTiles, sgTiles, reduction);
              return;
            }

    wTile = 1;
    hTile = 1;
    sgTiles.back() = 1;
    if (!unit) sgTiles[wgTiles.size() - 2] = 1;
    tg.setTiles(wgTiles, sgTiles, reduction);
  }

  // Get the supported block sizes, that can be used for tiling of the specified
  // width and height.
  //
  // Returns block widths, heights, counts, SG-tile multiplier,
  // WG-tile multiplier
  virtual std::tuple<SmallVector<unsigned>, SmallVector<unsigned>,
                     SmallVector<unsigned>, unsigned, unsigned>
  getSupportedBlockSizes(Target &tg, bool reduction, size_t width,
                         size_t height) {
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
      return std::make_tuple(SmallVector<unsigned>{1}, SmallVector<unsigned>{1},
                             SmallVector<unsigned>{1}, 1, 1);
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
    SmallVector<unsigned> widths, heights, counts;
    for (unsigned w : std::get<0>(loadSizes))
      if (width % w == 0 && llvm::is_contained(std::get<0>(storeSizes), w))
        widths.push_back(w);
    for (unsigned h : std::get<1>(loadSizes))
      if (height % h == 0 && llvm::is_contained(std::get<1>(storeSizes), h))
        heights.push_back(h);
    for (unsigned c : std::get<2>(loadSizes))
      if (llvm::is_contained(std::get<2>(storeSizes), c)) counts.push_back(c);
    for (auto l : {&widths, &heights, &counts})
      if (!llvm::is_contained(*l, 1)) l->push_back(1);

    llvm::sort(widths, std::greater<unsigned>());
    llvm::sort(heights, std::greater<unsigned>());
    llvm::sort(counts, std::greater<unsigned>());
    return std::make_tuple(widths, heights, counts, 2, getSgSize(tg));
  }

  virtual SmallVector<size_t> computeThreads(Target &tg) {
    size_t threads = getSgSize(tg);
    for (auto [wg, sg, r] : llvm::zip(tg.tiles, tg.sgTiles, tg.reductions)) {
      if (!r) {
        threads *= wg / sg;
      }
    }

    auto wgSize = getWgSize(tg);
    assert(threads <= wgSize && "wg/sg tiling exceeds max wg size");
    // Divide by 2 due to -ze-opt-large-register-file
    return {std::min(threads, wgSize / 2), 1, 1};
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

    auto result = tileConsumerAndFuseProducersUsingSCF(tg.rw, tg.op, opts);
    if (failed(result)) {
      tg.op->emitError() << "Failed to tile and fuse using SCF";
      return nullptr;
    }

    tagTileAndFuseResult(tg.op.getOperation(), tg.tiles, *result);

    LoopLikeOpInterface opReplacement = nullptr;
    SmallVector<Operation *> opsToReplace{tg.op.getOperation()};
    append_range(opsToReplace, result->fusedProducers);
    auto ctx = tg.op->getContext();
    RewritePatternSet patterns(ctx);
    for (auto toReplace : opsToReplace) {
      for (auto res : toReplace->getResults()) {
        if (auto repl = result->replacements.lookup(res)) {
          tg.mark(repl.getDefiningOp());
          tg.rw.replaceAllUsesWith(res, repl);
          if (auto loop = dyn_cast<LoopLikeOpInterface>(repl.getDefiningOp())) {
            if (isa<scf::ForallOp>(loop.getOperation())) {
              scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
            } else if (isa<scf::ForOp>(loop.getOperation())) {
              scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
            }
            if (tg.level == Level::WG && !fuseConsumers(tg, loop)) {
              return nullptr;
            }
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

    static size_t stamp = 0;
    auto st = ++stamp;
    // The loop's operation can be replaced by the patterns. Using a stamp
    // to find it again.
    opReplacement->setDiscardableAttr("gc.tiling.stamp", createAttr(ctx, st));
    if (failed(applyPatternsGreedily(tg.fn, std::move(patterns)))) {
      return nullptr;
    }
    if (failed(simplifyRegions(tg.rw, tg.fn->getRegions()))) {
      // Not simplified
    }
    tg.fn.walk([&](LoopLikeOpInterface loop) {
      if (getDiscardableAttr<size_t>(loop, "gc.tiling.stamp", 0) == st) {
        opReplacement = loop;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    return opReplacement;
  }

  inline bool fuseConsumers(Target &tg, LoopLikeOpInterface &loop) {
    for (bool fused = true; fused;) {
      fused = false;
      for (auto res : loop->getResults()) {
        if (!res.hasOneUse()) {
          continue;
        }
        auto &uses_begin = *res.use_begin();
        auto user = uses_begin.getOwner();
        if (user->getBlock() == loop->getBlock() && isParallel(user) &&
            user->getNumResults() == 1 && user->getResult(0).hasOneUse()) {
          AffineMap userMap;
          if (auto userIdx = dyn_cast<IndexingMapOpInterface>(user))
            userMap = userIdx.getMatchingIndexingMap(&uses_begin);
          auto result = tileAndFuseConsumer(tg.rw, user, {loop});
          if (failed(result)) {
            tg.op->emitError() << "Failed to fuse consumers";
            return false;
          }
          fused = true;
          tg.rw.replaceAllOpUsesWith(user, res);
          user->erase();
          tg.mark(loop);
          AffineMap prodMap = getResultIndexingMap(tg.op.getOperation(), 0);
          for (auto tiled : result->tiledOps) {
            tg.mark(tiled);
            if (auto ti = dyn_cast<TilingInterface>(tiled);
                ti && userMap && prodMap) {
              mergeWgTileSizesAttr(
                  tiled, remapTiles(tg.tiles, prodMap, userMap,
                                    ti.getLoopIteratorTypes().size()));
            }
          }
        }
      }
    }
    return true;
  }

  virtual std::optional<SCFTileAndFuseOptions::ControlFnResult>
  fusionControl(Target &tg, tensor::ExtractSliceOp candidateSliceOp,
                OpResult originalProducer, bool isDestinationOperand) {
    Operation *op = originalProducer.getOwner();
    if (!op) {
      return std::nullopt;
    }

    if (isDestinationOperand && tg.level == Level::SG) {
      return std::nullopt;
    }

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
        linalgOp && !linalgOp.hasOnlyProjectedPermutations()) {
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
};
#endif // TILING_UTILS_H
