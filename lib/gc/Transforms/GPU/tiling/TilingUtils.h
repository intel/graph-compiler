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

using namespace mlir;
using namespace mlir::gc;
using namespace mlir::scf;

constexpr char GC_ATTR_LEVEL[] = "gc.tiling.level";
constexpr char GC_ATTR_NUM_KERNELS[] = "gc.num_kernels";

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

// Controls the adjustment in case of more than 2 tiles.
enum class AdjustTilesMode {
  // Sort the input and switch to the First mode.
  Sort,
  // Adjust the first tile and call adjustTiles() recursively for the rest.
  First,
  // To allow for squeezing, set 1's for all tiles except the last 2.
  XeGpu,
};

template <typename T>
void adjustTwoTiles(T totalSize, T *aPtr, T *bPtr, AdjustTilesMode mode) {
  T a = *aPtr;
  T b = *bPtr;
  assert(a >= b);

  if (a * b <= totalSize) {
    return;
  }

  T minSize = static_cast<T>(mode == AdjustTilesMode::XeGpu ? 8 : 1);
  bool aPow2 = isPow2(a);
  bool bPow2 = isPow2(b);
  double ratio = static_cast<double>(a) / static_cast<double>(b);
  T x = static_cast<T>(std::sqrt(totalSize)) * static_cast<T>(std::sqrt(ratio));
  T y;

  if (aPow2) {
    x = std::min(ceilPow2(x), std::min(a, floorPow2(totalSize)));
  } else {
    x = std::min(findFactor(a, x), std::min(a, totalSize));
  }
  x = std::max(x, minSize);
  if (bPow2) {
    y = std::min(floorPow2(totalSize / x), b);
  } else {
    y = std::min(findFactor(b, totalSize / x), b);
  }
  if (y < minSize && a >= minSize && b >= minSize) {
    if (auto newX = ceilPow2(totalSize / minSize); newX >= minSize) {
      x = std::min(newX, a);
      y = minSize;
    }
  }

  // Adjust x and y to get the closest ratio
  auto distance =
      std::abs(ratio - static_cast<double>(x) / static_cast<double>(y));
  auto ax = aPow2 ? x * 2 : findFactor(a, x * 2);
  auto ay = std::max(bPow2 ? y / 2 : findFactor(b, y / 2), minSize);

  if (ax * ay <= totalSize &&
      std::abs(ratio - static_cast<double>(ax) / static_cast<double>(ay)) <
          distance) {
    x = ax;
    y = ay;
  } else {
    ax = std::max(aPow2 ? x / 2 : findFactor(a, x / 2), minSize);
    ay = bPow2 ? y * 2 : findFactor(b, y * 2);
    if (ax * ay <= totalSize &&
        std::abs(ratio - static_cast<double>(ax) / static_cast<double>(ay)) <
            distance) {
      x = ax;
      y = ay;
    }
  }

  *aPtr = x;
  *bPtr = y;
}

// Adjust tile sizes that meet the following conditions:
// 1. The product of all tiles is as close to totalSize as possible.
// 2. The new sizes are proportional to the initial sizes.
// 3. If the initial size is a power of 2, then the resulting size is a
// power of
//    2 either. Otherwise, the resulting size is a factor of the initial
//    size and, if possible, is a power of 2.
template <typename T>
void adjustTiles(T totalSize, T *begin, T *end,
                 AdjustTilesMode mode = AdjustTilesMode::Sort) {
  auto count = end - begin;
  if (count == 0) {
    return;
  }

  if (count == 1) {
    T minSize = static_cast<T>(mode == AdjustTilesMode::XeGpu ? 8 : 1);
    if (T a = *begin; isPow2(a)) {
      *begin = std::min(std::max(ceilPow2(a), minSize), floorPow2(totalSize));
    } else {
      *begin = std::min(findFactor(a, totalSize), minSize);
    }
    return;
  }

  if (count > 2) {
    if (mode == AdjustTilesMode::XeGpu) {
      for (unsigned i = 0; i < count - 2; ++i) {
        *(begin + i) = 1;
      }
      T *aPtr = end - 2;
      T *bPtr = end - 1;
      if (*aPtr < *bPtr) {
        std::swap(aPtr, bPtr);
      }
      adjustTwoTiles(totalSize, aPtr, bPtr, mode);
      return;
    }

    SmallVector<T> sorted;
    SmallVector<unsigned> indices;
    T *head;
    T *tail;

    if (mode == AdjustTilesMode::First) {
      head = begin;
      tail = end;
    } else {
      assert(mode == AdjustTilesMode::Sort);
      SmallVector<std::pair<T, unsigned>> pairs;
      pairs.reserve(count);
      for (unsigned i = 0; i < count; ++i) {
        pairs.emplace_back(*(begin + i), i);
      }
      llvm::sort(pairs);
      sorted.reserve(count);
      indices.reserve(count);
      for (auto &p : pairs) {
        sorted.push_back(p.first);
        indices.push_back(p.second);
      }
      head = sorted.data();
      tail = head + count;
    }

    // Split the array in two. The first one consists of the 2 elements -
    // the first one and the product of the rest. The second one is the
    // rest.
    T first[] = {*head, std::accumulate(head + 2, tail, *(head + 1),
                                        std::multiplies<>())};
    adjustTiles(totalSize, first, first + 2, AdjustTilesMode::First);
    adjustTiles(totalSize / *first, head + 1, tail, AdjustTilesMode::First);
    *head = *first;

    if (mode == AdjustTilesMode::Sort) {
      for (unsigned i = 0; i < count; ++i) {
        *(begin + indices[i]) = sorted[i];
      }
    }
  } else if (*begin >= *(end - 1)) {
    adjustTwoTiles(totalSize, begin, end - 1, mode);
  } else {
    adjustTwoTiles(totalSize, end - 1, begin, mode);
  }
}

template <typename T, unsigned N>
void adjustTiles(T totalSize, SmallVector<T, N> &tiles, bool xeGpuMode = true) {
  adjustTiles(totalSize, tiles.begin(), tiles.end(),
              xeGpuMode ? AdjustTilesMode::XeGpu : AdjustTilesMode::Sort);
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
  SmallVector<size_t> tiles{};
  SmallVector<size_t> sizes{};
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

    tiles.resize(0);
    sizes.resize(0);
    reductions.resize(0);
    for (auto [i, t, r] : llvm::enumerate(op.getLoopIteratorTypes(),
                                          op.getIterationDomain(rw))) {
      if (auto opt = getConstantIntValue(r.size)) {
        tiles.emplace_back(0);
        sizes.emplace_back(static_cast<size_t>(*opt));
        reductions.emplace_back(t == utils::IteratorType::reduction);
      } else {
        op->emitError("Dynamic tiles are not supported!");
        return false;
      }
    }
    return true;
  }

  std::pair<SmallVector<size_t>, size_t> getSizes(bool reduction) {
    size_t product = 1;
    SmallVector<size_t> filtered;
    for (size_t i = 0, n = sizes.size(); i < n; ++i) {
      if (reductions[i] == reduction) {
        filtered.push_back(sizes[i]);
        product *= sizes[i];
      }
    }
    return {filtered, product};
  }

  void setTiles(SmallVector<size_t> tiles, bool reduction) {
    for (size_t i = 0, j = 0, n = this->tiles.size(); i < n; ++i) {
      if (reductions[i] == reduction) {
        this->tiles[i] = tiles[j++];
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
      if (auto loop = apply(tg)) {
        computeThreads(tg);
        tg.kernelAttrs.setThreads(tg.tiles);
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
    while (auto ti = findLast<TilingInterface, Filter>(tg.fn, predicate)) {
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

  virtual void computeWgTiles(Target &tg) {
    auto [tiles, total] = tg.getSizes(false);
    auto wgSize = getWgSize(tg);
    auto sgSize = getSgSize(tg);
    total = std::min(total / wgSize / sgSize, wgSize * sgSize * 8);
    total = std::max<size_t>(sgSize, total);
    adjustTiles(std::max<size_t>(1, total), tiles);
    tg.setTiles(tiles, false);
  }

  virtual void computeSgTiles(Target &tg) {
    auto [tiles, total] = tg.getSizes(true);
    adjustTiles(std::max<size_t>(1, total / getSgSize(tg)), tiles);
    tg.setTiles(tiles, true);
  }

  virtual void computeThreads(Target &tg) {
    auto [sizes, _] = tg.getSizes(false);
    for (auto [t, s] : llvm::zip(tg.tiles, sizes)) {
      t = t == 0 ? 1 : std::max<size_t>(1, s / t);
    }
    size_t product = std::accumulate(tg.tiles.begin(), tg.tiles.end(), 1,
                                     std::multiplies<>());
    adjustTiles(std::max<size_t>(1, product / getSgSize(tg)), tg.tiles, false);

    if (tg.tiles.size() > 3) {
      product = std::accumulate(tg.tiles.begin(), tg.tiles.end(), 1,
                                std::multiplies<>());
      tg.tiles = {product, 1, 1};
    } else {
      tg.tiles.resize(3, 1);
    }
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
    // The loop's operation can be replaced by the patterns. Using a stamp to
    // find it again.
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
        auto user = res.use_begin()->getOwner();
        if (user->getBlock() == loop->getBlock() && isParallel(user) &&
            user->getNumResults() == 1 && user->getResult(0).hasOneUse()) {
          auto result = tileAndFuseConsumer(tg.rw, user, {loop});
          if (failed(result)) {
            tg.op->emitError() << "Failed to fuse consumers";
            return false;
          }
          fused = true;
          tg.rw.replaceAllOpUsesWith(user, res);
          user->erase();
          tg.mark(loop);
          for (auto tiled : result->tiledOps) {
            tg.mark(tiled);
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