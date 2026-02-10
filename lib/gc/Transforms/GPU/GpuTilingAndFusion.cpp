//===-- GpuTilingAndFusion.cpp - DESC ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Utils/Log.h"
#include "gc/Utils/Misc.h"
#include "gc/Utils/Transform.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::gc;
using namespace mlir::scf;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUTILINGANDFUSION
#define GEN_PASS_DEF_GPUTILINGANDFUSION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct GpuTilingAndFusion final
    : gc::impl::GpuTilingAndFusionBase<GpuTilingAndFusion> {

  void runOnOperation() override {
    auto fn = getOperation();
    if (fn.isExternal()) {
      return;
    }

    OpRewriter rw(fn);
    tileAndFuseLinalgOps(rw, fn, /*reduction=*/false);
    tileAndFuseLinalgOps(rw, fn, /*reduction=*/true);
  }

private:
  static constexpr char TILING_MARKER[] = "gc.tiling";

  void tileAndFuseLinalgOps(OpRewriter &rw, func::FuncOp fn, bool reduction) {
    unsigned nameCounter = 0;
    SmallString<64> kernelNameBase(fn.getName());
    kernelNameBase.append("_kernel");
    DevAttrs devAttrs(fn);
    size_t vectorWidth = devAttrs.getVectorWidth().value_or(16);
    size_t wgSize = devAttrs.getMaxWgSize().value_or(1024);
    auto sgSizes = devAttrs.getSgSizes().value_or(SmallVector<size_t>{32});
    size_t maxSgSize = *llvm::max_element(sgSizes);
    SCFTileAndFuseOptions opts;
    opts.tilingOptions.setTileSizeComputationFunction(
        [&](OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
          auto ti = dyn_cast<TilingInterface>(op);
          if (!ti) {
            return {};
          }

          op->setDiscardableAttr(
              TILING_MARKER, createAttr(op->getContext(), reduction ? 1 : 0));

          SmallString<64> kernelName;
          if (auto name = op->getDiscardableAttr(GC_ATTR_KERNEL_NAME)) {
            kernelName = getAttrValue<StringRef>(name);
          } else {
            kernelName = kernelNameBase;
            if (++nameCounter != 1) {
              char buffer[8];
              snprintf(buffer, sizeof(buffer), "%u", nameCounter);
              kernelName.append(buffer);
            }
            op->setDiscardableAttr(GC_ATTR_KERNEL_NAME,
                                   createAttr(op->getContext(), kernelName));
          }

          KernelAttrs kernelAttrs(fn, kernelName);
          SmallVector<size_t> tiles(ti.getLoopIteratorTypes().size(), 0);
          size_t sgSize = maxSgSize;
          getTiles(tiles, builder, ti, kernelAttrs, wgSize, sgSize, vectorWidth,
                   reduction);
          kernelAttrs.setSgSize(sgSize);

          SmallVector<OpFoldResult> result;
          result.reserve(tiles.size());
          for (auto t : tiles) {
            result.push_back(builder.getIndexAttr(t));
          }
          return result;
        });
    opts.setFusionControlFn(
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool) -> std::optional<SCFTileAndFuseOptions::ControlFnResult> {
          Operation *op = originalProducer.getOwner();
          if (!op) {
            return std::nullopt;
          }

          if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
              linalgOp && !linalgOp.hasOnlyProjectedPermutations()) {
            return std::nullopt;
          }

          if (auto ti = dyn_cast<TilingInterface>(op)) {
            // Don't fuse parallels into reduction.
            if (reduction &&
                all_of(ti.getLoopIteratorTypes(), [](utils::IteratorType t) {
                  return t == utils::IteratorType::parallel;
                })) {
              return std::nullopt;
            }
          }

          // If the result of this slice is used by a MatmulOp and the slice has
          // an operand produced by a previous MatmulOp, do not fuse.
          if (isOpDependsOnResult<0>(isMatmulOp, candidateSliceOp) &&
              isOperandDependsOnOp(isMatmulOp, candidateSliceOp)) {
            return std::nullopt;
          }

          return SCFTileAndFuseOptions::ControlFnResult{};
        });

    if (reduction) {
      // FIXME: Int causes a buffer allocation after the bufferization pass.
      // opts.tilingOptions.setReductionTilingStrategy(
      //     ReductionTilingStrategy::PartialReductionOuterParallel);
      opts.tilingOptions.setLoopType(SCFTilingOptions::LoopType::ForOp);
    } else {
      opts.tilingOptions.setLoopType(SCFTilingOptions::LoopType::ForallOp);
    }

    for (auto ti = findTi(rw, fn, reduction); ti;
         ti = findTi(rw, fn, reduction)) {
      if (reduction) {
        SmallVector<unsigned> reductionDims;
        auto itTypes = ti->getLoopIteratorTypes();
        for (unsigned i = 0; i < itTypes.size(); ++i) {
          if (itTypes[i] == utils::IteratorType::reduction) {
            reductionDims.push_back(i);
          }
        }
        opts.tilingOptions.setReductionDims(reductionDims);
      }

      auto result = tileConsumerAndFuseProducersUsingSCF(rw, *ti, opts);
      if (failed(result)) {
        ti->emitError() << "Failed to tile and fuse using SCF";
        return;
      }

      SmallVector<Operation *> opsToReplace{ti->getOperation()};
      append_range(opsToReplace, result->fusedProducers);
      for (Operation *toReplace : opsToReplace) {
        for (OpResult res : toReplace->getResults()) {
          if (auto repl = result->replacements.lookup(res)) {
            rw.replaceAllUsesWith(res, repl);
            if (auto loop = dyn_cast<ForallOp>(repl.getDefiningOp())) {
              replaceEmptySlices(rw, loop);
              if (auto v = toReplace->getDiscardableAttr(GC_ATTR_KERNEL_NAME)) {
                loop->setDiscardableAttr(GC_ATTR_KERNEL_NAME, v);
              }
            }
          }
        }
      }

      if (failed(simplifyRegions(rw, fn->getRegions()))) {
        // Not simplified
      }
    }
  }

  static void getTiles(SmallVector<size_t> &tiles, OpBuilder &builder,
                       TilingInterface ti, KernelAttrs &kernelAttrs,
                       size_t wgSize, size_t &sgSize, size_t vectorWidth,
                       bool reduction) {
    if (auto opt = kernelAttrs.getSgSize(); opt.has_value()) {
      sgSize = static_cast<int64_t>(opt.value());
    }
    if (auto opt = kernelAttrs.getTiles(); opt.has_value()) {
      auto fixedTiles = opt.value();
      for (const auto &[t, ft, it] :
           llvm::zip_equal(tiles, fixedTiles, ti.getLoopIteratorTypes())) {
        if (reduction == (it == utils::IteratorType::reduction)) {
          t = ft;
        }
      }
      return;
    }

    SmallVector<size_t> sizes;
    auto itTypes = ti.getLoopIteratorTypes();
    auto itDomains = ti.getIterationDomain(builder);
    size_t maxSize = 0;
    size_t numIterations = 1;

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (auto opt = getConstantIntValue(r.size)) {
        if ((t == utils::IteratorType::reduction) == reduction) {
          auto v = static_cast<size_t>(*opt);
          numIterations *= v;
          sizes.emplace_back(v);
          maxSize = std::max(maxSize, v);
        }
      } else {
        gcLogE("Dynamic tiles are not supported!");
        return;
      }
    }

    if (reduction && isMatmulOp(ti)) {
      tiles[2] = std::min(sgSize, floorPow2(numIterations));
      return;
    }

    // TODO: Analyse the graph of suppliers to be fused and adjust the
    // value.
    size_t workPerTile = reduction ? 1 : 4 * sgSize;
    size_t totalSize = vectorWidth * workPerTile;
    if (totalSize > numIterations) {
      totalSize = std::max(numIterations / vectorWidth * vectorWidth,
                           static_cast<size_t>(1));
    }

    auto adjusted = sizes;
    adjustTiles(totalSize, adjusted);

    if (adjusted == sizes) {
      // Split the largest tile.
      auto tile = findFactor(maxSize, maxSize / 2);

      if (tile == maxSize) {
        // Find another size, that can be split
        auto another = maxSize;
        auto sortedSizes = sizes;
        sort(sortedSizes, std::greater<>());
        for (auto s : sortedSizes) {
          if (s != maxSize && (tile = findFactor(s, s / 2)) != s) {
            another = s;
            break;
          }
        }
        if (another == maxSize) {
          tile = 1;
          // Find the smallest size that is not 1
          for (auto s : reverse(sortedSizes)) {
            if (s != 1) {
              maxSize = s;
              break;
            }
          }
        } else {
          maxSize = another;
        }
      }
      for (auto &t : adjusted) {
        if (t == maxSize) {
          t = tile;
          break;
        }
      }
    }

    unsigned tc = 0;
    unsigned ac = 0;
    for (auto t : itTypes) {
      if ((t == utils::IteratorType::reduction) == reduction) {
        tiles[tc++] = adjusted[ac++];
      } else {
        ++tc;
      }
    }

    if (!reduction && !kernelAttrs.getThreads().has_value()) {
      size_t numThreads = numIterations * sgSize / vectorWidth / workPerTile / 2;
      auto itTypes = ti.getLoopIteratorTypes();
      for (unsigned i = 0; i < itTypes.size(); ++i) {
        if (itTypes[i] == utils::IteratorType::parallel) {
          numThreads /= tiles[i];
        }
      }
      // Align to subgroup size
      numThreads = ((numThreads + sgSize - 1) / sgSize) * sgSize;
      numThreads = std::min(std::max(numThreads, sgSize), wgSize);
      adjustTiles(numThreads, sizes, false);
      kernelAttrs.setThreads(sizes);
    }
  }

  static std::optional<TilingInterface> findTi(OpBuilder &b, Operation *op,
                                               bool reduction) {
    std::optional<TilingInterface> last;
    op->walk<WalkOrder::PreOrder>([&](TilingInterface ti) {
      auto it = reduction ? utils::IteratorType::reduction
                          : utils::IteratorType::parallel;
      if (!llvm::any_of(ti.getLoopIteratorTypes(),
                        [it](utils::IteratorType t) { return t == it; })) {
        return WalkResult::skip();
      }
      if (auto m = ti->getDiscardableAttr(TILING_MARKER)) {
        if (!reduction || getAttrValue<int>(m) == 1) {
          return WalkResult::skip();
        }
      } else if (auto parentLoop = dyn_cast<ForallOp>(ti->getParentOp());
                 parentLoop && parentLoop->hasAttr(GC_ATTR_KERNEL_NAME) &&
                 !reduction) {
        return WalkResult::skip();
      }
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(ti.getOperation());
          linalgOp && !linalgOp.hasOnlyProjectedPermutations()) {
        return WalkResult::skip();
      }

      int64_t numTiles = 0;
      int64_t numIterations = 1;
      for (auto [t, r] :
           zip(ti.getLoopIteratorTypes(), ti.getIterationDomain(b))) {
        if ((t == utils::IteratorType::reduction) == reduction) {
          ++numTiles;
          if (auto v = getConstantIntValue(r.size)) {
            numIterations *= *v;
          }
        }
      }
      if (numTiles > 0 && numIterations >= 32) {
        last = ti;
      }

      return WalkResult::skip();
    });
    return last;
  }

  // If a slice inside the loop is created from an external empty tensor and
  // the tensor is not passed to the loop's shared_outs, but referenced
  // directly, replace the slice with an empty tensor of the same size.
  static void replaceEmptySlices(OpRewriter &rw, ForallOp loop) {
    loop.walk([&](tensor::ExtractSliceOp slice) {
      if (auto empty = slice.getSource().getDefiningOp<tensor::EmptyOp>();
          empty && empty->getParentOfType<ForallOp>() != loop) {
        auto type = slice.getType();
        rw.setInsertionPointAfter(slice);
        SmallVector<Value> dynDims;
        for (int64_t i = 0, r = type.getRank(); i < r; ++i) {
          if (type.isDynamicDim(i)) {
            dynDims.push_back(rw.create<tensor::DimOp>(slice, i));
          }
        }
        rw.replaceOp(slice, rw.create<tensor::EmptyOp>(type.getShape(),
                                                       type.getElementType(),
                                                       dynDims));
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
  static void adjustTwoTiles(T totalSize, T *aPtr, T *bPtr,
                             AdjustTilesMode mode) {
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
    T x =
        static_cast<T>(std::sqrt(totalSize)) * static_cast<T>(std::sqrt(ratio));
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
  // 3. If the initial size is a power of 2, then the resulting size is a power
  // of
  //    2 either. Otherwise, the resulting size is a factor of the initial size
  //    and, if possible, is a power of 2.
  template <typename T>
  static void adjustTiles(T totalSize, T *begin, T *end,
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

      // Split the array in two. The first one consists of the 2 elements - the
      // first one and the product of the rest. The second one is the rest.
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
  static void adjustTiles(T totalSize, SmallVector<T, N> &tiles,
                          bool xeGpuMode = true) {
    adjustTiles(totalSize, tiles.begin(), tiles.end(),
                xeGpuMode ? AdjustTilesMode::XeGpu : AdjustTilesMode::Sort);
  }
};
} // namespace
