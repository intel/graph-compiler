//===-- GpuTilingAndFusion.cpp - DESC ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./GpuUtils.h"
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
  friend struct TileAndFuseLinalgOpsPattern;
  explicit GpuTilingAndFusion()
      : GpuTilingAndFusion(GpuTilingAndFusionOptions{}) {}
  explicit GpuTilingAndFusion(const GpuTilingAndFusionOptions &opts)
      : GpuTilingAndFusionBase(opts) {}

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
  static constexpr char NO_TILE_MARKER[] = "gc.no_tile";

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
          if (reduction) {
            op->setDiscardableAttr(NO_TILE_MARKER, builder.getUnitAttr());
          }

          KernelAttrs kernelAttrs(fn, kernelName);
          SmallVector<size_t> tiles(ti.getLoopIteratorTypes().size(), 0);
          size_t sgSize = maxSgSize;
          getTiles(tiles, builder, ti, kernelAttrs, wgSize, sgSize, vectorWidth,
                   reduction);

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

          if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
            if (!linalgOp.hasOnlyProjectedPermutations()) {
              return std::nullopt;
            }

            // Don't fuse parallels into reduction.
            if (reduction && all_of(linalgOp.getIteratorTypesArray(),
                                    [](utils::IteratorType t) {
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
      tiles = *opt;
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
      size_t numThreads = numIterations * sgSize / vectorWidth / workPerTile;
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
    op->walk<WalkOrder::PreOrder>([&](linalg::LinalgOp linalgOp) {
      if (!linalgOp.hasOnlyProjectedPermutations()) {
        return WalkResult::skip();
      }
      if (linalgOp->hasAttr(NO_TILE_MARKER)) {
        return WalkResult::skip();
      }
      if (auto parentLoop = linalgOp->getParentOfType<ForallOp>();
          parentLoop && parentLoop->hasAttr(GC_ATTR_KERNEL_NAME) &&
          (!reduction || !linalgOp->hasAttr(GC_ATTR_KERNEL_NAME))) {
        return WalkResult::skip();
      }

      if (auto ti = dyn_cast<TilingInterface>(linalgOp.getOperation())) {
        int64_t numTiles = 0;
        int64_t numIterations = 1;
        for (auto [t, r] :
             zip(ti.getLoopIteratorTypes(), ti.getIterationDomain(b))) {
          if ((t == utils::IteratorType::parallel) == reduction) {
            numTiles++;
            if (auto v = getConstantIntValue(r.size)) {
              numIterations *= *v;
            }
          }
        }
        if (numTiles > 0 && numIterations >= 32) {
          last = ti;
        }
      }

      return WalkResult::skip();
    });
    return last;
  }

  // If a slice inside the loop is created from an external empty tensor and the
  // tensor is not passed to the loop's shared_outs, but referenced directly,
  // replace the slice with an empty tensor of the same size.
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
};
} // namespace
