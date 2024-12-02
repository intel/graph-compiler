//===-- GpuTilingAndFusion.cpp - DESC ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./GpuUtils.h"
#include "gc/Dialect/Linalgx/Utils.h"

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
    : GpuPass<GpuTilingAndFusion>,
      gc::impl::GpuTilingAndFusionBase<GpuTilingAndFusion> {
  friend struct GpuPass;
  explicit GpuTilingAndFusion()
      : GpuTilingAndFusion(GpuTilingAndFusionOptions{}) {}
  explicit GpuTilingAndFusion(const GpuTilingAndFusionOptions &opts)
      : GpuPass(), GpuTilingAndFusionBase(opts) {}

  void runOnOperation() override {
    auto fn = getOperation();
    if (fn.isExternal()) {
      return;
    }

    OpRewriter rw(fn);
    auto loopMarker = rw.getStringAttr("gcGpuLoop");
    tileAndFuseLinalgOps(rw, fn, loopMarker);
    tileForallOps(rw, fn, loopMarker);
  }

private:
  void tileAndFuseLinalgOps(OpRewriter &rw, func::FuncOp &fn,
                            StringAttr &loopMarker) {
    auto markerValue = rw.getBoolAttr(true);
    auto numEus = getNumEus(rw);
    auto numEusPerSlice = getNumEusPerSlice(rw);
    auto numThreadsPerEu = getNumThreadsPerEu(rw);
    auto localMemSize = getLocalMemSize(rw);
    auto vectorWidth = getVectorWidth(rw);
    auto cachePerThread =
        std::max(localMemSize / numEusPerSlice / numThreadsPerEu, vectorWidth);
    SCFTileAndFuseOptions opts;
    opts.tilingOptions.setTileSizeComputationFunction(
        [&rw, cachePerThread, vectorWidth,
         numThreads = numEus * numThreadsPerEu](
            OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
          auto ti = dyn_cast<TilingInterface>(op);
          if (!ti) {
            return {};
          }

          rw.loc = op->getLoc();
          rw.setInsertionPoint(op);
          auto itTypes = ti.getLoopIteratorTypes();
          auto itDomains = ti.getIterationDomain(builder);
          assert(itTypes.size() == itDomains.size());

          SmallVector<int64_t> sizes;
          int64_t maxSize = 0;
          int64_t numIterations = 1;
          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t == utils::IteratorType::parallel) {
              if (auto v = getConstantIntValue(r.size)) {
                numIterations *= *v;
                sizes.emplace_back(*v);
                maxSize = std::max(maxSize, *v);
              } else {
                return computeDynamicTiles(rw, ti, numThreads, cachePerThread);
              }
            }
          }

          assert(!sizes.empty());
          auto elementSize = getElementSize(op);
          auto sizePerThread = numIterations / numThreads * elementSize;
          auto totalSize = std::max(sizePerThread, cachePerThread);
          totalSize = std::max(totalSize / elementSize, 64L);
          bool xeGpu = canLowerToXeGPU(op);

          // If the operation could be lowered to XeGPU, make the tiles
          // multiple of the vector width.
          if (xeGpu) {
            totalSize = std::max(totalSize / vectorWidth, 1L) * vectorWidth;
          }

          SmallVector<int64_t> tiles = sizes;
          adjustTiles(totalSize, tiles, xeGpu);

          // If the tiles are equal to the sizes, split the largest tile
          // to avoid loops elimination by the canonicalizer pass.
          if (tiles == sizes) {
            auto tile = findFactor(maxSize, maxSize / 2);

            if (tile == maxSize) {
              // Find another size, that can be split
              auto another = maxSize;
              sort(sizes, std::greater<>());
              for (auto s : sizes) {
                if (s != maxSize && (tile = findFactor(s, s / 2)) != s) {
                  another = s;
                  break;
                }
              }
              if (another == maxSize) {
                tile = 1;
                // Find the smallest size that is not 1
                for (auto s : reverse(sizes)) {
                  if (s != 1) {
                    maxSize = s;
                    break;
                  }
                }
              } else {
                maxSize = another;
              }
            }

            for (auto &t : tiles) {
              if (t == maxSize) {
                t = tile;
                break;
              }
            }
          }

          unsigned counter = 0;
          SmallVector<OpFoldResult> result;
          result.reserve(itDomains.size());

          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t == utils::IteratorType::parallel) {
              result.emplace_back(rw.createConstant(tiles[counter++]));
            } else {
              result.emplace_back(rw.createConstant(0L));
            }
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
          }

          // If the result of this slice is used by a MatmulOp and the slice has
          // an operand produced by a previous MatmulOp, do not fuse.
          if (isOpDependsOnResult<0>(linalgx::isMatmulOp, candidateSliceOp) &&
              isOperandDependsOnOp(linalgx::isMatmulOp, candidateSliceOp)) {
            return std::nullopt;
          }

          return SCFTileAndFuseOptions::ControlFnResult{};
        });
    opts.tilingOptions.setLoopType(SCFTilingOptions::LoopType::ForallOp);

    for (auto ti = findTi(rw, fn, loopMarker); ti;
         ti = findTi(rw, fn, loopMarker)) {
      auto result = tileConsumerAndFuseProducersUsingSCF(rw, *ti, opts);

      if (failed(result)) {
        ti->emitError() << "Failed to tile and fuse using SCF";
        return;
      }

      SmallVector opsToReplace{ti->getOperation()};
      append_range(opsToReplace, result->fusedProducers);
      for (Operation *toReplace : opsToReplace) {
        for (OpResult res : toReplace->getResults()) {
          if (auto repl = result->replacements.lookup(res)) {
            rw.replaceAllUsesWith(res, repl);
            if (auto loop = dyn_cast<ForallOp>(repl.getDefiningOp())) {
              loop->setAttr(loopMarker, markerValue);
            }
          }
        }
      }

      if (failed(simplifyRegions(rw, fn->getRegions()))) {
        // Not simplified
      }
    }
  }

  static std::optional<TilingInterface> findTi(OpBuilder &b, Operation *op,
                                               const StringAttr &loopMarker) {
    std::optional<TilingInterface> last;
    op->walk<WalkOrder::PreOrder>([&](linalg::LinalgOp linalgOp) {
      if (!linalgOp.hasOnlyProjectedPermutations()) {
        return WalkResult::skip();
      }
      if (auto parentLoop = linalgOp->getParentOfType<ForallOp>();
          parentLoop && parentLoop->hasAttr(loopMarker)) {
        return WalkResult::skip();
      }

      if (auto ti = dyn_cast<TilingInterface>(linalgOp.getOperation())) {
        int64_t numTiles = 0;
        int64_t numIterations = 1;
        for (auto [t, r] :
             zip(ti.getLoopIteratorTypes(), ti.getIterationDomain(b))) {
          if (t == utils::IteratorType::parallel) {
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

  static SmallVector<OpFoldResult> computeDynamicTiles(OpRewriter &rw,
                                                       TilingInterface ti,
                                                       int64_t numThreads,
                                                       int64_t cachePerThread) {
    auto itTypes = ti.getLoopIteratorTypes();
    auto itDomains = ti.getIterationDomain(rw);
    assert(itTypes.size() == itDomains.size());
    rw.loc = ti.getLoc();
    rw.setInsertionPoint(ti.getOperation());

    Value dynamicSize;
    auto staticSize = getElementSize(ti.getOperation());
    unsigned loopCount = 0;

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        continue;
      }
      loopCount++;
      if (auto v = getConstantIntValue(r.size)) {
        staticSize *= *v;
      } else if (dynamicSize) {
        dynamicSize =
            rw.create<arith::MulIOp>(dynamicSize, r.size.get<Value>());
      } else {
        dynamicSize = r.size.get<Value>();
      }
    }

    assert(loopCount);
    assert(dynamicSize);
    if (staticSize > 1) {
      dynamicSize =
          rw.create<arith::MulIOp>(dynamicSize, rw.createConstant(staticSize));
    }
    auto i64Type = rw.getI64Type();
    dynamicSize = rw.create<arith::UIToFPOp>(
        rw.getF64Type(), rw.create<arith::IndexCastOp>(i64Type, dynamicSize));

    // TODO: Call the adjustTiles() function for the tiles calculation.

    auto nt = rw.createConstant(static_cast<double>(numThreads));
    auto cpt = rw.createConstant(static_cast<double>(cachePerThread));
    Value totalSize = rw.create<arith::MaximumFOp>(
        rw.getF64Type(), rw.create<arith::DivFOp>(dynamicSize, nt), cpt);
    auto pow = rw.createConstant(1.0 / loopCount);
    // The average tile size is totalSize^(1 / loopCount)
    Value avgTileSize = rw.create<math::PowFOp>(totalSize, pow);
    avgTileSize = rw.create<arith::MaximumFOp>(
        rw.getF64Type(), rw.createConstant(1.0), avgTileSize);
    avgTileSize = rw.create<arith::FPToSIOp>(i64Type, avgTileSize);

    SmallVector<OpFoldResult> tiles;
    tiles.reserve(itDomains.size());

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        tiles.emplace_back(rw.getIndexAttr(1));
      } else {
        Value value;
        if (auto v = getConstantIntValue(r.size)) {
          value = rw.create<arith::ConstantIntOp>(*v, i64Type);
        } else {
          value = rw.create<arith::IndexCastOp>(i64Type, r.size.get<Value>());
        }
        value = rw.create<arith::MinSIOp>(i64Type, value, avgTileSize);
        tiles.emplace_back(
            rw.create<arith::IndexCastOp>(rw.getIndexType(), value));
      }
    }

    return tiles;
  }

  static int64_t getElementSize(Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (auto inits = linalgOp.getDpsInits(); !inits.empty()) {
        if (auto t = getElementTypeOrSelf(inits[0].getType());
            t.isIntOrFloat()) {
          return std::max(1L, t.getIntOrFloatBitWidth() / 8L);
        }
      }
    }
    return 1L;
  }

  // TODO: Add more checks
  static bool canLowerToXeGPU(Operation *operation) {
    auto op = dyn_cast<linalg::LinalgOp>(operation);
    if (!op) {
      return false;
    }
    if (op.hasDynamicShape()) {
      return false;
    }

    auto checkOperand = [&](Value operand, bool isOutput = false) {
      ShapedType type;
      if (auto memref = dyn_cast<MemRefType>(operand.getType())) {
        type = memref;
      } else if (auto tensor = dyn_cast<RankedTensorType>(operand.getType())) {
        type = tensor;
      } else {
        return false;
      }

      if (auto shape = type.getShape(); shape.size() >= 2) {
        return !isOutput ||
               std::accumulate(shape.begin() + 1, shape.end(), shape[0],
                               std::multiplies<>()) >= 16;
      }
      return false;
    };

    if (auto inits = op.getDpsInits();
        !inits.empty() && !checkOperand(inits[0], true)) {
      return false;
    }

    if (auto inputs = op.getDpsInputs();
        !std::all_of(inputs.begin(), inputs.end(),
                     [&](Value v) { return checkOperand(v); })) {
      return false;
    }

    return true;
  }

  void tileForallOps(OpRewriter &rw, func::FuncOp &fn, StringAttr &loopMarker) {
    auto wgSize = getWorkGroupSize(rw);
    fn.walk<WalkOrder::PreOrder>([&rw, wgSize, loopMarker](ForallOp loop) {
      if (loop->removeAttr(loopMarker)) {
        replaceEmptySlices(rw, loop);

        // If there is only one user, and it's located in a different block,
        // and this block is not inside a loop, move the loop to the user block.
        if (loop->hasOneUse()) {
          auto user = *loop->getUsers().begin();
          if (user->getBlock() != loop->getBlock()) {
            if (!user->getParentOfType<LoopLikeOpInterface>()) {
              loop->moveBefore(user);
            }
          }
        }

        tileForallOp(rw, loop, wgSize);
      }
      return WalkResult::skip();
    });
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

  static void tileForallOp(OpRewriter &rw, ForallOp op, int64_t wgSize) {
    rw.loc = op.getLoc();
    rw.setInsertionPoint(op);
    OpFoldResult zero{rw.createConstant(0L)};
    OpFoldResult one{rw.createConstant(1L)};
    auto innerSteps = op.getMixedStep();
    auto outerBounds = op.getMixedUpperBound();
    SmallVector<OpFoldResult> outerSteps;
    auto count = innerSteps.size();

    { // Calculate outer steps
      SmallVector<int64_t> tiles;
      tiles.reserve(count);
      for (auto s : innerSteps) {
        if (auto v = getConstantIntValue(s)) {
          tiles.emplace_back(*v);
        } else {
          // TODO: Add support for dynamic sizes
          tiles.emplace_back(32);
        }
      }
      adjustTiles(wgSize, tiles);
      outerSteps.reserve(count);
      for (auto [s, b, t] : zip(innerSteps, outerBounds, tiles)) {
        if (auto sv = getConstantIntValue(s)) {
          auto step = *sv * t;
          if (auto bv = getConstantIntValue(b)) {
            step = std::min(step, *bv);
          }
          outerSteps.emplace_back(rw.createConstant(step));
        } else {
          outerSteps.emplace_back(
              rw.create<arith::MulIOp>(s.get<Value>(), rw.createConstant(t)));
        }
      }
    }

    auto outerLoop =
        rw.create<ForallOp>(op.getMixedLowerBound(), outerBounds, outerSteps,
                            op.getOutputs(), std::nullopt);
    rw.setInsertionPointToStart(outerLoop.getBody());
    SmallVector<OpFoldResult> innerBounds;
    SmallVector<Range> ranges;

    {
      auto idxType = rw.getIndexType();
      auto ctx = rw.getContext();
      auto minMap = AffineMap::get(
          /*dimCount=*/3, /*symbolCount=*/0,
          {getAffineDimExpr(0, ctx),
           getAffineDimExpr(1, ctx) - getAffineDimExpr(2, ctx)},
          rw.getContext());
      innerBounds.reserve(count);
      ranges.reserve(count);
      for (auto [i, u, s] : zip(outerLoop.getInductionVars(),
                                outerLoop.getMixedUpperBound(), outerSteps)) {
        OpFoldResult iub;
        auto cu = getConstantIntValue(u);
        auto cs = getConstantIntValue(s);
        if (cu && cs && (*cu % *cs == 0)) {
          iub = s;
        } else {
          Value vub = cu ? rw.createConstant(*cu) : u.get<Value>();
          Value vs = cs ? rw.createConstant(*cs) : s.get<Value>();
          iub = OpFoldResult(rw.create<affine::AffineMinOp>(
              idxType, minMap, ValueRange{vs, vub, i}));
        }
        innerBounds.emplace_back(iub);
        ranges.emplace_back(Range{i, iub, one});
      }
    }

    SmallVector<Value> innerOutputs;
    for (auto o : outerLoop.getRegionIterArgs()) {
      innerOutputs.emplace_back(rw.create<tensor::ExtractSliceOp>(o, ranges));
    }

    auto innerLoop =
        rw.create<ForallOp>(SmallVector(count, zero), innerBounds, innerSteps,
                            innerOutputs, op.getMapping());
    SmallVector<Type> argTypes{innerLoop.getBody()->getArgumentTypes()};
    innerLoop.getRegion().takeBody(op.getRegion());
    for (auto [arg, type] :
         zip(innerLoop.getBody()->getArguments(), argTypes)) {
      arg.setType(type);
    }

    // Collect all users of the inner loop outputs
    llvm::SmallSet<Operation *, 4> outUsers;
    for (auto out : innerLoop.getRegionIterArgs()) {
      for (auto user : out.getUsers()) {
        outUsers.insert(user);
      }
    }

    // Replace the induction variables of the inner loop with the sum of the
    // outer and inner induction variables, but only in the operations, that
    // are not using the inner loop outputs, which are already sliced.
    rw.setInsertionPointToStart(innerLoop.getBody());
    for (auto [inIdx, outIdx] :
         zip(innerLoop.getInductionVars(), outerLoop.getInductionVars())) {
      auto newIdx = rw.create<arith::AddIOp>(inIdx, outIdx);
      outUsers.insert(newIdx);
      inIdx.replaceAllUsesExcept(newIdx, outUsers);
    }

    rw.setInsertionPointToStart(outerLoop.getTerminator().getBody());
    for (auto [i, o] :
         zip(innerLoop.getResults(), outerLoop.getRegionIterArgs())) {
      rw.create<tensor::ParallelInsertSliceOp>(i, o, ranges);
    }

    rw.replaceOp(op, outerLoop);
  }
};
} // namespace
