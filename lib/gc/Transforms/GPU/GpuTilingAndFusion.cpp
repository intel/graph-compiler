//===-- GpuTilingAndFusion.cpp - DESC ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
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

#include "./GpuUtils.h"
#include "gc/Utils/Log.h"

using namespace mlir;
// using namespace mlir::gc::gpu;

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
      : GpuTilingAndFusion(gc::GpuTilingAndFusionOptions{}) {}
  explicit GpuTilingAndFusion(const gc::GpuTilingAndFusionOptions &opts)
      : GpuPass(), GpuTilingAndFusionBase(opts) {}

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    scf::SCFTileAndFuseOptions opts;
    opts.setFusionControlFn(
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          Operation *op = originalProducer.getOwner();
          if (!op) {
            return std::nullopt;
          }
          if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
            if (!linalgOp.hasOnlyProjectedPermutations()) {
              return std::nullopt;
            }
          }
          return scf::SCFTileAndFuseOptions::ControlFnResult{};
        });
    opts.tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    // The outer loop is converted to a GPU kernel and the tile sizes are mapped
    // to the grid sizes.
    opts.tilingOptions.setTileSizeComputationFunction(
        // The tile sizes calculation is based on the following equation:
        // n * TS0 * TS1 * ... * TSn = euMem
        // where:
        // n - an average number of bytes, processed by each iteration
        // TS0, TS1, ... TSn - the tile sizes for each loop correspondingly
        // euMem - the physical memory (cache) size of the GPU execution unit
        //
        // To calculate the tile size TS, we need to divide the total loop size
        // S by the ratio r:
        //
        // n * (S0/r0) * (S1/r1) * ... * (Sn/rn) = euMem
        // r0 * r1 * ... * rn = (n * S0 * S1 * ... * Sn) / euMem
        // If all sizes are equal, then S0 = ... = Sn = S, r0 = ... = rn = r:
        // r^n = (n * S^n) / euMem
        // r = (n * S^n / euMem)^(1/n)
        [euMem = getEuMem(rewriter), euThreads = getEuThreads(rewriter)](
            OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
          auto ti = dyn_cast<TilingInterface>(op);
          if (!ti) {
            return {};
          }

          auto itTypes = ti.getLoopIteratorTypes();
          auto itDomains = ti.getIterationDomain(builder);
          assert(itTypes.size() == itDomains.size());

          // TODO: Add a parameter to the options?
          size_t totalSize = calcOperandsSize(op);
          unsigned loopCount = 0;
          SmallVector<int64_t> sizes;

          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t == utils::IteratorType::parallel) {
              if (auto v = getConstantIntValue(r.size)) {
                loopCount++;
                sizes.emplace_back(*v);
                totalSize *= *v;
              } else {
                return calcDynamicSizes(builder, ti, euMem, euThreads);
              }
            }
          }

          if (loopCount == 0) {
            return {};
          }

          auto outerTileSize = static_cast<size_t>(
              std::ceil(static_cast<double>(euMem) /
                        static_cast<double>(calcOperandsSize(op))));
          SmallVector<int64_t> outerTiles;
          SmallVector<int64_t> innerTiles;
          normaliseTiles(outerTileSize, sizes, outerTiles);
          normaliseTiles(euThreads, sizes, innerTiles);

          unsigned counter = 0;
          SmallVector<OpFoldResult> tiles;
          tiles.reserve(itDomains.size());

          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t != utils::IteratorType::parallel) {
              tiles.emplace_back(builder.getIndexAttr(1));
            } else if (auto v = getConstantIntValue(r.size)) {
              tiles.emplace_back(
                  ceil(builder, outerTiles[counter], innerTiles[counter]));
              counter++;
            } else {
              abort(); // Must never get here
            }
          }

          return tiles;
        });

    auto fn = getOperation();
    tileAndFuse(fn, rewriter, opts);
  }

private:
  static void tileAndFuse(Operation *op, RewriterBase &rewriter,
                          const scf::SCFTileAndFuseOptions &opts) {
    for (auto ti = findTi(op); ti; ti = findTi(op)) {
      auto result =
          scf::tileConsumerAndFuseProducersUsingSCF(rewriter, *ti, opts);

      if (failed(result)) {
        ti->emitError() << "Failed to tile and fuse using SCF";
        return;
      }

      SmallVector<Operation *> opsToReplace{ti->getOperation()};
      append_range(opsToReplace, result->fusedProducers);
      for (Operation *toReplace : opsToReplace) {
        if (toReplace->getParentOp() == nullptr) {
          continue;
        }

        for (OpResult res : toReplace->getResults()) {
          if (auto repl = result->replacements.lookup(res)) {
            rewriter.replaceAllUsesWith(res, repl);
          }
        }

        if (failed(simplifyRegions(rewriter, op->getRegions()))) {
          gcLogD("Failed to simplify regions");
        }

        if (toReplace->getParentOp() == nullptr) {
          continue;
        }

        // For some reason (probably a bug?) the operation could be
        // referenced by a dead code inside the replacement, that prevents
        // this operation from being erased. Erasing the dead code first.
        for (auto u : toReplace->getUsers()) {
          if (u->use_empty()) {
            rewriter.eraseOp(u);
          }
        }

        if (toReplace->use_empty()) {
          rewriter.eraseOp(toReplace);
        } else {
          gcLogE("Unable to erase operation!");
        }
      }
    }
  }

  static std::optional<TilingInterface> findTi(Operation *op) {
    std::optional<TilingInterface> last;
    op->walk<WalkOrder::PreOrder>([&](linalg::LinalgOp linalgOp) {
      if (linalgOp.hasOnlyProjectedPermutations() &&
          !linalgOp->getParentOfType<scf::ForallOp>()) {
        if (auto ti = dyn_cast<TilingInterface>(linalgOp.getOperation())) {
          last = ti;
        }
      }
      return WalkResult::skip();
    });
    return last;
  }

  static SmallVector<OpFoldResult> calcDynamicSizes(OpBuilder &builder,
                                                    TilingInterface ti,
                                                    size_t euMem,
                                                    size_t euThreads) {
    auto itTypes = ti.getLoopIteratorTypes();
    auto itDomains = ti.getIterationDomain(builder);
    assert(itTypes.size() == itDomains.size());

    auto loc = ti.getLoc();
    Value dynamicSize;
    size_t staticSize = calcOperandsSize(ti.getOperation()) * euThreads;
    unsigned loopCount = 0;

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        continue;
      }
      loopCount++;
      if (auto v = getConstantIntValue(r.size)) {
        staticSize *= *v;
      } else if (dynamicSize) {
        dynamicSize = builder.create<arith::MulIOp>(loc, dynamicSize,
                                                    r.size.get<Value>());
      } else {
        dynamicSize = r.size.get<Value>();
      }
    }

    assert(loopCount);
    assert(dynamicSize);
    if (staticSize > 1) {
      dynamicSize = builder.create<arith::MulIOp>(
          loc, dynamicSize,
          builder.create<arith::ConstantIndexOp>(loc, staticSize));
    }
    dynamicSize = builder.create<arith::UIToFPOp>(
        loc, builder.getF64Type(),
        builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                           dynamicSize));

    auto memSize = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(static_cast<double>(euMem)), builder.getF64Type());
    auto pow = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(1.0 / loopCount), builder.getF64Type());
    Value ratio = builder.create<math::PowFOp>(
        loc, builder.create<arith::DivFOp>(loc, dynamicSize, memSize), pow);
    ratio = builder.create<arith::MaximumFOp>(
        loc, builder.getF64Type(),
        builder.create<arith::ConstantFloatOp>(loc, APFloat(1.0),
                                               builder.getF64Type()),
        ratio);

    SmallVector<OpFoldResult> tiles;
    tiles.reserve(itDomains.size());

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        tiles.emplace_back(builder.getIndexAttr(1));
      } else {
        Value value;
        if (auto v = getConstantIntValue(r.size)) {
          value = builder.create<arith::ConstantFloatOp>(
              loc, APFloat(static_cast<double>(*v)), builder.getF64Type());
        } else {
          value = builder.create<arith::UIToFPOp>(
              loc, builder.getF64Type(),
              builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                 r.size.get<Value>()));
        }
        auto ts = builder.create<arith::FPToUIOp>(
            loc, builder.getI64Type(),
            builder.create<math::CeilOp>(
                loc, builder.create<arith::DivFOp>(loc, value, ratio)));
        tiles.emplace_back(builder.create<arith::IndexCastOp>(
            loc, builder.getIndexType(), ts));
      }
    }

    return tiles;
  }

  static size_t calcOperandsSize(Operation *op) {
    size_t size = 0;
    auto typeSize = [](Type t) -> size_t {
      Type et;
      if (auto mt = dyn_cast<MemRefType>(t)) {
        et = mt.getElementType();
      } else if (auto tt = dyn_cast<TensorType>(t)) {
        et = tt.getElementType();
      } else {
        return 0;
      }
      return et.isIntOrFloat() ? et.getIntOrFloatBitWidth() / 8 : 1;
    };
    for (auto operand : op->getOperands()) {
      if (auto defOp = operand.getDefiningOp()) {
        for (auto t : defOp->getResultTypes()) {
          size += typeSize(t);
        }
      } else {
        size += typeSize(operand.getType());
      }
    }
    return size == 0 ? 1 : size;
  }
};
} // namespace
