//===- GlobalAnalysis.cpp - Propagate packing on linalg named ops *- C++-*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "gc/Analysis/GlobalAnalysis.h"

namespace mlir {
namespace gc {

std::ostream &operator<<(std::ostream &ss, const TensorLayout &layout) {
  SmallVector<int64_t> outerAxis = layout.getOuterAxis();
  SmallVector<int64_t> innerAxis = layout.getInnerAxis();
  SmallVector<OpFoldResult> tileSizes = layout.getTileSizes();
  ss << "[";
  for (size_t i = 0; i < outerAxis.size(); ++i) {
    if (i != 0) {
      ss << ", ";
    }
    ss << outerAxis[i];
  }
  for (size_t i = 0; i < innerAxis.size(); ++i) {
    ss << (i == 0 ? "; " : ", ");
    ss << innerAxis[i];
  }
  ss << "]";
  if (!tileSizes.empty()) {
    ss << "; {";
    for (size_t i = 0; i < tileSizes.size(); ++i) {
      if (i != 0) {
        ss << ", ";
      }
      if (getConstantIntValue(tileSizes[i]).has_value()) {
        ss << *getConstantIntValue(tileSizes[i]);
      }
    }
    ss << "}";
  }
  return ss;
}

bool TensorLayout::operator==(const TensorLayout &layout) {
  return (this->OuterAxis == layout.getOuterAxis()) &&
         (this->InnerAxis == layout.getInnerAxis()) &&
         (this->TileSizes == layout.getTileSizes());
}

std::ostream &operator<<(std::ostream &ss, const OperatorLayout &opLayout) {
  ss << "operator has " << opLayout.getSupportedInputLayouts().size()
     << " inputs; " << opLayout.getSupportedOutputLayouts().size()
     << " outputs." << std::endl;
  for (const auto &layout : opLayout.getSupportedInputLayouts()) {
    ss << "input layout: " << layout << std::endl;
  }
  for (const auto &layout : opLayout.getSupportedOutputLayouts()) {
    ss << "output layout: " << layout << std::endl;
  }
  return ss;
}

// inferring the relationship of two indexing map
// j -> i, means j is represented as the same symbol as i
// we don't allow duplicate in symbols
// e.g. if 2 j corresponding to 1 i, then return failure
static FailureOr<DenseMap<int64_t, int64_t>>
inferIndexingMapRelation(AffineMap indexingMapBase,
                         AffineMap indexingMapTarget) {
  DenseMap<int64_t, int64_t> res;
  ArrayRef<AffineExpr> resultsBase = indexingMapBase.getResults();
  ArrayRef<AffineExpr> resultsTarget = indexingMapTarget.getResults();
  for (size_t j = 0; j < resultsTarget.size(); ++j) {
    for (size_t i = 0; i < resultsBase.size(); ++i) {
      auto base = dyn_cast<AffineDimExpr>(resultsBase[i]);
      auto target = dyn_cast<AffineDimExpr>(resultsTarget[j]);
      if (base && target && base.getPosition() == target.getPosition()) {
        if (res.find(j) != res.end())
          return failure();
        res[j] = i;
      }
    }
    if (res.find(j) == res.end())
      res[j] = -1;
  }
  // check res
  DenseSet<int64_t> indexSet;
  for (auto pair : res) {
    if (indexSet.find(pair.second) != indexSet.end()) {
      return failure();
    }
    if (pair.second >= 0) {
      indexSet.insert(pair.second);
    }
  }
  return res;
}

// given j --> i and max rank of i, return i --> j
static DenseMap<int64_t, int64_t>
getReversedIndexMap(const DenseMap<int64_t, int64_t> &indexMap,
                    size_t maxRank) {
  DenseMap<int64_t, int64_t> res;
  for (auto pair : indexMap) {
    if (pair.second >= 0) {
      res[pair.second] = pair.first;
    }
  }
  for (size_t i = 0; i < maxRank; ++i) {
    if (res.find(i) == res.end()) {
      res[i] = -1;
    }
  }
  return res;
}

static FailureOr<TensorLayout>
inferTargetLayout(TensorLayout layoutBase,
                  const DenseMap<int64_t, int64_t> &indexMap) {
  int64_t dimDifference = indexMap.size() - layoutBase.getTensorRank();
  SmallVector<int64_t> baseOuterAxis = layoutBase.getOuterAxis();
  SmallVector<int64_t> baseInnerAxis = layoutBase.getInnerAxis();
  SmallVector<OpFoldResult> baseTileSizes = layoutBase.getTileSizes();
  SmallVector<int64_t> targetOuterAxis;
  SmallVector<int64_t> targetInnerAxis;
  SmallVector<OpFoldResult> targetTileSizes;
  DenseMap<int64_t, int64_t> reverseIndexMap =
      getReversedIndexMap(indexMap, layoutBase.getTensorRank());
  for (auto oa : baseOuterAxis) {
    if (reverseIndexMap[oa] >= 0) {
      targetOuterAxis.push_back(reverseIndexMap[oa]);
    }
  }
  // filling up new j axes
  SmallVector<int64_t> newDimAxis;
  for (auto pair : indexMap) {
    if (pair.second < 0) {
      newDimAxis.push_back(pair.first);
    }
  }
  targetOuterAxis.insert(targetOuterAxis.begin(), newDimAxis.begin(),
                         newDimAxis.end());
  for (auto &&[ia, ts] : llvm::zip(baseInnerAxis, baseTileSizes)) {
    if (reverseIndexMap[ia] >= 0) {
      targetInnerAxis.push_back(reverseIndexMap[ia]);
      targetTileSizes.push_back(ts);
    }
  }
  return TensorLayout(targetOuterAxis, targetInnerAxis, targetTileSizes);
}

GlobalAnalysis::GlobalAnalysis(Operation *root) {
  root->walk([&](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // get input layouts
      std::cout << std::endl;
      std::cout << "----------------------------------" << std::endl;
      linalgOp.getOperation()->getName().print(llvm::errs());
      std::cout << std::endl;
      std::cout << "----------------------------------" << std::endl;
      std::cout << std::endl;
      SmallVector<AffineMap> indexing_maps = linalgOp.getIndexingMapsArray();
      auto curInputs = linalgOp.getDpsInputOperands();
      auto curResults = linalgOp.getOperation()->getResults();

      // ---------------- Get Current Input Layouts -------------------
      // get current input layouts
      std::cout << "----- printing ground-truth input layouts -----"
                << std::endl;
      SmallVector<TensorLayout> curInputLayouts;
      for (auto input : curInputs) {
        auto parent = input->get().getDefiningOp();
        if (layout.find(parent) != layout.end()) {
          // TODO(yifei): it is not always 0 here
          curInputLayouts.push_back(layout[parent].getOutputLayout(0));
        } else {
          curInputLayouts.push_back(TensorLayout::createPlainLayout(
              linalgOp.getMatchingIndexingMap(input).getNumResults()));
        }
      }
      // debug info
      for (auto layout : curInputLayouts) {
        std::cout << "layout: " << layout << std::endl;
      }

      // ------ Get Current Op's Suggested Layout & Do Propagation ------
      IRRewriter rewriter(linalgOp);
      if (mlir::linalg::isaContractionOpInterface(linalgOp)) {
        // query the cost model
        // OperatorLayout suggestedLayout = costModel->queryLayout(linalgOp,
        // curInputLayouts);

        // hardcode one for now
        // A side layout, [0, 1, 0, 1]; {32, 32}
        TensorLayout A_layout(
            {0, 1}, {0, 1},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(32),
                                      rewriter.getIndexAttr(32)});
        // B side layout, [1, 0, 0, 1]; {32, 32}
        TensorLayout B_layout(
            {1, 0}, {0, 1},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(32),
                                      rewriter.getIndexAttr(32)});
        // C side layout, [0, 1, 0, 1]; {32, 32}
        TensorLayout C_layout(
            {0, 1}, {0, 1},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(32),
                                      rewriter.getIndexAttr(32)});
        OperatorLayout suggestedLayout({A_layout, B_layout}, {C_layout});
        layout[linalgOp] = suggestedLayout;
      } else {
        SmallVector<TensorLayout> inputLayouts, outputLayouts;
        inputLayouts.push_back(curInputLayouts[0]);
        // TODO(yifei): wisely choose the input format basis
        // Let's only refer to input[0] for now
        for (size_t i = 1; i < curInputs.size(); ++i) {
          std::cout << "inferring indexing map relation" << std::endl;
          // getMatchingIndexingMap
          auto res = inferIndexingMapRelation(
              linalgOp.getMatchingIndexingMap(curInputs[0]),
              linalgOp.getMatchingIndexingMap(curInputs[i]));
          for (auto tp : *res) {
            std::cout << "target index: " << tp.first
                      << " maps to base index: " << tp.second << std::endl;
          }
          TensorLayout inputLayout =
              *inferTargetLayout(curInputLayouts[0], *res);
          inputLayouts.push_back(inputLayout);
        }
        auto res_out = inferIndexingMapRelation(
            linalgOp.getMatchingIndexingMap(curInputs[0]),
            linalgOp.getIndexingMapMatchingResult(curResults[0]));
        TensorLayout outputLayout =
            *inferTargetLayout(curInputLayouts[0], *res_out);
        outputLayouts.push_back(outputLayout);
        for (auto tp : *res_out) {
          std::cout << "target index: " << tp.first
                    << " maps to base index: " << tp.second << std::endl;
        }
        OperatorLayout suggestedLayout(inputLayouts, outputLayouts);
        layout[linalgOp] = suggestedLayout;
      }
    } else if (isa<tensor::PadOp>(op) || isa<tensor::ExpandShapeOp>(op)) {
    }
  });
}
} // namespace gc
} // namespace mlir
