//===- GlobalAnalysis.cpp - Propagate packing on linalg named ops *- C++-*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <numeric>

#include "gc/Analysis/GlobalAnalysis.h"
#include "gc/Analysis/MatmulConfigAnalysis.h"

namespace mlir {
namespace gc {

#define DEBUG_TYPE "global-analysis"

llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                              const TensorLayout &layoutCache) {
  SmallVector<int64_t> outerAxis = layoutCache.getOuterAxis();
  SmallVector<int64_t> innerAxis = layoutCache.getInnerAxis();
  SmallVector<OpFoldResult> tileSizes = layoutCache.getTileSizes();
  ss << "[";
  llvm::interleaveComma(outerAxis, ss);
  if (!innerAxis.empty()) {
    ss << "; ";
    llvm::interleaveComma(innerAxis, ss);
  }
  ss << "]";
  if (!tileSizes.empty()) {
    ss << "; {";
    llvm::interleaveComma(tileSizes, ss);
    ss << "}";
  }
  return ss;
}

bool TensorLayout::operator==(const TensorLayout &layout) {
  return (this->outerAxis == layout.getOuterAxis()) &&
         (this->innerAxis == layout.getInnerAxis()) &&
         (this->tileSizes == layout.getTileSizes());
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                              const OperatorLayout &opLayout) {
  for (auto &&[idx, layoutCache] :
       llvm::enumerate(opLayout.getSupportedInputLayouts())) {
    ss << "input " << idx << "'s layout: " << layoutCache << "\n";
  }
  for (auto &&[idx, layoutCache] :
       llvm::enumerate(opLayout.getSupportedOutputLayouts())) {
    ss << "output " << idx << "'s layout: " << layoutCache << "\n";
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
  SmallVector<int64_t> baseOuterAxis = layoutBase.getOuterAxis();
  SmallVector<int64_t> baseInnerAxis = layoutBase.getInnerAxis();
  SmallVector<OpFoldResult> baseTileSizes = layoutBase.getTileSizes();
  SmallVector<int64_t> targetOuterAxis;
  SmallVector<int64_t> targetInnerAxis;
  SmallVector<OpFoldResult> targetTileSizes;
  DenseMap<int64_t, int64_t> reverseIndexMap =
      getReversedIndexMap(indexMap, layoutBase.getRank());
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

static size_t getTargetInputIdx(ArrayRef<TensorLayout> curInputLayouts) {
  for (size_t i = 0; i < curInputLayouts.size(); ++i) {
    if (!curInputLayouts[i].isPlainLayout()) {
      return i;
    }
  }
  return 0;
}

static bool supportedContractionNamedOpList(linalg::LinalgOp &linalgOp) {
  if (isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
          linalg::MatmulTransposeBOp, linalg::BatchMatmulOp,
          linalg::BatchMatmulTransposeAOp, linalg::BatchMatmulTransposeBOp>(
          linalgOp))
    return true;
  return false;
}

std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
getPackingAxis(int64_t numRank, bool transposed) {
  assert(numRank >= 2 &&
         "The rank of matmul semantic contraction op shall be at least 2.");
  SmallVector<int64_t> outerAxisPerm(numRank);
  SmallVector<int64_t> innerAxisPos(2);
  std::iota(outerAxisPerm.begin(), outerAxisPerm.end(), 0);
  innerAxisPos[0] = numRank - 2;
  innerAxisPos[1] = numRank - 1;
  if (transposed) {
    std::swap(outerAxisPerm[numRank - 2], outerAxisPerm[numRank - 1]);
    std::swap(innerAxisPos[0], innerAxisPos[1]);
  }
  return std::make_pair(outerAxisPerm, innerAxisPos);
}

GlobalAnalysis::GlobalAnalysis(Operation *root) {
  root->walk([&](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Inferring layout of op: " << op->getName() << "\n");
      auto curInputs = linalgOp.getDpsInputOperands();
      auto curResults = linalgOp.getOperation()->getResults();
      // ---------------- Get Current Input Layouts -------------------
      SmallVector<TensorLayout> curInputLayouts;
      for (auto input : curInputs) {
        auto parent = input->get().getDefiningOp();
        if (layoutCache.find(parent) != layoutCache.end()) {
          // TODO(yifei): it is not always 0 here
          curInputLayouts.push_back(layoutCache[parent].getOutputLayout(0));
        } else {
          curInputLayouts.push_back(TensorLayout::createPlainLayout(
              linalgOp.getMatchingIndexingMap(input).getNumResults()));
        }
      }
      // ------ Get Current Op's Suggested Layout & Do Propagation ------
      IRRewriter rewriter(linalgOp);
      // TODO: extend to packed/vnni matmul ops
      if (supportedContractionNamedOpList(linalgOp)) {
        // get input and output rank
        auto ARank = cast<ShapedType>(linalgOp.getDpsInputs()[0].getType())
                         .getShape()
                         .size();
        auto BRank = cast<ShapedType>(linalgOp.getDpsInputs()[1].getType())
                         .getShape()
                         .size();
        auto CRank =
            cast<ShapedType>(linalgOp.getOperation()->getResults()[0].getType())
                .getShape()
                .size();
        bool ASideTransposed =
            isa<linalg::MatmulTransposeAOp, linalg::BatchMatmulTransposeAOp>(
                linalgOp);
        bool BSideTransposed =
            isa<linalg::MatmulTransposeBOp, linalg::BatchMatmulTransposeBOp>(
                linalgOp);
        // set outer&inner axis values
        auto APackInfo = getPackingAxis(ARank, ASideTransposed);
        auto BPackInfo = getPackingAxis(BRank, BSideTransposed);
        auto CPackInfo = getPackingAxis(CRank, /*transposed*/ false);
        // query the cost model for tile sizes
        MatmulConfig cfg =
            MatmulConfigAnalysis(linalgOp.getOperation()).getConfig();
        uint32_t iim = cfg.innerMostKBlock, iin = cfg.innerMostNBlock,
                 iik = cfg.innerMostKBlock;
        // current layout is MKmk, NKkn, MNmn
        TensorLayout ALayout(
            APackInfo.first, APackInfo.second,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(iim),
                                      rewriter.getIndexAttr(iik)});
        TensorLayout BLayout(
            BPackInfo.first, BPackInfo.second,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(iik),
                                      rewriter.getIndexAttr(iin)});
        TensorLayout CLayout(
            CPackInfo.first, CPackInfo.second,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(iim),
                                      rewriter.getIndexAttr(iin)});
        OperatorLayout suggestedLayout({ALayout, BLayout}, {CLayout});
        layoutCache[linalgOp] = suggestedLayout;
      } else if (!mlir::linalg::isaContractionOpInterface(linalgOp) &&
                 !supportedContractionNamedOpList(linalgOp)) {
        SmallVector<TensorLayout> inputLayouts, outputLayouts;
        size_t targetIdx = getTargetInputIdx(curInputLayouts);
        // TODO(yifei): wisely choose the input format basis
        // Let's only refer to input[0] for now
        for (size_t i = 0; i < curInputs.size(); ++i) {
          // getMatchingIndexingMap
          if (i != targetIdx) {
            auto res = inferIndexingMapRelation(
                linalgOp.getMatchingIndexingMap(curInputs[targetIdx]),
                linalgOp.getMatchingIndexingMap(curInputs[i]));
            TensorLayout inputLayout =
                *inferTargetLayout(curInputLayouts[targetIdx], *res);
            inputLayouts.push_back(inputLayout);
          } else {
            inputLayouts.push_back(curInputLayouts[targetIdx]);
          }
        }
        auto res_out = inferIndexingMapRelation(
            linalgOp.getMatchingIndexingMap(curInputs[targetIdx]),
            linalgOp.getIndexingMapMatchingResult(curResults[0]));
        TensorLayout outputLayout =
            *inferTargetLayout(curInputLayouts[targetIdx], *res_out);
        outputLayouts.push_back(outputLayout);
        OperatorLayout suggestedLayout(inputLayouts, outputLayouts);
        layoutCache[linalgOp] = suggestedLayout;
      }
    } else if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
      auto inputOperand = padOp.getSource();
      auto inputRank =
          cast<ShapedType>(inputOperand.getType()).getShape().size();
      auto parent = inputOperand.getDefiningOp();
      TensorLayout curInputLayout =
          layoutCache.find(parent) != layoutCache.end()
              ? layoutCache[parent].getOutputLayout(0)
              : TensorLayout::createPlainLayout(inputRank);
      SmallVector<TensorLayout> inputLayouts{curInputLayout},
          outputLayouts{curInputLayout};
      OperatorLayout suggestedLayout(inputLayouts, outputLayouts);
      layoutCache[padOp] = suggestedLayout;
    } else if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
      auto reassociation = expandShapeOp.getReassociation();
      auto staticOutputShape = expandShapeOp.getStaticOutputShape();
      auto parent = expandShapeOp.getSrc().getDefiningOp();
      auto inputShape = expandShapeOp.getSrcType().getShape();
      TensorLayout curInputLayout =
          layoutCache.find(parent) != layoutCache.end()
              ? layoutCache[parent].getOutputLayout(0)
              : TensorLayout::createPlainLayout(inputShape.size());
      DenseMap<int64_t, int64_t> outputInputIdxMapping, inputOutputIndexMapping;
      int64_t accumulationOffset = 0;
      for (int64_t i = 0; i < static_cast<int64_t>(reassociation.size()); ++i) {
        auto subReassociation = llvm::cast<ArrayAttr>(reassociation[i]);
        for (int64_t j = 0; j < static_cast<int64_t>(subReassociation.size());
             ++j) {
          if (staticOutputShape[accumulationOffset + j] == inputShape[i]) {
            outputInputIdxMapping[accumulationOffset + j] = i;
            inputOutputIndexMapping[i] = accumulationOffset + j;
          }
        }
        accumulationOffset += subReassociation.size();
      }
      auto inputOuterAxis = curInputLayout.getOuterAxis();
      auto inputInnerAxis = curInputLayout.getInnerAxis();
      int64_t diffDifference = staticOutputShape.size() - inputShape.size();
      int64_t startIdx = 0;
      SmallVector<int64_t> outputOuterAxis, outputInnerAxis;
      for (int64_t i = 0; i < static_cast<int64_t>(staticOutputShape.size());
           ++i) {
        if (outputInputIdxMapping.find(i) != outputInputIdxMapping.end()) {
          outputOuterAxis.push_back(inputOuterAxis[outputInputIdxMapping[i]] +
                                    diffDifference);
        } else {
          outputOuterAxis.push_back(startIdx++);
        }
      }
      for (int64_t i = 0; i < static_cast<int64_t>(inputInnerAxis.size());
           ++i) {
        outputInnerAxis.push_back(inputOutputIndexMapping[inputInnerAxis[i]]);
      }
      TensorLayout outputLayout(outputOuterAxis, outputInnerAxis,
                                curInputLayout.getTileSizes());
      SmallVector<TensorLayout> inputLayouts{curInputLayout},
          outputLayouts{outputLayout};
      OperatorLayout suggestedLayout(inputLayouts, outputLayouts);
      layoutCache[expandShapeOp] = suggestedLayout;
    }
  });
}

namespace utils {
bool isPackableNamedOp(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (!supportedContractionNamedOpList(linalgOp)) {
      return true;
    }
  } else if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::PadOp>(
                 op))
    return true;
  return false;
}
} // namespace utils
} // namespace gc
} // namespace mlir
