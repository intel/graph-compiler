//===-- GlobalAnalysis.cpp - Analyze layout on named ops --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "gc/Analysis/GlobalAnalysis.h"
#include "gc/Analysis/MatmulConfigAnalysis.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace gc {

#define DEBUG_TYPE "global-analysis"

llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                              const TensorLayout &layout) {
  SmallVector<int64_t> outerAxis = layout.getOuterAxis();
  SmallVector<int64_t> innerAxis = layout.getInnerAxis();
  SmallVector<OpFoldResult> tileSizes = layout.getTileSizes();
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

bool TensorLayout::operator==(const TensorLayout &layout) const {
  return (this->outerAxis == layout.getOuterAxis()) &&
         (this->innerAxis == layout.getInnerAxis()) &&
         (this->tileSizes == layout.getTileSizes());
}

bool TensorLayout::operator!=(const TensorLayout &layout) const {
  return !(*this == layout);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                              const OperatorLayout &opLayout) {
  if (!opLayout.getSupportedInputLayouts().empty()) {
    ss << "Input layouts: ";
    llvm::interleave(opLayout.getSupportedInputLayouts(), ss, "; ");
    ss << ". ";
  }
  if (!opLayout.getSupportedOutputLayouts().empty()) {
    ss << "Output layouts: ";
    llvm::interleave(opLayout.getSupportedOutputLayouts(), ss, "; ");
    ss << ". ";
  }
  return ss;
}

// infer the relation between two indexing maps
// returns target dim -> base dim, means target is the same as base
// we don't allow duplication, e.g. 2 target corresponding to 1 base
static FailureOr<DenseMap<int64_t, int64_t>>
inferIndexingMapRelation(AffineMap indexingMapBase,
                         AffineMap indexingMapTarget) {
  // symbols are not allowed to occur
  if (indexingMapBase.getNumSymbols() != 0 ||
      indexingMapTarget.getNumSymbols() != 0)
    return failure();
  DenseMap<int64_t, int64_t> res;
  ArrayRef<AffineExpr> resultsBase = indexingMapBase.getResults();
  ArrayRef<AffineExpr> resultsTarget = indexingMapTarget.getResults();
  for (size_t j = 0; j < resultsTarget.size(); ++j) {
    for (size_t i = 0; i < resultsBase.size(); ++i) {
      auto base = dyn_cast<AffineDimExpr>(resultsBase[i]);
      auto target = dyn_cast<AffineDimExpr>(resultsTarget[j]);
      if (base && target && base.getPosition() == target.getPosition()) {
        // dim j already mapped to certain i
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

// given target --> base and max rank of base, return base --> target
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

static TensorLayout
inferTargetLayout(const TensorLayout &layoutBase,
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
  // TODO(yifei): double consider the performance, whether to push all new axis
  // at the beginning of outer perm
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
    if (!curInputLayouts[i].isPlain()) {
      return i;
    }
  }
  return 0;
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

// copied from mlir
static SmallVector<int64_t>
projectToInnerMostNonUnitDimsPos(ArrayRef<int64_t> dimsPos,
                                 ArrayRef<ReassociationIndices> reassocIndices,
                                 ArrayRef<int64_t> targetShape) {
  SmallVector<int64_t> projectedDimsPos;
  for (auto pos : dimsPos) {
    // In the case all dims are unit, this will return the inner-most one.
    int64_t projectedPos = reassocIndices[pos].back();
    for (auto i : llvm::reverse(reassocIndices[pos])) {
      int64_t dim = targetShape[i];
      if (dim > 1 || ShapedType::isDynamic(dim)) {
        projectedPos = i;
        break;
      }
    }
    projectedDimsPos.push_back(projectedPos);
  }
  return projectedDimsPos;
}

// Check if all dims in dimsPos are divisible by the corresponding tile sizes.
static bool isDimsDivisibleByTileSizes(ArrayRef<int64_t> dimsPos,
                                       ArrayRef<int64_t> shape,
                                       ArrayRef<int64_t> tileSizes) {
  return llvm::all_of(llvm::zip_equal(dimsPos, tileSizes),
                      [shape](std::tuple<int64_t, int64_t> sizePair) {
                        int64_t dim = shape[std::get<0>(sizePair)];
                        return !ShapedType::isDynamic(dim) &&
                               (dim % std::get<1>(sizePair)) == 0;
                      });
}

// if forceBlocking is set to true, we will unconditionally convert
// input/weight/output to blocking layout; otherwise we follow the default
// heuristic logic
static SmallVector<OperatorLayout, 2>
queryMatmulLayout(IRRewriter &rewriter, linalg::LinalgOp matmulOp,
                  ArrayRef<TensorLayout> curInputLayouts,
                  bool forceBlocking = false) {
  SmallVector<OperatorLayout, 2> ret;
  // infer layout for linalg contraction named ops
  auto ARank = matmulOp.getRank(matmulOp.getDpsInputOperand(0));
  auto BRank = matmulOp.getRank(matmulOp.getDpsInputOperand(1));
  auto CRank = matmulOp.getRank(matmulOp.getDpsInitOperand(0));
  auto elementType = getElementTypeOrSelf(matmulOp.getDpsInputs()[0].getType());
  auto AShape = matmulOp.getShape(matmulOp.getDpsInputOperand(0));
  auto BShape = matmulOp.getShape(matmulOp.getDpsInputOperand(1));
  int64_t M = AShape[0], K = AShape[1], N = BShape[1];
  bool ASideTransposed =
      isa<linalg::MatmulTransposeAOp, linalg::BatchMatmulTransposeAOp>(
          matmulOp);
  bool BSideTransposed =
      isa<linalg::MatmulTransposeBOp, linalg::BatchMatmulTransposeBOp>(
          matmulOp);
  // set outer&inner axis values
  auto APackInfo = getPackingAxis(ARank, ASideTransposed);
  auto BPackInfo = getPackingAxis(BRank, BSideTransposed);
  auto CPackInfo = getPackingAxis(CRank, /*transposed*/ false);
  // query the cost model for tile sizes
  MatmulConfig cfg = MatmulConfigAnalysis(matmulOp.getOperation()).getConfig();
  uint32_t iim = cfg.innerMostMBlock, iin = cfg.innerMostNBlock,
           iik = cfg.innerMostKBlock;
  if (forceBlocking) {
    TensorLayout ALayout(APackInfo.first, APackInfo.second,
                         SmallVector<OpFoldResult>{rewriter.getIndexAttr(iim),
                                                   rewriter.getIndexAttr(iik)});
    TensorLayout BLayout(BPackInfo.first, BPackInfo.second,
                         SmallVector<OpFoldResult>{rewriter.getIndexAttr(iik),
                                                   rewriter.getIndexAttr(iin)});
    TensorLayout CLayout(CPackInfo.first, CPackInfo.second,
                         SmallVector<OpFoldResult>{rewriter.getIndexAttr(iim),
                                                   rewriter.getIndexAttr(iin)});
    ret.emplace_back(SmallVector<TensorLayout>{ALayout, BLayout},
                     SmallVector<TensorLayout>{CLayout});
    return ret;
  }
  // TODO(yifei): add detailed check for constant A or B
  bool constantA = false, constantB = true;
  SmallVector<TensorLayout> ALayouts, BLayouts, CLayouts;
  if (constantA || curInputLayouts[0].isBlocking() || (M % iim) || (K % iik) ||
      (elementType.isBF16() &&
       curInputLayouts[0] == TensorLayout({1, 0}, {}, {}))) {
    ALayouts.emplace_back(
        APackInfo.first, APackInfo.second,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(iim),
                                  rewriter.getIndexAttr(iik)});
  } else {
    ALayouts.emplace_back(APackInfo.first, SmallVector<int64_t>{},
                          SmallVector<OpFoldResult>{});
  }
  if (constantB || curInputLayouts[1].isBlocking() || K % iik || N % iin ||
      elementType.isBF16()) {
    BLayouts.emplace_back(
        BPackInfo.first, BPackInfo.second,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(iik),
                                  rewriter.getIndexAttr(iin)});
  } else {
    BLayouts.emplace_back(BPackInfo.first, SmallVector<int64_t>{},
                          SmallVector<OpFoldResult>{});
  }
  if (M == iim && M >= 32 && N % iin == 0) {
    CLayouts.emplace_back(CPackInfo.first, SmallVector<int64_t>{},
                          SmallVector<OpFoldResult>{});
  } else if (M % iim || N % iin) {
    CLayouts.emplace_back(
        CPackInfo.first, CPackInfo.second,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(iim),
                                  rewriter.getIndexAttr(iin)});
  } else {
    if (BSideTransposed) {
      CLayouts.emplace_back(CPackInfo.first, SmallVector<int64_t>{},
                            SmallVector<OpFoldResult>{});
    } else {
      // push 2 possibilities
      CLayouts.emplace_back(CPackInfo.first, SmallVector<int64_t>{},
                            SmallVector<OpFoldResult>{});
      CLayouts.emplace_back(
          CPackInfo.first, CPackInfo.second,
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(iim),
                                    rewriter.getIndexAttr(iin)});
      // duplicate ALayouts and BLayouts
      ALayouts.emplace_back(ALayouts[0]);
      BLayouts.emplace_back(BLayouts[0]);
    }
  }
  for (auto [ALayout, BLayout, CLayout] :
       llvm::zip(ALayouts, BLayouts, CLayouts)) {
    ret.emplace_back(SmallVector<TensorLayout>{ALayout, BLayout},
                     SmallVector<TensorLayout>{CLayout});
  }
  return ret;
}

GlobalAnalysis::GlobalAnalysis(Operation *root) {
  IRRewriter rewriter(root);
  int64_t totalLayoutPossibilities = 1;
  std::vector<int64_t> possibilities;
  int64_t numMatmuls = 0;
  root->walk([&](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (mlir::gc::utils::isSupportedContractionNamedOp(linalgOp)) {
        auto curInputs = linalgOp.getDpsInputOperands();
        SmallVector<TensorLayout> curInputLayouts;
        for (auto input : curInputs)
          curInputLayouts.push_back(TensorLayout::createPlainLayout(
              linalgOp.getMatchingIndexingMap(input).getNumResults()));
        auto suggestedLayouts =
            queryMatmulLayout(rewriter, linalgOp, curInputLayouts);
        possibilities.push_back(suggestedLayouts.size());
        totalLayoutPossibilities *= possibilities.back();
        numMatmuls++;
      }
    }
    return WalkResult::advance();
  });
  auto computePackingCost =
      [&](linalg::LinalgOp linalgOp, ArrayRef<TensorLayout> curInputLayouts,
          ArrayRef<TensorLayout> suggestedLayout) -> int64_t {
    int64_t cost = 0;
    for (auto [operand, curLayout, suggestedLayout] :
         llvm::zip(linalgOp.getDpsInputOperands(), curInputLayouts,
                   suggestedLayout)) {
      if (curLayout != suggestedLayout) {
        ArrayRef<int64_t> shape = linalgOp.getShape(operand);
        int64_t inputSize = std::accumulate(
            shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
        if (suggestedLayout.isBlocking())
          cost += inputSize * 0.9;
        else
          cost += inputSize;
      }
    }
    return cost;
  };
  std::vector<int64_t> curChoice(possibilities.size(), 0);
  int64_t bestCost = std::numeric_limits<int64_t>::max();
  for (int64_t trialIdx = 0; trialIdx < totalLayoutPossibilities; ++trialIdx) {
    // trialIdx to map
    int64_t tmpIdx = trialIdx;
    for (size_t i = 0; i < possibilities.size(); i++) {
      curChoice[i] = tmpIdx % possibilities[i];
      tmpIdx /= possibilities[i];
    }
    LLVM_DEBUG(llvm::dbgs() << "Inferring with layout choice: [");
    LLVM_DEBUG(llvm::interleaveComma(curChoice, llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "].\n");
    int64_t curMatmulIdx = 0;
    int64_t curCost = 0;
    DenseMap<Operation *, OperatorLayout> tmpLayoutCache;
    root->walk([&](Operation *op) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        auto curInputs = linalgOp.getDpsInputOperands();
        auto curResults = linalgOp.getOperation()->getResults();
        // get current op's input layouts
        SmallVector<TensorLayout> curInputLayouts;
        for (auto input : curInputs) {
          auto parent = input->get().getDefiningOp();
          if (tmpLayoutCache.find(parent) != tmpLayoutCache.end()) {
            // TODO(yifei): it is not always 0 here
            curInputLayouts.push_back(
                tmpLayoutCache[parent].getOutputLayout(0));
          } else {
            curInputLayouts.push_back(TensorLayout::createPlainLayout(
                linalgOp.getMatchingIndexingMap(input).getNumResults()));
          }
        }
        // infer current op's output layout accordingly
        if (mlir::gc::utils::isSupportedContractionNamedOp(linalgOp)) {
          auto suggestedLayouts =
              queryMatmulLayout(rewriter, linalgOp, curInputLayouts, false);
          tmpLayoutCache[linalgOp] =
              suggestedLayouts[curChoice[curMatmulIdx++]];
          curCost += computePackingCost(
              linalgOp, curInputLayouts,
              tmpLayoutCache[linalgOp].getSupportedInputLayouts());
        } else if (mlir::gc::utils::isPackableNamedOp(op)) {
          // infer layout for non-contraction/non-convolution linalg named ops
          // and linalg generic ops
          SmallVector<TensorLayout> inputLayouts, outputLayouts;
          size_t targetIdx = getTargetInputIdx(curInputLayouts);
          for (size_t i = 0; i < curInputs.size(); ++i) {
            // getMatchingIndexingMap
            if (i != targetIdx) {
              auto indexRelation = inferIndexingMapRelation(
                  linalgOp.getMatchingIndexingMap(curInputs[targetIdx]),
                  linalgOp.getMatchingIndexingMap(curInputs[i]));
              if (failed(indexRelation)) {
                return WalkResult::skip();
              }
              TensorLayout inputLayout =
                  inferTargetLayout(curInputLayouts[targetIdx], *indexRelation);
              inputLayouts.push_back(inputLayout);
            } else {
              inputLayouts.push_back(curInputLayouts[targetIdx]);
            }
          }
          auto indexRelation = inferIndexingMapRelation(
              linalgOp.getMatchingIndexingMap(curInputs[targetIdx]),
              linalgOp.getIndexingMapMatchingResult(curResults[0]));
          if (failed(indexRelation)) {
            return WalkResult::skip();
          }
          TensorLayout outputLayout =
              inferTargetLayout(curInputLayouts[targetIdx], *indexRelation);
          outputLayouts.push_back(outputLayout);
          OperatorLayout suggestedLayout(inputLayouts, outputLayouts);
          tmpLayoutCache[linalgOp] = suggestedLayout;
          curCost +=
              computePackingCost(linalgOp, curInputLayouts, inputLayouts);
        }
      } else if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
        auto inputOperand = padOp.getSource();
        auto inputRank =
            cast<ShapedType>(inputOperand.getType()).getShape().size();
        auto parent = inputOperand.getDefiningOp();
        TensorLayout curInputLayout =
            tmpLayoutCache.find(parent) != tmpLayoutCache.end()
                ? tmpLayoutCache[parent].getOutputLayout(0)
                : TensorLayout::createPlainLayout(inputRank);
        SmallVector<TensorLayout> inputLayouts{curInputLayout},
            outputLayouts{curInputLayout};
        OperatorLayout suggestedLayout(inputLayouts, outputLayouts);
        tmpLayoutCache[padOp] = suggestedLayout;
      } else if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
        SmallVector<ReassociationIndices> reassocIndices =
            expandShapeOp.getReassociationIndices();
        auto staticOutputShape = expandShapeOp.getStaticOutputShape();
        auto parent = expandShapeOp.getSrc().getDefiningOp();
        auto inputShape = expandShapeOp.getSrcType().getShape();
        TensorLayout curInputLayout =
            tmpLayoutCache.find(parent) != tmpLayoutCache.end()
                ? tmpLayoutCache[parent].getOutputLayout(0)
                : TensorLayout::createPlainLayout(inputShape.size());
        SmallVector<int64_t> innerTileSizes;
        auto tileSizes = getConstantIntValues(curInputLayout.getTileSizes());
        if (tileSizes) {
          innerTileSizes = *tileSizes;
        } else {
          return WalkResult::skip();
        }
        SmallVector<int64_t> innerPosPos = curInputLayout.getInnerAxis();
        SmallVector<int64_t> outerDimsPerm = curInputLayout.getOuterAxis();
        SmallVector<int64_t> projectedInnerDimsPos =
            projectToInnerMostNonUnitDimsPos(innerPosPos, reassocIndices,
                                             staticOutputShape);

        if (!isDimsDivisibleByTileSizes(projectedInnerDimsPos,
                                        staticOutputShape, innerTileSizes)) {
          return WalkResult::skip();
        }
        SmallVector<int64_t> newOuterDimsPerm;
        for (auto outerPos : outerDimsPerm) {
          newOuterDimsPerm.insert(newOuterDimsPerm.end(),
                                  reassocIndices[outerPos].begin(),
                                  reassocIndices[outerPos].end());
        }
        TensorLayout outputLayout(newOuterDimsPerm, projectedInnerDimsPos,
                                  curInputLayout.getTileSizes());
        SmallVector<TensorLayout> inputLayouts{curInputLayout},
            outputLayouts{outputLayout};
        OperatorLayout suggestedLayout(inputLayouts, outputLayouts);
        tmpLayoutCache[expandShapeOp] = suggestedLayout;
      } else if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
        SmallVector<ReassociationIndices> reassocIndices =
            collapseShapeOp.getReassociationIndices();
        auto parent = collapseShapeOp.getSrc().getDefiningOp();
        auto inputShape = collapseShapeOp.getSrcType().getShape();
        TensorLayout curInputLayout =
            tmpLayoutCache.find(parent) != tmpLayoutCache.end()
                ? tmpLayoutCache[parent].getOutputLayout(0)
                : TensorLayout::createPlainLayout(inputShape.size());
        auto innerPos = curInputLayout.getInnerAxis();
        llvm::SetVector<int64_t> innerPosSet(innerPos.begin(), innerPos.end());
        for (auto [idx, indices] : llvm::enumerate(reassocIndices)) {
          // For each reassociation, figure out which dimensions get packed if
          // any.
          llvm::SetVector<int64_t> collapseDimPos(indices.begin(),
                                                  indices.end());
          llvm::SetVector<int64_t> packedDims =
              llvm::set_intersection(innerPosSet, collapseDimPos);
          // only one of the collapsed indices can be packed
          if (packedDims.size() > 1)
            return WalkResult::skip();
          // Only the inner-most expanded dimension should be packed. Otherwise,
          // elements order will be affected after operation reordering.
          if (!packedDims.empty() && packedDims[0] != indices.back())
            return WalkResult::skip();
        }

        // Project pack.inner_dims_pos to positions before shape expansion.
        SmallVector<int64_t> projectedInnerDimsPos;
        for (auto pos : innerPos) {
          for (auto [idx, indices] : llvm::enumerate(reassocIndices)) {
            if (llvm::any_of(indices, [&](int64_t collapseDim) {
                  return collapseDim == pos;
                })) {
              projectedInnerDimsPos.push_back(idx);
              break;
            }
          }
        }
        assert(projectedInnerDimsPos.size() == innerPos.size() &&
               "Invalid dim pos projection");

        // outerPerm shall be a permutation of reassocIndices
        auto outerPerm = curInputLayout.getOuterAxis();
        SmallVector<int64_t> newOuterDimsPerm;
        int64_t axisIdx = 0;
        while (axisIdx < static_cast<int64_t>(outerPerm.size())) {
          for (auto [idx, indices] : llvm::enumerate(reassocIndices)) {
            if (llvm::any_of(indices, [&](int64_t collapseDim) {
                  return collapseDim == outerPerm[axisIdx];
                })) {
              for (auto collapseDim : indices) {
                if (collapseDim != outerPerm[axisIdx++])
                  return WalkResult::skip();
              }
              newOuterDimsPerm.push_back(idx);
              break;
            }
          }
        }
        TensorLayout outputLayout(newOuterDimsPerm, projectedInnerDimsPos,
                                  curInputLayout.getTileSizes());
        SmallVector<TensorLayout> inputLayouts{curInputLayout},
            outputLayouts{outputLayout};
        OperatorLayout suggestedLayout(inputLayouts, outputLayouts);
        tmpLayoutCache[collapseShapeOp] = suggestedLayout;
      }
      if (tmpLayoutCache.find(op) != tmpLayoutCache.end()) {
        LLVM_DEBUG(llvm::dbgs() << "Inferred layout of op: " << op->getName()
                                << " is: " << tmpLayoutCache[op] << "\n");
      }
      return WalkResult::advance();
    });
    if (curCost < bestCost) {
      bestCost = curCost;
      layoutCache = tmpLayoutCache;
      LLVM_DEBUG(llvm::dbgs()
                 << "Current cost " << curCost
                 << " is lower than the best cost; update best cost."
                 << "\n");
    }
  }
}

namespace utils {
bool isSupportedContractionNamedOp(linalg::LinalgOp &linalgOp) {
  return isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
             linalg::MatmulTransposeBOp>(
      linalgOp);
}

bool isPackableNamedOp(Operation *op) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (!mlir::linalg::isaContractionOpInterface(linalgOp) &&
        !isa<linalg::ConvolutionOpInterface>(linalgOp.getOperation()) &&
        !isSupportedContractionNamedOp(linalgOp)) {
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
