//===-- Utils.cpp - linalgx utils -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"

#include "gc/Dialect/Linalgx/Utils.h"

namespace mlir {
namespace linalgx {

/// BatchDimMap represent batch dims indices in 3 of the matmul data shapes.
/// BatchDimMap requires 3 int64 arrays params. Empty array indicates no batch
/// dims. Only allow 3 kinds of matmul: non-batch, batch and batch reduce
///
/// e.g. for a batch reduce matmul A[b,m,k]*B[b,k,n]->C[m,n], map={{0},{0},{}}
/// for a batch matmul A[b,m,k]*B[b,k,n]->C[b,m,n], map={{0},{0},{0}}
struct BatchDimMap {
public:
  BatchDimMap() = default;
  BatchDimMap(ArrayRef<int64_t> batchARef, ArrayRef<int64_t> batchBRef,
              ArrayRef<int64_t> batchCRef)
      : batchA(batchARef), batchB(batchBRef), batchC(batchCRef) {}
  /// Get original arrays
  ArrayRef<int64_t> getBatchA() const { return ArrayRef<int64_t>(batchA); }
  ArrayRef<int64_t> getBatchB() const { return ArrayRef<int64_t>(batchB); }
  ArrayRef<int64_t> getBatchC() const { return ArrayRef<int64_t>(batchC); }
  /// Get size of batch dims
  int64_t getBatchNum() const { return batchA.size(); }
  /// Get attr for batch info
  bool isBatchEmpty() const {
    return batchA.empty() && batchB.empty() && batchC.empty();
  }
  bool isBatchMatmul() const {
    return !isBatchEmpty() && batchEqualAB() && batchEqualAC();
  }
  bool isBatchReduce() const {
    return !isBatchEmpty() && batchEqualAB() && batchC.empty();
  }

private:
  bool batchEqualAB() const { return llvm::equal(batchA, batchB); }
  bool batchEqualAC() const { return llvm::equal(batchA, batchC); }
  SmallVector<int64_t> batchA;
  SmallVector<int64_t> batchB;
  SmallVector<int64_t> batchC;
};

/// PackingMap represent the dim mapping between 2 sets of sorted indices.
/// PackingMap requires 2 int64 arrays params, it is needed to verify that one
/// of them contain only 1 index, since multi-dims to multi-dims mapping is not
/// allowed. This will define a 1->N index set mapping, src is the 1 index, dst
/// is the multi-dims index list. Some helpers are provided to get the mapping
/// order(first<-second or first->second) and mapping src/dst indices.
///
/// e.g. in A[a,b] -> B[x,y,z], if dim [a] corresponding to dim [x]; dim [b]
/// corresponding to packed dims [y,z]. We can express it as
/// `PackingMap<[a] -> [x]>`, `PackingMap<[b] -> [y,z]>`, where
/// dims mapping order is A -> B
struct PackingMap {
public:
  PackingMap(ArrayRef<int64_t> firstRef, ArrayRef<int64_t> secondRef)
      : first(firstRef), second(secondRef) {}
  // Get original arrays
  ArrayRef<int64_t> getFirst() const { return ArrayRef<int64_t>(first); }
  ArrayRef<int64_t> getSecond() const { return ArrayRef<int64_t>(second); }
  // SrcDims.size() == 1; DstDims.size() >= 1
  ArrayRef<int64_t> getPackingSrcDims() const {
    return getPackingSrcIndex() == 0 ? getFirst() : getSecond();
  }
  ArrayRef<int64_t> getPackingDstDims() const {
    return getPackingDstIndex() == 0 ? getFirst() : getSecond();
  }
  // Index first is 0; Index second is 1
  unsigned getPackingSrcIndex() const { return getFirst().size() == 1 ? 0 : 1; }
  unsigned getPackingDstIndex() const { return getFirst().size() == 1 ? 1 : 0; }

private:
  SmallVector<int64_t> first;
  SmallVector<int64_t> second;
};

/// PackingAttr to represent a matmul packing:
/// vnni or non-vnni matmul, dim size of weight, batch dims, M,N,K packing map
/// Mapping order(Matmul C=A*B): mPacking A->C, nPacking B->C, kPacking A->B
struct PackingAttr {
  bool isVnni = false;
  int64_t weightDims = 0;
  BatchDimMap batchDimMap;
  SmallVector<PackingMap> mPacking;
  SmallVector<PackingMap> nPacking;
  SmallVector<PackingMap> kPacking;
};

/// Common Utils
LogicalResult emitError(StringRef msg) {
  llvm::errs() << "Linalgx Utils Error: " << msg << "\n";
  return failure();
}

/// Verify Utils
/// Since the mapping is explicit, these are the criteria to verify this op:
/// 1. packing matmul input/output must have rank
/// 2. packing matmul batch dims must be valid
/// 3. all dims mapped inside packing matmul must be permutation of its dims
/// 4. all of mapping dims must match size
/// 5. dynamic dims are viewed as invalid for now
bool verifyPacking(ShapedType shapeA, ShapedType shapeB, ShapedType shapeC,
                   const PackingAttr &attr) {
  // check rank
  bool hasRank = shapeA.hasRank() && shapeB.hasRank() && shapeC.hasRank();
  if (!hasRank)
    return false;

  // check batch axis
  bool validBatch = attr.batchDimMap.isBatchEmpty() ||
                    attr.batchDimMap.isBatchMatmul() ||
                    attr.batchDimMap.isBatchReduce();
  if (!validBatch)
    return false;

  // check packing axis
  auto getBatchAxisSet = [](llvm::SmallSet<int64_t, 8> &indexSet,
                            ArrayRef<int64_t> batchDims) {
    indexSet.insert(batchDims.begin(), batchDims.end());
  };
  auto getPackingAxisSet = [](ArrayRef<PackingMap> mapArray,
                              llvm::SmallSet<int64_t, 8> &firstIndexSet,
                              llvm::SmallSet<int64_t, 8> &secondIndexSet) {
    for (auto &packingMap : mapArray) {
      ArrayRef<int64_t> firstDims = packingMap.getFirst();
      firstIndexSet.insert(firstDims.begin(), firstDims.end());
      ArrayRef<int64_t> secondDims = packingMap.getSecond();
      secondIndexSet.insert(secondDims.begin(), secondDims.end());
    }
  };
  llvm::SmallSet<int64_t, 8> indexSetA;
  llvm::SmallSet<int64_t, 8> indexSetB;
  llvm::SmallSet<int64_t, 8> indexSetC;
  getBatchAxisSet(indexSetA, attr.batchDimMap.getBatchA());
  getBatchAxisSet(indexSetB, attr.batchDimMap.getBatchB());
  getBatchAxisSet(indexSetC, attr.batchDimMap.getBatchC());
  getPackingAxisSet(attr.mPacking, indexSetA, indexSetC);
  getPackingAxisSet(attr.nPacking, indexSetB, indexSetC);
  getPackingAxisSet(attr.kPacking, indexSetA, indexSetB);
  bool checkAxis = (shapeA.getRank() == (int64_t)indexSetA.size()) &&
                   (shapeB.getRank() == (int64_t)indexSetB.size()) &&
                   (shapeC.getRank() == (int64_t)indexSetC.size());
  if (!checkAxis)
    return false;

  // check packing dims match
  auto matchBatch = [&](const BatchDimMap &batchDimMap) {
    bool matchBatch = true;
    for (int64_t i = 0; i < batchDimMap.getBatchNum(); i++) {
      int64_t dimA = batchDimMap.getBatchA()[i];
      int64_t dimB = batchDimMap.getBatchB()[i];
      matchBatch = matchBatch && //
                   (shapeA.getDimSize(dimA) == shapeB.getDimSize(dimB));
      if (batchDimMap.isBatchMatmul()) {
        int64_t dimC = batchDimMap.getBatchC()[i];
        matchBatch = matchBatch && //
                     (shapeA.getDimSize(dimA) == shapeC.getDimSize(dimC));
      }
    }
    return matchBatch;
  };
  auto matchDims = [](ArrayRef<PackingMap> mapArray, ShapedType firstShape,
                      ShapedType secondShape) {
    for (auto &packingMap : mapArray) {
      bool isDynamic = false;
      int64_t firstSize = 1;
      ArrayRef<int64_t> firstDims = packingMap.getFirst();
      for (auto dim : firstDims) {
        int64_t size = firstShape.getDimSize(dim);
        if (size == ShapedType::kDynamic)
          isDynamic = true;
        firstSize *= size;
      }
      int64_t secondSize = 1;
      ArrayRef<int64_t> secondDims = packingMap.getSecond();
      for (auto dim : secondDims) {
        int64_t size = secondShape.getDimSize(dim);
        if (size == ShapedType::kDynamic)
          isDynamic = true;
        secondSize *= size;
      }
      if (isDynamic)
        return false; // does not support dynamic dims
      if (firstSize != secondSize)
        return false;
    }
    return true;
  };
  return matchBatch(attr.batchDimMap) &&
         matchDims(attr.mPacking, shapeA, shapeC) &&
         matchDims(attr.nPacking, shapeB, shapeC) &&
         matchDims(attr.kPacking, shapeA, shapeB);
}

/// IteratorTypes Utils
/// batch represented iterations are considered `reduction` if batch reduce
/// m packing, n packing represented iterations are considered `parallel`
/// k packing represented iterations are considered `reduction`
SmallVector<utils::IteratorType>
getIteratorTypesArray(const PackingAttr &attr) {
  SmallVector<utils::IteratorType> iteratorTypes;
  // get packing num for each packing map
  auto getBatchIteratorTypes = [&](const BatchDimMap &batchDimMap) {
    iteratorTypes.insert(iteratorTypes.end(), batchDimMap.getBatchNum(),
                         batchDimMap.isBatchReduce()
                             ? utils::IteratorType::reduction
                             : utils::IteratorType::parallel);
  };
  auto getPackingIteratorTypes = [&](ArrayRef<PackingMap> packingMaps,
                                     utils::IteratorType iterTy) {
    for (auto &mapping : packingMaps) {
      size_t packingNum = mapping.getPackingDstDims().size();
      iteratorTypes.insert(iteratorTypes.end(), packingNum, iterTy);
    }
  };
  // Process order: b, m, n, k packing
  getBatchIteratorTypes(attr.batchDimMap);
  getPackingIteratorTypes(attr.mPacking, utils::IteratorType::parallel);
  getPackingIteratorTypes(attr.nPacking, utils::IteratorType::parallel);
  getPackingIteratorTypes(attr.kPacking, utils::IteratorType::reduction);
  return iteratorTypes;
}

/// IndexingMaps Utils
/// Each packing_map will represent how symbols can be added to indexing maps.
/// For packing_map dst, AffineExpr for its indices are the AffineSymbols that
/// representing the iterator; For packing_map src, AffineExpr for its index is
/// a compound expr that calculated as its indexing related to the dst
/// AffineSymbols and dim size.
unsigned getPackingDimsExpr(MLIRContext *context,
                            SmallVector<SmallVector<AffineExpr>> &exprsArr,
                            ShapedType shapeA, ShapedType shapeB,
                            ShapedType shapeC, const PackingAttr &attr) {
  SmallVector<AffineExpr> exprsA(shapeA.getRank());
  SmallVector<AffineExpr> exprsB(shapeB.getRank());
  SmallVector<AffineExpr> exprsC(shapeC.getRank());
  unsigned dims = 0;
  // dims count from 0
  auto getBatchExprs = [&](const BatchDimMap &batchDimMap) {
    for (; (int64_t)dims < batchDimMap.getBatchNum(); dims++) {
      AffineExpr curr = getAffineDimExpr(dims, context);
      exprsA[batchDimMap.getBatchA()[dims]] = curr;
      exprsB[batchDimMap.getBatchB()[dims]] = curr;
      if (batchDimMap.isBatchMatmul())
        exprsC[batchDimMap.getBatchC()[dims]] = curr;
    }
  };
  auto getPackingExprs = [&](ArrayRef<PackingMap> mapArray,
                             ArrayRef<ShapedType> types,
                             ArrayRef<SmallVector<AffineExpr> *> exprs) {
    for (auto &packingMap : mapArray) {
      unsigned srcIndex = packingMap.getPackingSrcIndex();
      unsigned dstIndex = packingMap.getPackingDstIndex();
      ArrayRef<int64_t> srcDims = packingMap.getPackingSrcDims();
      ArrayRef<int64_t> dstDims = packingMap.getPackingDstDims();
      SmallVector<AffineExpr> &dstExprs = *exprs[dstIndex];
      SmallVector<AffineExpr> &srcExprs = *exprs[srcIndex];
      AffineExpr compound = getAffineConstantExpr(0, context);
      for (auto dim : dstDims) {
        AffineExpr curr = getAffineDimExpr(dims++, context);
        AffineExpr constant =
            getAffineConstantExpr(types[dstIndex].getDimSize(dim), context);
        compound = compound * constant + curr;
        dstExprs[dim] = curr;
      }
      srcExprs[srcDims.front()] = compound;
    }
  };
  // Process order: b, m, n, k packing, kept same as packing iterator types
  getBatchExprs(attr.batchDimMap);
  getPackingExprs(attr.mPacking, {shapeA, shapeC}, {&exprsA, &exprsC});
  getPackingExprs(attr.nPacking, {shapeB, shapeC}, {&exprsB, &exprsC});
  getPackingExprs(attr.kPacking, {shapeA, shapeB}, {&exprsA, &exprsB});
  exprsArr.emplace_back(exprsA);
  exprsArr.emplace_back(exprsB);
  exprsArr.emplace_back(exprsC);
  return dims;
}

SmallVector<AffineMap> getIndexingMaps(MLIRContext *context, ShapedType shapeA,
                                       ShapedType shapeB, ShapedType shapeC,
                                       const PackingAttr &attr) {
  SmallVector<SmallVector<AffineExpr>> exprsArr;
  unsigned dims = getPackingDimsExpr(context, exprsArr, //
                                     shapeA, shapeB, shapeC, attr);
  AffineMap mapA =
      simplifyAffineMap(AffineMap::get(dims, 0, exprsArr[0], context));
  AffineMap mapB =
      simplifyAffineMap(AffineMap::get(dims, 0, exprsArr[1], context));
  AffineMap mapC =
      simplifyAffineMap(AffineMap::get(dims, 0, exprsArr[2], context));
  return {mapA, mapB, mapC};
}

/// Packing Shape Utils
int64_t getVnniBlockDimSize(Type elemType) {
  if (elemType.isBF16()) {
    return 2;
  } else if (elemType.isInteger(8)) {
    return 4;
  }
  return -1;
}

bool verifyVnniWeight(ShapedType weightShape, int64_t weightDims, bool isVnni) {
  if (!isVnni)
    return true;
  return weightShape.hasRank() && (weightShape.getRank() == weightDims) &&
         (weightShape.getDimSize(weightDims - 1) ==
          getVnniBlockDimSize(weightShape.getElementType()));
}

/// Packing Attr Utils
PackingAttr getPackingAttr(PackingType opType) {
  PackingAttr attr;
  switch (opType) {
  case PackingType::MM4D: {
    attr.weightDims = 4;
    attr.mPacking = {PackingMap{{0}, {0}}, PackingMap{{2}, {2}}};
    attr.nPacking = {PackingMap{{0}, {1}}, PackingMap{{3}, {3}}};
    attr.kPacking = {PackingMap{{1}, {1}}, PackingMap{{3}, {2}}};
  } break;
  case PackingType::MM2D4D: {
    attr.weightDims = 4;
    attr.mPacking = {PackingMap{{0}, {0}}};
    attr.nPacking = {PackingMap{{0, 3}, {1}}};
    attr.kPacking = {PackingMap{{1}, {1, 2}}};
  } break;
  case PackingType::VNNI_MM2D: {
    attr.isVnni = true;
    attr.weightDims = 5;
    attr.mPacking = {PackingMap{{0}, {0}}};
    attr.nPacking = {PackingMap{{0, 3}, {1}}};
    attr.kPacking = {PackingMap{{1}, {1, 2, 4}}};
  } break;
  case PackingType::VNNI_MM4D: {
    attr.isVnni = true;
    attr.weightDims = 5;
    attr.mPacking = {PackingMap{{0}, {0}}, PackingMap{{2}, {2}}};
    attr.nPacking = {PackingMap{{0}, {1}}, PackingMap{{3}, {3}}};
    attr.kPacking = {PackingMap{{1}, {1}}, PackingMap{{3}, {2, 4}}};
  } break;
  case PackingType::VNNI_BRMM3D: {
    attr.isVnni = true;
    attr.weightDims = 4;
    attr.batchDimMap = {{0}, {0}, {}};
    attr.mPacking = {PackingMap{{1}, {0}}};
    attr.nPacking = {PackingMap{{2}, {1}}};
    attr.kPacking = {PackingMap{{2}, {1, 3}}};
  } break;
  default: {
    llvm::errs() << "Not a valid PackingType.\n";
  } break;
  }
  return attr;
}

/// Generic Utils
bool isGenericAttrEquivalent(linalg::GenericOp op, ShapedType shapeA,
                             ShapedType shapeB, ShapedType shapeC,
                             const PackingAttr &attr) {
  MLIRContext *context = op.getContext();
  /// Use a common order to renumber the dim id to get remapped indexing maps
  /// and iterator types, so loop order invariant comparison can be performed
  auto remapAttrDims = [&](ArrayRef<AffineMap> inMaps,
                           ArrayRef<utils::IteratorType> inIters,
                           SmallVector<AffineMap> &retMaps,
                           SmallVector<utils::IteratorType> &retIters) {
    size_t dimSize = inIters.size();
    DenseMap<AffineExpr, AffineExpr> replaceMap;
    std::map<unsigned, utils::IteratorType> iterMap;
    // get shape-to-loop map
    AffineMap inverse = inversePermutation(concatAffineMaps(inMaps));
    assert(inverse && "shape-to-loops map to be non-null");
    assert(dimSize == inverse.getResults().size());
    // renumber the dim id based on shape-to-loop map
    // get a replacement map and iterator types map
    for (auto [idx, expr] : llvm::enumerate(inverse.getResults())) {
      replaceMap[getAffineDimExpr(idx, context)] = expr;
      iterMap[cast<AffineDimExpr>(expr).getPosition()] = inIters[idx];
    }
    // replace old dim id with new ones in indexing maps
    for (auto map : inMaps) {
      retMaps.push_back(map.replace(replaceMap));
    }
    // sort IteratorType to new array using ordered map
    std::transform(iterMap.begin(), iterMap.end(), std::back_inserter(retIters),
                   [](const std::pair<unsigned, utils::IteratorType> &d) {
                     return d.second;
                   });
  };
  // re-mapped ref attrs
  SmallVector<AffineMap> mapsRef;
  SmallVector<utils::IteratorType> itersRef;
  remapAttrDims(getIndexingMaps(context, shapeA, shapeB, shapeC, attr), //
                getIteratorTypesArray(attr),                            //
                mapsRef, itersRef);
  // re-mapped op attrs
  SmallVector<AffineMap> mapsOp;
  SmallVector<utils::IteratorType> itersOp;
  remapAttrDims(op.getIndexingMapsArray(),  //
                op.getIteratorTypesArray(), //
                mapsOp, itersOp);
  // check equivalence
  return llvm::equal(mapsRef, mapsOp) && llvm::equal(itersRef, itersOp);
}

/// Packing Matmul Utils
Value createMatmulCalc(OpBuilder &b, Location loc, ValueRange args) {
  assert(args.size() == 3 && "Matmul region expects 3 args.");
  // Get data type
  Type outTy = args[2].getType();
  bool isTypeFP = llvm::isa<FloatType>(outTy);
  bool isTypeInt = llvm::isa<IntegerType>(outTy);
  auto createMulCalc = [&](Value val0, Value val1) -> Value {
    if (isTypeFP)
      return b.create<arith::MulFOp>(loc, val0, val1);
    if (isTypeInt)
      return b.create<arith::MulIOp>(loc, val0, val1);
    return nullptr;
  };
  auto createAddCalc = [&](Value val0, Value val1) -> Value {
    if (isTypeFP)
      return b.create<arith::AddFOp>(loc, val0, val1);
    if (isTypeInt)
      return b.create<arith::AddIOp>(loc, val0, val1);
    return nullptr;
  };
  // Create calc
  assert((isTypeFP || isTypeInt) && "Matmul must have valid type.");
  Value value1 = convertScalarToDtype(b, loc, args[0], outTy, false);
  Value value2 = convertScalarToDtype(b, loc, args[1], outTy, false);
  Value value3 = createMulCalc(value1, value2);
  Value value4 = createAddCalc(args[2], value3);
  return value4;
}

FailureOr<linalg::GenericOp>
makeGenericPackedMatmulOp(OpBuilder &builder, Location loc, PackingType opType,
                          ValueRange inputs, ValueRange outputs) {
  // Check input/output size
  if (inputs.size() != 2 || outputs.size() != 1) {
    return emitError("input/output size must be 2/1!");
  }
  // Get shapes of inputs and output
  auto shapeA = cast<ShapedType>(inputs.front().getType());
  auto shapeB = cast<ShapedType>(inputs.back().getType());
  auto shapeC = cast<ShapedType>(outputs.back().getType());
  // Attr of packed matmul
  PackingAttr packingAttr = getPackingAttr(opType);
  // Verify dims and shape is valid
  if (!verifyPacking(shapeA, shapeB, shapeC, packingAttr) ||
      !verifyVnniWeight(shapeB, packingAttr.weightDims, packingAttr.isVnni)) {
    return emitError("Failed to verify packing!");
  }
  // Get attrs for GenericOp
  auto indexingMaps = getIndexingMaps(builder.getContext(), //
                                      shapeA, shapeB, shapeC, packingAttr);
  auto iteratorTypes = getIteratorTypesArray(packingAttr);
  // Make the GenericOp
  return builder.create<linalg::GenericOp>(
      loc, shapeC, inputs, outputs, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = createMatmulCalc(b, loc, args);
        b.create<linalg::YieldOp>(loc, result);
      });
}

bool isGenericPackedMatmulOpImpl(linalg::GenericOp genericOp,
                                 PackingType opType) {
  // Check for matmul body
  if (!linalg::detail::isContractionBody(
          *genericOp.getBlock(), [](Operation *first, Operation *second) {
            return ((isa<arith::MulFOp>(first) && isa<arith::AddFOp>(second)) ||
                    (isa<arith::MulIOp>(first) && isa<arith::AddIOp>(second)));
          })) {
    return false;
  }
  // Check for packing
  ValueRange inputs = genericOp.getDpsInputs();
  ValueRange outputs = genericOp.getDpsInits();
  auto shapeA = cast<ShapedType>(inputs.front().getType());
  auto shapeB = cast<ShapedType>(inputs.back().getType());
  auto shapeC = cast<ShapedType>(outputs.back().getType());
  PackingAttr packingAttr = getPackingAttr(opType);
  if (!verifyPacking(shapeA, shapeB, shapeC, packingAttr) ||
      !verifyVnniWeight(shapeB, packingAttr.weightDims, packingAttr.isVnni)) {
    return false;
  }
  // Check for indexing maps and iterator types equivalence
  if (!isGenericAttrEquivalent(genericOp, shapeA, shapeB, shapeC,
                               packingAttr)) {
    return false;
  }
  // Pass all checks
  return true;
}

bool isGenericPackedMatmulOp(Operation *op, PackingType opType) {
  // Check for generic op
  return isa<linalg::GenericOp>(op) &&
         isGenericPackedMatmulOpImpl(cast<linalg::GenericOp>(op), opType);
}

bool isMatmulOp(Operation *op) {
  if (isa<linalg::LinalgOp>(op) &&
      linalg::isaContractionOpInterface(cast<linalg::LinalgOp>(op))) {
    return true;
  }
  if (isa<linalg::GenericOp>(op)) {
    for (int ty = 0; ty < (int)PackingType::NUM_TYPES; ty++) {
      if (isGenericPackedMatmulOpImpl(cast<linalg::GenericOp>(op),
                                      (PackingType)ty)) {
        return true;
      }
    }
  }
  return false;
}

} // namespace linalgx
} // namespace mlir
