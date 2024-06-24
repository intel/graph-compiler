#include "gc/Analysis/TileLayoutAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <optional>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <functional>
#include <ranges>

using namespace mlir;
using namespace mlir::gc;

//===----------------------------------------------------------------------===//
// DeadCodeAnalysis
//===----------------------------------------------------------------------===//

TileLayoutAnalysis::TileLayoutAnalysis(Operation *op) {
  visit(op);
  //   registerPointKind<CFGEdge>();
}

struct StaticRange {
  int64_t offset;
  int64_t size;
  int64_t stride;
};

/// Return the iteration domain range.
SmallVector<StaticRange> getStaticIterationDomain(Operation *op) {
  mlir::linalg::LinalgOp linalgOp = cast<mlir::linalg::LinalgOp>(op);

  SmallVector<int64_t> viewSizes = linalgOp.getStaticShape();
  AffineMap invertedMap = linalgOp.getShapesToLoopsMap();
  llvm::errs() << "sz: ";
  llvm::interleaveComma(viewSizes, llvm::errs());
  llvm::errs() << "\n";

  auto applied_map = invertedMap.compose(viewSizes);
  llvm::errs() << "appl: ";
  llvm::interleaveComma(applied_map, llvm::errs());
  llvm::errs() << "\n";

  SmallVector<StaticRange> loopIterationDomain(applied_map.size());
  for (auto num_loop = 0; num_loop < applied_map.size(); num_loop++) {
    auto loop_size = applied_map[num_loop];
    // llvm::errs() << "loop size: " << loop_size << "\n";
    loopIterationDomain[num_loop] = StaticRange{0, loop_size, 1};
  }
  return loopIterationDomain;
}

static int64_t getTileForDim(linalg::LinalgOp linalgOp, unsigned dim) {
  const int64_t tile = 32;
  SmallVector<int64_t, 4> loopsRange = linalgOp.getStaticLoopRanges();
  llvm::errs() << "[Loops]: ";
  llvm::interleaveComma(loopsRange, llvm::errs());
  llvm::errs() << "\n";
  if (loopsRange[dim] == ShapedType::kDynamic)
    return tile;
  if (loopsRange[dim] < tile || loopsRange[dim] % tile != 0)
    return 0;
  return tile;
}

static uint64_t calcTensorSizeInBytes(ShapedType &shapedType) {
  auto shape = shapedType.getShape();
  llvm::errs() << "shape [ ";
  llvm::interleaveComma(shape, llvm::errs());
  llvm::errs() << " ] \n";
  uint64_t resultSize = std::accumulate(shape.begin(), shape.end(), 1,
                                        std::multiplies<uint64_t>{});
  llvm::errs() << "size shape multiplied " << resultSize << "\n";
  llvm::errs() << "dtype: " << shapedType.getElementType()
               << " bw: " << shapedType.getElementTypeBitWidth() << "\n";
  resultSize *= (shapedType.getElementTypeBitWidth() / 8);
  llvm::errs() << "size " << resultSize << "\n\n";
  return resultSize;
}

// Гипотеза
//
// На эффективность размеров тайлов - в силу того, как организована память -
// влияют измерения. Проще говоря, измерения не равнозначны - это нужно отразить
// при генерации разбиения. Одна из опций ввести по каждому из направлений некий
// аналог силы или веса. Тогда можно сказать, что каждый тайл - это некоторый
// объемный объект, находящийся в векторных полях с нормалями перпендикулярныи
// поверхностям.
//
// При таком подходе по каждому из направлений есть свой вектор, усложняющий
// доступ к элементу, а значение может быть пропорционально страйду между
// соотвествующими элементами измерения. (возможно стоит отталкиваться также от
// размера кеша) Для этого нужно более детально рассматривать модель доступа
// памяти. Тут есть пространство для обсуждений. В таком случае можно
// рассмотреть суммарный минимальный поток векторных полей через каждый тайл.
//
// Например, по направлению "w" - вектор сопротивления очень маленький, условно,
// 1, по "h" - пускай 2, по "c" допустим 3. Тогда поток будет следующим
//             F(tile) = h*w*3 + h*c*1 + c*w*2
//          w        1
//      +--------+  --+          +
//   c /        /|            3 /
//    /        / |   |         /
//   +--------+  |   | 2
//   |        |  |   +
// h |        |  +
//   |        | /
//   |        |/
//   +--------+
//
// Тогда решение данной системы:
//  h*w*c < someCacheSize
//  h*w*c -> max
//  h*w*3 + h*c*1 + c*w*2  -> min
// укажет на "удачность" конфигурации. В данной системе не учтены краевые
// эффекты.
//
// Поток - это не единственный вариант, важно учесть неравнозначность
// направлений. Часто бОльшие тайлы лучше, но это сильно зависит также от
// возможностей параллелизации. То есть важно соотношение количества потоков и
// доступной памяти.
//
// Данную систему нужно решать в целых числах, что тоже нечет в себе
// дополнительную сложность.

static SmallVector<int64_t> generateSplit(ShapedType &shapedType,
                                          uint64_t sizeInBytes) {
  auto shapeRange = llvm::make_range(shapedType.getShape().begin(),
                                     shapedType.getShape().end());
  // not real iterator
  auto maxDimIt = llvm::max_element(shapeRange);

  // shapeRange.begin()
  return {};
}

// +1 if residial non zero
uint64_t roundup(uint64_t val, uint64_t divisor) {
  return (val + divisor - 1) / divisor;
}

SmallVector<StaticRange> split1DRange(StaticRange dimension,
                                      uint64_t numTiles) {
  llvm::errs() << "num: " << numTiles << "\n";
  int64_t tileSize = roundup(
      ((dimension.size - dimension.offset) / dimension.stride), numTiles);
  llvm::errs() << "tile size: " << tileSize << "\n";
  SmallVector<StaticRange> splitRange{numTiles};
  int64_t off = dimension.offset;
  int64_t sz = (dimension.size - dimension.offset) / dimension.stride;
  for (int64_t tile = 0; tile < numTiles; tile++) {
    // llvm::errs() << "[" << tile << "] tile sz mul: " << (tile + 1) * tileSize
    //              << " full sz: " << sz << "\n";
    // llvm::errs() << "cmp: " << ((tile + 1) * tileSize < sz) << " true - "
    //              << tileSize * dimension.stride << " false - "
    //              << (sz % tileSize) * dimension.stride
    //              << "\n";
    splitRange[tile] =
        StaticRange{.offset = off + tile * tileSize,
                    .size = ((tile + 1) * tileSize <= sz)
                                ? tileSize * dimension.stride
                                : (sz % tileSize) * dimension.stride,
                    .stride = dimension.stride};
    llvm::errs() << "[" << tile << "] off: " << splitRange[tile].offset
                 << " size: " << splitRange[tile].size << "\n";
  }
  return splitRange;
}

// ofmSplit / result split
// N dimension
// 0: [0..12] [13..22]  
// 1: [0 .. 78]
// ..
// N: [0 .. 140]

void propagateBackSplit(SmallVector<SmallVector<StaticRange>> ofmSplit,
                        mlir::linalg::LinalgOp linalgOp) {
  // SmallVector<int64_t> shape;
  // shape.reserve(ofmSplit.size());
  // for (auto splitAlongDim : llvm::enumerate(ofmSplit)) {
  //   shape[splitAlongDim.index()] = llvm::max_element(std::transform(
  //       splitAlongDim.value().begin(), splitAlongDim.value().end(),
  //       splitAlongDim.value().begin(), [](StaticRange &r) { return r.size; }));
  // }

  // ArrayRef<int64_t> ref(shape.begin(), shape.end());
  // auto composedTile = shapeToItDim.compose(ref);
  // llvm::errs() << "composed Tile: ";
  // llvm::interleaveComma(composedTile, llvm::errs());
  // llvm::errs() << "\n";
}

static void calculateOfmSplit(SmallVector<uint64_t> numTiles, OpResult result) {
  // return;

  // linalgOp.getIndexingMapMatchingResult(result);
  auto linalgOp = cast<mlir::linalg::LinalgOp>(result.getOwner());

  // Check that the indexing map used for the output is a projected
  // permutation. This could be relaxed with a more general approach that can
  // map the offsets and sizes from the result to iteration space tiles
  // (filling in full extent for dimensions not used to access the result).

  // indexinf map to convert iteration domain to ofm shape
  AffineMap indexingMap = linalgOp.getIndexingMapMatchingResult(result);
  // we need reverse to convert splitted shape to it domain and than propagate
  
  // AffineMap reverseOfmIndexMap = inversePermutation(indexingMap);

  // if (!indexingMap.isProjectedPermutation()) {
  //   return op->emitOpError(
  //       "unhandled tiled implementation generation when result is not "
  //       "accessed using a permuted projection");
  // }

  auto numLoops = linalgOp.getNumLoops();
  auto tilingInterfaceOp = cast<TilingInterface>(result.getOwner());
  SmallVector<int64_t> iterationTileOffsets(numLoops),
      iterationTileSizes(numLoops);
  SmallVector<StaticRange> ofmRanges{numLoops};
  SmallVector<SmallVector<StaticRange>> ofmSplitRanges{numLoops};

  llvm::errs() << "result type: " << result.getType() << "\n";
  llvm::errs() << "indexing map: ";
  indexingMap.print(llvm::errs());
  llvm::errs() << "\n";
  // Not sure what makes Perm
  if (!indexingMap.isPermutation()) {
    llvm::errs() << " not a permutation \n";
    SmallVector<StaticRange> iterationDomain =
        getStaticIterationDomain(result.getOwner());
    llvm::errs() << "{\n";
    for (const auto &range : llvm::enumerate(iterationDomain)) {
      llvm::errs() << "  [" << range.index()
                   << "] iteration domain: " << range.value().size << "\n";
      // iterationTileOffsets[range.index()] = range.value().offset;
      // iterationTileSizes[range.index()] = range.value().size;
      ofmRanges[range.index()] = range.value();
    }
    llvm::errs() << "}\n";
  }
  // dimPosition is in iteration space
  for (const auto &resultExpr : llvm::enumerate(indexingMap.getResults())) {
    unsigned dimPosition =
        cast<AffineDimExpr>(resultExpr.value()).getPosition();

    // llvm::errs() << "[" << resultExpr.index() << "] iteration tile offset: "
    //              << iterationTileOffsets[dimPosition]
    //              << " size: " << iterationTileSizes[dimPosition] << "\n";
    ofmSplitRanges[resultExpr.index()] = split1DRange(
        ofmRanges[resultExpr.index()], numTiles[resultExpr.index()]);
    // iterationTileOffsets[dimPosition] = offsets[resultExpr.index()];
    // iterationTileSizes[dimPosition] = sizes[resultExpr.index()];
  }

  propagateBackSplit(ofmSplitRanges, linalgOp);

  // auto t = getShape(result);
}

struct OpArgs {
  SmallVector<Value> inputs_and_weights;
  SmallVector<uint64_t> input_and_weight_sizes;
  SmallVector<OpResult> outputs;
  SmallVector<uint64_t> output_sizes;
};

static uint64_t calcTotalRequiredMem(OpArgs &info) {
  return std::accumulate(info.input_and_weight_sizes.begin(),
                         info.input_and_weight_sizes.end(), 0) +
         std::accumulate(info.output_sizes.begin(), info.output_sizes.end(), 0);
}

static bool isTilingRequired(OpArgs &info, uint64_t reasonableSizeInBytes) {
  auto tot = calcTotalRequiredMem(info);
  llvm::errs() << "tot: " << tot << " reasonableSize: " << reasonableSizeInBytes
               << " cmp: " << (tot > reasonableSizeInBytes ? "true" : "false")
               << "\n";
  return tot > reasonableSizeInBytes;
}

static ShapedType getShape(Value val) {
  auto t = val.getType();
  llvm::errs() << "t: " << t << "\n";
  if (isa<VectorType>(t))
    llvm::errs() << "Vector type\n";
  if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
    // Failsafe.
    assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
           "expected a ranked tensor or memref in LinalgInterface::getRank");
    return shapedType;
  }
}

FailureOr<ArrayRef<uint64_t>>
tryOFMTiling(std::pair<Operation *, OpArgs> info) {
  assert(info.second.outputs.size() == 1 &&
         "Multioutput case unsupported for now");
  auto result = info.second.outputs[0];
  auto ofmShape = getShape(result).getShape();

  // we should take into a count data lyout
  // let's assume NCHW, or Row-major order.
  // So in shape <s_n, s_{n-1} ... s_1, s_0>
  // data in s_0 will have stride 1 (amount of data to skip to access next
  // element along this axis) s_1 have stride of s_0 size s_2 stride s_0 * s_1
  // etc

  // so most effective in this model will be split along s_n
  // so we will try to split on shape[0] and than shape[1] and so on

  SmallVector<bool> tilableDimensions;
  SmallVector<uint64_t> numTilesPerDim =
      SmallVector<uint64_t>(ofmShape.size(), 1);
  for (uint64_t dimIt = 0; dimIt < ofmShape.size(); dimIt++) {
    llvm::errs() << "dimIt: " << dimIt << "\n";
    if (ofmShape[dimIt] == 1) {
      tilableDimensions.push_back(false);
      continue;
    }
    tilableDimensions.push_back(true);
    numTilesPerDim[dimIt] = 2;
    llvm::errs() << "numTiles: ";
    llvm::interleaveComma(numTilesPerDim, llvm::errs());
    llvm::errs() << "\n";
    calculateOfmSplit(numTilesPerDim, result);
    // numTilesPerDim.push_back(2);
  }
}

static void
backpropagateRecommendeTileSizeFromResult(linalg::LinalgOp linalgOp,
                                          uint64_t reasonableSizeInBytes) {
  // mlir::DenseMap<OpResult*, uint32_t>

  // if (outputSizeInBytes > reasonableSizeInBytes) {
  //   generateSplit(shapedType, reasonableSizeInBytes);
  // }

  std::pair<Operation *, OpArgs> info;

  SmallVector<OpResult> outputs;
  SmallVector<uint64_t> output_sizes;
  // mlir::DenseMap<OpResult*, SmallVector<SmallVector> >
  llvm::errs() << "\n =====outputs:===== \n";
  for (OpResult res : linalgOp.getOperation()->getResults()) {
    outputs.push_back(res);
    auto t = res.getType();
    llvm::errs() << "t: " << t << "\n";
    if (isa<VectorType>(t))
      return;
    if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
      // Failsafe.
      assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
             "expected a ranked tensor or memref in LinalgInterface::getRank");
      uint64_t outputSizeInBytes = calcTensorSizeInBytes(shapedType);
      output_sizes.push_back(outputSizeInBytes);
    }
  }

  SmallVector<Value> inputs_and_weights;
  SmallVector<uint64_t> input_and_weight_sizes;
  llvm::errs() << "\n =====inputs:===== \n";
  for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
    Value inp = operand->get();
    inputs_and_weights.push_back(inp);
    auto t = inp.getType();
    llvm::errs() << "t: " << t << "\n";
    if (isa<VectorType>(t))
      return;
    if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
      // Failsafe.
      assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
             "expected a ranked tensor or memref in LinalgInterface::getRank");
      uint64_t operandSizeInBytes = calcTensorSizeInBytes(shapedType);
      input_and_weight_sizes.push_back(operandSizeInBytes);
    }
  }

  Operation *op = linalgOp.getOperation();
  info = {op, OpArgs{.inputs_and_weights = inputs_and_weights,
                     .input_and_weight_sizes = input_and_weight_sizes,
                     .outputs = outputs,
                     .output_sizes = output_sizes}};

  auto totMem = calcTotalRequiredMem(info.second);
  llvm::errs() << "Op total size " << totMem << "\n\n";

  if (!isTilingRequired(info.second, reasonableSizeInBytes)) {
    llvm::errs() << "[Tiling] Nothing to do.\n";
  }

  // Try to generate split

  // conservative way is at first try to split OFM -
  // in common it don't require any additional tmp buffers

  // canTileOfmOnlySplit()
  // tryOFMTiling
  tryOFMTiling(info);
  // tryWTiling
  // tryOFM+WTiling
}

static SmallVector<int64_t>
getInitialTileSizesForMatmulOp(linalg::LinalgOp linalgOp) {
  SmallVector<int64_t> tiles(linalgOp.getNumLoops(), 0);

  mlir::SmallVector<std::pair<Value, unsigned>> operandDimPairs{};

  /// Given a dimension of the iteration space of a Linalg operation, finds
  /// all the operands in the operation that are defined on such dimension.
  /// Returns all the operand values found and their dimension positions in
  /// `operandDimPairs`.

  // expecting to treat it as d0
  // so expected result is in0 and out
  linalgOp.mapIterationSpaceDimToAllOperandDims(0, operandDimPairs);

  llvm::errs() << "\nLinalgOps: \n";
  linalgOp.dump();
  llvm::errs() << "\n";
  for (auto &p : operandDimPairs) {
    llvm::errs() << "val: {";
    mlir::OpPrintingFlags printFlags;
    p.first.printAsOperand(llvm::errs(), printFlags);
    llvm::errs() << " ";
    llvm::errs() << p.first.getType();
    llvm::errs() << "} dim pos in shape: " << p.second << "\n";
  }
  if (isa<linalg::MatmulOp>(linalgOp)) {
    tiles[0] = getTileForDim(linalgOp, 0); // i loop
    tiles[1] = getTileForDim(linalgOp, 1); // j loop
    return tiles;
  }
  llvm::errs() << "No initial tile sizes.\n";
}

SmallVector<Range> getIterationDomain(mlir::TilingInterface *tilingOp) {

  // This shouldn't trigger anyting
  OpBuilder newBuilder(tilingOp->getContext());

  // TODO there can be an issue
  // I am not sure if it's legal to use newBuilder for such thing.
  // From the other side it's not clear why I should pass builder to
  // apply affine map to some static data
  return tilingOp->getIterationDomain(newBuilder);
}

/*
    if (tileSizes.empty()) {
      tilingOptions.setTileSizeComputationFunction(
          [](OpBuilder &, Operation *) -> SmallVector<OpFoldResult> {
            return {};
          });
    } else {
      tilingOptions.setTileSizeComputationFunction([&, index = i](OpBuilder &b,
                                                                  Operation *) {
        SmallVector<OpFoldResult> sizes;
        sizes.reserve(tileSizes.size());
        unsigned dynamicIdx = 0;

        for (auto [ofrIdx, ofr] : llvm::enumerate(getMixedSizes())) {
          if (auto attr = llvm::dyn_cast_if_present<Attribute>(ofr)) {
            if (scalableSizes[ofrIdx]) {
              auto val = b.create<arith::ConstantIndexOp>(
                  getLoc(), cast<IntegerAttr>(attr).getInt());
              Value vscale =
                  b.create<vector::VectorScaleOp>(getLoc(), b.getIndexType());
              sizes.push_back(
                  b.create<arith::MulIOp>(getLoc(), val, vscale).getResult());
            } else {
              sizes.push_back(attr);
            }
            continue;
          }
          ArrayRef<Operation *> dynamicSizes = dynamicSizeProducers[dynamicIdx];
          ArrayRef<int64_t> params = paramSizes[dynamicIdx];
          ++dynamicIdx;
          assert((dynamicSizes.empty() ^ params.empty()) &&
                 "expected either dynamic sizes or parameters");
          if (!params.empty()) {
            sizes.push_back(b.getIndexAttr(params[index]));
          } else {
            sizes.push_back(dynamicSizes[index]->getResult(0));
          }
        }
        return sizes;
      });
    }
*/

void visitMatmulOp(Operation *matmulOp) {

  auto linalgOp = cast<mlir::linalg::LinalgOp>(matmulOp);

  SmallVector<int64_t> viewSizes = linalgOp.getStaticShape();
  llvm::errs() << "{\n";
  llvm::errs() << " should be statical shapes of all operans than res \n";

  llvm::errs() << "   [ ";
  llvm::interleaveComma(viewSizes, llvm::errs());
  llvm::errs() << " ] \n";
  AffineMap invertedMap = linalgOp.getShapesToLoopsMap();
  llvm::errs() << " affine map\n   ";
  // all shapes to
  invertedMap.print(llvm::errs());
  llvm::errs() << "\n";
  for (auto aff_res : invertedMap.getResults()) {
    aff_res.print(llvm::errs());
    llvm::errs() << " position " << cast<AffineDimExpr>(aff_res).getPosition()
                 << "\n";
  }

  for (auto map : linalgOp.getIndexingMapsArray()) {
    llvm::errs() << " indexing affine map\n   ";
    map.print(llvm::errs());
    llvm::errs() << "\n";
  }
  assert(invertedMap && "expected a valid Linalg op to call the method");
  auto applied_map = invertedMap.compose(viewSizes);
  llvm::errs() << " applied \n   [";
  llvm::interleaveComma(applied_map, llvm::errs());
  llvm::errs() << "] \n";

  auto itTypes = linalgOp.getIteratorTypesArray();
  llvm::errs() << " it types \n   [";
  llvm::interleaveComma(itTypes, llvm::errs());
  llvm::errs() << "] \n";

  llvm::errs() << "}\n\n";

  backpropagateRecommendeTileSizeFromResult(linalgOp, 32);

  /*
  // llvm::DenseMap<Operation *, SmallVector<OpFoldResult>> initialTiles;
  auto tiles = getInitialTileSizesForMatmulOp(linalgOp);
  linalgOp.dump();
  llvm::errs() << " [Tiles]: ";
  llvm::interleaveComma(tiles, llvm::errs());
  llvm::errs() << "\n";
  // llvm::errs() << "i loop " << getTileForDim(linalgOp, 0) << "\n";
  // llvm::errs() << "j loop " << getTileForDim(linalgOp, 1) << "\n";
  // initialTiles[linalgOp] =
  // getAsOpFoldResult(rewriter.getI64ArrayAttr(tiles));

  llvm::SmallDenseSet<Operation *> fusedOps;

  // FailureOr<scf::SCFTileAndFuseResult> fuseAndTileResult = fuseWithEltwise(
  //     cast<TilingInterface>(linalgOp), initialTiles, fusedOps, 5, 2);
  auto consumer = cast<TilingInterface>(linalgOp);

  scf::SCFTilingOptions options;
  options.setTileSizes(tiles);

  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(options);
  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand) {
        Operation *candidateOp = originalProducer.getOwner();
        if (!candidateOp || (fusedOps.count(candidateOp) &&
                             !isa<linalg::FillOp>(candidateOp))) {
          return std::make_tuple(false, false);
        }
        return std::make_tuple(true, false);
      };
  tileAndFuseOptions.setFusionControlFn(controlFn);

  // This transformation is only valid for ops that return values (i.e. not
  // valid to use with operations that have memref operands).
  if (!consumer->getNumResults()) {
    llvm::errs() << consumer << " invalid pattern for op with no results\n";
  }

  // 1. First tile the consumer.
  SetVector<Operation *> fusedProducers, tiledAndFusedOps;
  llvm::SmallDenseMap<Value, size_t> origProducerToLoopResultNum;

  if (!options.tileSizeComputationFunction) {
    llvm::errs() << consumer << " missing tile size computation function\n";
  }

  // 1. Get the range of the loops that are represented by the operation.

  // iterationDomain is a Range of OpFoldResults
  SmallVector<Range> iterationDomain = getIterationDomain(&consumer);
  size_t numLoops = iterationDomain.size();

  using SCFTileSizeComputationFunction =
      std::function<SmallVector<OpFoldResult>(OpBuilder &, Operation *)>;

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<OpFoldResult> tileSizes =
      options.tileSizeComputationFunction(rewriter, consumer);

  if (tileSizes.size() < iterationDomain.size()) {
    auto zero = rewriter.getIndexAttr(0);
    tileSizes.append(numLoops - tileSizes.size(), zero);
  }

  // 3. If there is an interchange specified, permute the iteration domain and
  // the tile sizes.
  SmallVector<int64_t> interchangeVector;
  if (!options.interchangeVector.empty()) {
    interchangeVector = fillInterchangeVector(options.interchangeVector,
                                              iterationDomain.size());
  }
  if (!interchangeVector.empty()) {
    if (!isPermutationVector(interchangeVector)) {
      return rewriter.notifyMatchFailure(
          consumer,
          "invalid intechange vector, not a permutation of the entire "
          "iteration space");
    }

    applyPermutationToVector(iterationDomain, interchangeVector);
    applyPermutationToVector(tileSizes, interchangeVector);
  }

  FailureOr<TilingResult> tilingResult;

  // 6. Find the destination tensors to use for the operation.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(rewriter, consumer.getLoc(),
                                             consumer, destinationTensors))) {
    llvm::errs() << consumer << " unable to create destination tensors\n";
  }

  llvm::errs() << "==================================\n\n";
  // mlir::linalg::LinalgOp linalgOp =
  //     cast<mlir::linalg::LinalgOp>(consumer.getOperation());
  SmallVector<Value> valuesToTile = linalgOp->getOperands();

  SmallVector<OpFoldResult> offsets, sizes;
  for (auto [tileSize, loopRange] :
       llvm::zip_equal(tileSizes, iterationDomain)) {
    if (isConstantIntValue(tileSize, 0)) {
      offsets.push_back(loopRange.offset);
      sizes.push_back(loopRange.size);
      continue;
    }
    offsets.push_back(loopRange.offset);
    llvm::errs() << "non const tileSize: " << tileSize << "\n";
    auto bounded =
        calcBoundedTileSize(rewriter, consumer.getLoc(), loopRange, tileSize);

    llvm::errs() << "bounded: " << bounded << "\n";
    sizes.push_back(bounded);
  }
  llvm::errs() << "offsets [ ";
  llvm::interleaveComma(offsets, llvm::errs());
  llvm::errs() << " ]\n";
  llvm::errs() << "  sizes [ ";
  llvm::interleaveComma(sizes, llvm::errs());
  llvm::errs() << " ]\n";

  SmallVector<std::optional<mlir::linalg::SliceParameters>> allSliceParameter =
      computeAllSliceParameters(rewriter, consumer.getLoc(), linalgOp,
                                valuesToTile, offsets, sizes, {}, true);

  // DictionaryAttr
  DenseMap<Value, mlir::linalg::SliceParameters> tiledShapes;
  for (auto item : llvm::zip(valuesToTile, allSliceParameter)) {
    Value valueToTile = std::get<0>(item);
    std::optional<mlir::linalg::SliceParameters> sliceParams =
        std::get<1>(item);
    llvm::errs() << "value to tile -  " << valueToTile.getType() << " : {\n";
    llvm::errs() << "   off [ ";
    llvm::interleaveComma(sliceParams->offsets, llvm::errs());
    llvm::errs() << " ]\n";
    llvm::errs() << "   szs [ ";
    llvm::interleaveComma(sliceParams->sizes, llvm::errs());
    llvm::errs() << " ]\n";
    llvm::errs() << "   str [ ";
    llvm::interleaveComma(sliceParams->strides, llvm::errs());
    llvm::errs() << " ]\n";
    llvm::errs() << "}\n";
    tiledShapes.insert({valueToTile, *sliceParams});
  }
  */
}

LogicalResult TileLayoutAnalysis::visit(Operation *op) {
  if (!op)
    return failure();

  auto linalgOp = cast<mlir::linalg::LinalgOp>(op);
  if (linalgOp && isa<linalg::MatmulOp>(linalgOp)) {
    visitMatmulOp(op);
  }

  return success();
}
