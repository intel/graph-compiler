/*******************************************************************************
 * Copyright 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#include "gc/Transforms/Passes.h"

#include <iostream>

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_SPLITCOMPUTEINTENSIVEPATTERNS
#include "gc/Transforms/Passes.h.inc"
} // namespace gc

size_t NUM_OF_NUMA = 2;
size_t SUPPORTED_RANK = 2;

void printValueType(Value value) {
  if (!value) {
    llvm::outs() << "Invalid value\n";
    return;
  }

  Type type = value.getType();
  type.print(llvm::outs());
  llvm::outs() << "\n";
}

void getSplitedTensors(SmallVector<Value> &outputs, Value tensor,
                       int64_t target_dim, PatternRewriter &rewriter) {
  auto Type = tensor.getType().cast<RankedTensorType>();
  auto loc = tensor.getLoc();
  int64_t rank = Type.getRank();
  llvm::outs() << "split rank: " << rank << "\n";
  if (!Type || Type.getRank() > SUPPORTED_RANK) {
    return;
  }
  llvm::outs() << "split shape: [";
  for (int64_t dim : Type.getShape()) {
    llvm::outs() << dim << " ";
  }
  llvm::outs() << "]\n";
  llvm::outs() << "target_dim: " << target_dim << "\n";
  bool has_tail = Type.getDimSize(target_dim) % NUM_OF_NUMA != 0;
  int64_t split_length =
      (Type.getDimSize(target_dim) + NUM_OF_NUMA - 1) / NUM_OF_NUMA;
  SmallVector<int64_t> shape(Type.getShape().begin(), Type.getShape().end());
  shape[target_dim] = split_length;
  // Split the weight tensor into NUM_OF_NUMA parts
  auto splitEvenType = RankedTensorType::get(shape, Type.getElementType());
  auto splitTailType = splitEvenType;
  if (has_tail) {
    shape[target_dim] = Type.getDimSize(target_dim) % split_length;
    splitTailType = RankedTensorType::get(shape, Type.getElementType());
  }
  llvm::outs() << "start to extract slice\n";
  for (auto split_idx : llvm::seq<unsigned>(0, NUM_OF_NUMA)) {
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    for (auto i : llvm::seq<unsigned>(0, rank)) {
      sizes.push_back(rewriter.getIndexAttr((split_idx == (NUM_OF_NUMA - 1))
                                                ? splitTailType.getShape()[i]
                                                : splitEvenType.getShape()[i]));
      offsets.push_back(
          rewriter.getIndexAttr((split_idx == 0 || i != target_dim)
                                    ? 0
                                    : splitEvenType.getShape()[i] * split_idx));
    }
    Value res =
        rewriter
            .create<tensor::ExtractSliceOp>(
                loc,
                split_idx == (NUM_OF_NUMA - 1) ? splitTailType : splitEvenType,
                tensor, offsets, sizes, strides)
            ->getResult(0);
    auto res_type = res.getType().cast<RankedTensorType>();
    llvm::outs() << "splited shape: [";
    for (int64_t dim : res_type.getShape()) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "]\n";
    outputs.push_back(res);
    std::cout << outputs.size() << std::endl;
  }
}

void splitBroadcast(SmallVector<Value> &outputs,
                    linalg::BroadcastOp broadcastOp, int64_t target_dim,
                    PatternRewriter &rewriter) {
  auto loc = broadcastOp->getLoc();
  SmallVector<Value> broadcastInputs;
  auto in = broadcastOp.getInput();
  if (in.getType().getShape().size() > SUPPORTED_RANK) {
    llvm::outs() << "cannot split broadcast on current size.\n";
    return;
  }
  auto out = broadcastOp.getInit();
  auto outType = out.getType().dyn_cast<RankedTensorType>();
  auto shape = outType.getShape();
  if (shape.size() != SUPPORTED_RANK || target_dim != 1) {
    llvm::outs()
        << "cannot split broadcast on current size or current target dim \n";
    return;
  }
  llvm::outs() << "Tensor shape: [";
  for (int64_t dim : shape) {
    llvm::outs() << dim << " ";
  }
  llvm::outs() << "]\n";
  llvm::outs() << "duplicate broadcast inputs\n";
  getSplitedTensors(broadcastInputs, in,
                    /*target_dim*/ in.getType().getShape().size() - 1,
                    rewriter);
  if (auto emptyOp = dyn_cast<tensor::EmptyOp>(out.getDefiningOp())) {
    int64_t split_length = (shape[1] + NUM_OF_NUMA - 1) / NUM_OF_NUMA;
    int64_t split_tail =
        shape[1] % NUM_OF_NUMA != 0 ? shape[1] % split_length : split_length;
    for (auto split_idx : llvm::seq<unsigned>(0, NUM_OF_NUMA)) {
      Value empty = rewriter.create<tensor::EmptyOp>(
          loc,
          ArrayRef<int64_t>{shape[0], (split_idx == (NUM_OF_NUMA - 1))
                                          ? split_tail
                                          : split_length},
          outType.getElementType());
      Value res =
          rewriter
              .create<linalg::BroadcastOp>(loc, broadcastInputs[split_idx],
                                           empty, broadcastOp.getDimensions())
              .getResults()[0];
      outputs.push_back(res);
      std::cout << outputs.size() << std::endl;
    }
  }
}

void SplitMMonN(Operation *op, SmallVector<Value> &outputs,
                SmallVector<Value> &inputs, TensorType &resultTy,
                int64_t target_dim, Location &loc, PatternRewriter &rewriter) {
  /*Split on N axis*/
  std::cout << "split on N" << std::endl;
  int64_t M = inputs[0].getType().cast<RankedTensorType>().getDimSize(0);
  int64_t N =
      inputs[1].getType().cast<RankedTensorType>().getDimSize(target_dim);
  int64_t K = inputs[0].getType().cast<RankedTensorType>().getDimSize(1);
  SmallVector<Value> splited_weights;
  getSplitedTensors(splited_weights, inputs[1], target_dim, rewriter);
  if (splited_weights.size() != NUM_OF_NUMA)
    return;

  for (Value weight : splited_weights) {
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultTy.getElementType()));
    std::cout << "weight.getType().cast<RankedTensorType>().getDimSize(1): "
              << weight.getType().cast<RankedTensorType>().getDimSize(
                     target_dim)
              << std::endl;
    Value empty = rewriter.create<tensor::EmptyOp>(
        loc,
        ArrayRef<int64_t>{
            M,
            weight.getType().cast<RankedTensorType>().getDimSize(target_dim)},
        resultTy.getElementType());
    Value tensor =
        rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    auto newMM = isa<linalg::MatmulOp>(op)
                     ? rewriter.create<linalg::MatmulOp>(
                           /*location=*/loc,
                           /*resultTensorTypes=*/
                           tensor.getType().cast<RankedTensorType>(),
                           /*inputs=*/ValueRange{inputs[0], weight},
                           /*outputs=*/tensor)
                     : rewriter.create<linalg::MatmulTransposeBOp>(
                           /*location=*/loc,
                           /*resultTensorTypes=*/
                           tensor.getType().cast<RankedTensorType>(),
                           /*inputs=*/ValueRange{inputs[0], weight},
                           /*outputs=*/tensor);
    mlir::BoolAttr boolAttr = rewriter.getBoolAttr(true);
    newMM->setAttr("splited", boolAttr);
    outputs.push_back(newMM->getResult(0));
  }
}

void SplitMMonK(Operation *op, SmallVector<Value> &outputs,
                SmallVector<Value> &inputs, TensorType &resultTy, Location &loc,
                PatternRewriter &rewriter) {
  /*Split on K axis*/
  std::cout << "split on K" << std::endl;
  int64_t M = inputs[0].getType().cast<RankedTensorType>().getDimSize(0);
  int64_t N = inputs[1].getType().cast<RankedTensorType>().getDimSize(1);
  int64_t K = inputs[0].getType().cast<RankedTensorType>().getDimSize(1);
  SmallVector<Value> splited_data, splited_weights;
  getSplitedTensors(splited_data, inputs[0], /*target_dim*/ 1, rewriter);
  std::cout << "splited_data size: " << splited_data.size() << std::endl;
  if (splited_data.size() != NUM_OF_NUMA)
    return;
  getSplitedTensors(splited_weights, inputs[1], /*target_dim*/ 0, rewriter);
  std::cout << "splited_weights size: " << splited_weights.size() << std::endl;
  if (splited_weights.size() != NUM_OF_NUMA)
    return;

  for (auto [data, weight] : llvm::zip_equal(splited_data, splited_weights)) {
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultTy.getElementType()));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                   resultTy.getElementType());
    Value tensor =
        rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    auto newMM = rewriter.create<linalg::MatmulOp>(
        /*location=*/loc,
        /*resultTensorTypes=*/tensor.getType().cast<RankedTensorType>(),
        /*inputs=*/ValueRange{data, weight},
        /*outputs=*/tensor);
    mlir::BoolAttr boolAttr = rewriter.getBoolAttr(true);
    newMM->setAttr("splited", boolAttr);
    outputs.push_back(newMM->getResult(0));
  }
}

bool isSupportedPostOp(Operation *op) {
  // Check if the operation is a linalg operation
  if (!isa<linalg::LinalgOp>(op))
    return false;

  // Get the inputs and outputs of the linalg operation
  bool isMax = isa<linalg::MaxOp>(op);
  bool isAdd = isa<linalg::AddOp>(op);
  bool isMul = isa<linalg::MulOp>(op);
  // bool isTranspose = isa<linalg::TransposeOp>(op);
  return isMax || isAdd || isMul;
}

// Helper function to get all post ops following the given operation
void getUnOps(Operation *op, SmallVectorImpl<Operation *> &postOps) {
  for (auto user : op->getUsers()) {
    if (isSupportedPostOp(user))
      postOps.push_back(user);
    if (isa<linalg::MatmulOp, linalg::TransposeOp, tensor::ExpandShapeOp>(user))
      return;
    // Recursively search for unary ops, unless it's a matmul op
    getUnOps(user, postOps);
    // }
  }
}

template <typename opType>
void duplicateBinary(SmallVector<Value> &outputs,
                     std::vector<SmallVector<Value>> &inputs,
                     TensorType &resultTy, PatternRewriter &rewriter) {
  for (int i = 0; i < NUM_OF_NUMA; ++i) {
    auto loc = inputs[i][0].getLoc();
    TensorType type = inputs[i][0].getType().cast<RankedTensorType>();
    Value Empty = rewriter.create<tensor::EmptyOp>(loc, type.getShape(),
                                                   type.getElementType());
    auto tmpOp = rewriter.create<opType>(loc, inputs[i], ValueRange{Empty});
    for (auto result : tmpOp->getResults()) {
      outputs.push_back(result);
    }
  }
}

void duplicateTranspose(SmallVector<Value> &outputs,
                        std::vector<SmallVector<Value>> &inputs,
                        linalg::TransposeOp transposeOp, TensorType &resultTy,
                        PatternRewriter &rewriter) {
  ArrayRef<int64_t> permutation = transposeOp.getPermutation();
  if (permutation.size() != SUPPORTED_RANK) {
    llvm::outs() << "unsupported rank\n";
    return;
  }
  for (int i = 0; i < NUM_OF_NUMA; ++i) {
    auto loc = inputs[i][0].getLoc();
    TensorType type = inputs[i][0].getType().cast<RankedTensorType>();
    const auto &inputShape = type.getShape();
    SmallVector<int64_t> transShape{inputShape[permutation[0]],
                                    inputShape[permutation[1]]};
    auto transTy = type.clone(transShape);
    llvm::outs() << "TransTy shape: [";
    for (int64_t dim : transTy.getShape()) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "]\n";
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(transTy.getElementType()));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, transTy.getShape(),
                                                   transTy.getElementType());
    Value tensor =
        rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    auto tmpOp = rewriter.create<linalg::TransposeOp>(loc, inputs[i][0], tensor,
                                                      permutation);
    for (auto result : tmpOp->getResults()) {
      outputs.push_back(result);
    }
  }
}

void deleteOperation(Operation *op) {
  // Step 1: Ensure the operation exists
  if (!op)
    return;

  // Step 2: Check each operand of the operation
  for (auto operand : op->getOperands()) {
    if (!operand)
      continue;
    if (operand.use_empty())
      continue; // Skip if operand has no uses

    // If the operand is an operation and is either emptyOp or fillOp
    if (auto definingOp = operand.getDefiningOp()) {
      // if (isa<tensor::EmptyOp>(definingOp) ||
      // isa<linalg::FillOp>(definingOp)) {
      //   llvm::outs() << "is empty \n";
      //   // Recursively delete the operand operation if it has only one use
      if (definingOp->hasOneUse()) {
        deleteOperation(definingOp);
      }
      // }
    }
  }

  // Step 3: Disconnect the operation from its operands and users
  op->dropAllUses();
  op->dropAllReferences();

  // Step 4: Erase the operation from its parent block
  op->erase();
}

void deleteOperands(Operation *op) {
  for (auto operand : op->getOperands()) {
    // llvm::outs() << "operands: " << operand << "\n";
    if (!operand)
      continue;
    if (operand.use_empty()) {
      continue;
    } // Skip if operand has no uses
    if (auto definingOp = operand.getDefiningOp()) {
      if (definingOp->hasOneUse()) {
        deleteOperands(definingOp);
        definingOp->dropAllUses();
        definingOp->dropAllReferences();
        definingOp->erase();
      }
    }
  }
}

Value addN(Value &initTensor, SmallVector<Value> &ins, TensorType &resultTy,
           Location &loc, PatternRewriter &rewriter) {
  llvm::outs() << "start addN \n";
  // Create indexing maps (for input tensors and output tensor)
  int num_of_args = int(ins.size()) + 1;
  MLIRContext *context = rewriter.getContext();
  SmallVector<AffineMap> indexingMaps(
      num_of_args,
      AffineMap::getMultiDimIdentityMap(resultTy.getRank(), context));
  llvm::outs() << "created affinemap \n";
  // Create iterator types (parallel for all dimensions)
  // ArrayRef<StringRef> iteratorTypes(resultTy.getRank(), "parallel");
  SmallVector<utils::IteratorType> iteratorTypes(resultTy.getRank(),
                                                 utils::IteratorType::parallel);
  llvm::outs() << "created IteratorType \n";
  // Create the linalg.generic op
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, resultTy, ValueRange{ins}, ValueRange{initTensor}, indexingMaps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        // Define the body of the linalg.generic operation (elementwise
        // addition)
        Value sum =
            nestedBuilder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
        for (auto i = 2; i < num_of_args - 1; ++i)
          sum = nestedBuilder.create<arith::AddFOp>(
              nestedLoc, sum, args[i]); // Add more if more inputs
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
      });

  // Mark the output as the result of the function (for demonstration purposes)
  return genericOp.getResults().front();
  ;
}

LogicalResult splitSingleMM(Operation *op, PatternRewriter &rewriter) {
  SmallVector<Operation *> postOps = {};
  getUnOps(op, postOps);
  auto loc = op->getLoc();
  auto resultTy = dyn_cast<TensorType>(op->getResultTypes().front());
  auto input_operands = op->getOperands().drop_back();
  SmallVector<Value> input_tensors;
  for (Value operand : input_operands) {
    if (!operand.getType().isa<TensorType>()) {
      continue;
    }
    input_tensors.push_back(operand);
  }
  bool istransB = isa<linalg::MatmulTransposeBOp>(op);
  llvm::outs() << "is trans B\n";
  int64_t M = input_tensors[0].getType().cast<RankedTensorType>().getDimSize(0);
  int64_t N = input_tensors[1].getType().cast<RankedTensorType>().getDimSize(
      istransB ? 0 : 1);
  int64_t K = input_tensors[0].getType().cast<RankedTensorType>().getDimSize(1);
  std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

  int64_t target_dim = N / K >= 2 ? 0 : 0;
  SmallVector<Value> splites_res;
  if (target_dim == 1) {
    SplitMMonN(op, splites_res, input_tensors, resultTy, target_dim ^ istransB,
               loc, rewriter);
    if (splites_res.size() != NUM_OF_NUMA)
      return failure();
    SmallVector<Value> Outputs = splites_res;
    auto lastInput = op->getResult(0);
    llvm::outs() << "postOps num: " << postOps.size() << "\n";
    for (auto postOp : postOps) {
      llvm::outs() << "Operation name: " << postOp->getName().getStringRef()
                   << "\n";
      auto opInputs = postOp->getOperands().drop_back();
      llvm::outs() << "inputs: " << opInputs.size() << "\n";
      auto opOutputs = postOp->getResults();
      llvm::outs() << "outputs: " << opOutputs.size() << "\n";

      std::vector<SmallVector<Value>> Inputs;
      for (auto input : opInputs) {
        if (input == lastInput) {
          std::cout << "enter mm output" << std::endl;
          for (size_t i = 0; i < NUM_OF_NUMA; ++i) {
            SmallVector<Value> innerVector;
            innerVector.push_back(Outputs[0]);
            Inputs.push_back(innerVector);
            Outputs.erase(Outputs.begin());
            llvm::outs() << "inputs[" << i << "].size: " << Inputs[i].size()
                         << " \n";
          }
        } else if (auto definingOp = input.getDefiningOp()) {
          llvm::outs() << "is definingOp\n";
          std::cout << "Input operation name: "
                    << definingOp->getName().getStringRef().str() << std::endl;
          if (auto fillOp = dyn_cast<linalg::FillOp>(definingOp)) {
            llvm::outs() << "is fill \n";
            SmallVector<Value> splited_inputs;
            getSplitedTensors(splited_inputs, input, target_dim, rewriter);
            int i = 0;
            for (const auto &splited_input : splited_inputs) {
              Inputs[i].push_back(splited_input);
              llvm::outs() << "inputs[" << i << "].size: " << Inputs[i].size()
                           << " \n";
              i++;
            }
            llvm::outs() << "split input done \n";
          } else if (auto broadcastOp =
                         dyn_cast<linalg::BroadcastOp>(definingOp)) {
            llvm::outs() << "is broadcast \n";
            SmallVector<Value> splited_inputs;
            splitBroadcast(splited_inputs, broadcastOp, target_dim, rewriter);
            llvm::outs() << "inputs[0].size: " << Inputs[0].size() << " \n";
            int i = 0;
            for (const auto &splited_input : splited_inputs) {
              Inputs[i].push_back(splited_input);
              i++;
            }
            deleteOperation(broadcastOp);
            llvm::outs() << "split input done \n";
          } else if (auto constantOp =
                         dyn_cast<arith::ConstantOp>(definingOp)) {
            llvm::outs() << "is constant \n";
            auto newConstantOp = rewriter.create<arith::ConstantOp>(
                constantOp.getLoc(), constantOp.getType(),
                constantOp.getValue());
            SmallVector<Value> splited_inputs;
            getSplitedTensors(splited_inputs, newConstantOp, target_dim,
                              rewriter);
            int i = 0;
            for (const auto &splited_input : splited_inputs) {
              Inputs[i].push_back(splited_input);
              llvm::outs() << "inputs[" << i << "].size: " << Inputs[i].size()
                           << " \n";
              i++;
            }
            deleteOperation(constantOp);
            llvm::outs() << "split input done \n";
          }
        } else {
          llvm::outs() << "doesnot match anything \n";
          SmallVector<Value> splited_inputs;
          getSplitedTensors(splited_inputs, input, target_dim, rewriter);
          llvm::outs() << "inputs[0].size: " << Inputs[0].size() << " \n";
          int i = 0;
          for (const auto &splited_input : splited_inputs) {
            Inputs[i].push_back(splited_input);
            i++;
          }
          llvm::outs() << "split input done \n";
        }
      }
      if (auto postOpType = llvm::dyn_cast<linalg::AddOp>(postOp))
        duplicateBinary<linalg::AddOp>(Outputs, Inputs, resultTy, rewriter);
      else if (auto postOpType = llvm::dyn_cast<linalg::MulOp>(postOp))
        duplicateBinary<linalg::MulOp>(Outputs, Inputs, resultTy, rewriter);
      else if (auto postOpType = llvm::dyn_cast<linalg::MaxOp>(postOp))
        duplicateBinary<linalg::MaxOp>(Outputs, Inputs, resultTy, rewriter);
      // else if (auto transOp = llvm::dyn_cast<linalg::TransposeOp>(postOp)) {
      //   duplicateTranspose(Outputs, Inputs, transOp, resultTy, rewriter);
      //   target_dim ^= 0x1;
      // }
      llvm::outs() << "post op creation and deletion done \n";
      lastInput = postOp->getResult(0);
      if (auto lastop = lastInput.getDefiningOp())
        std::cout << "lastInput operation name: "
                  << lastop->getName().getStringRef().str() << std::endl;
    }
    // Concatenate the two halves back together on N axis
    auto newop = rewriter.create<tensor::ConcatOp>(Outputs.back().getLoc(),
                                                   target_dim, Outputs);
    llvm::outs() << "created concat \n";
    auto replaced_op = postOps.size() ? postOps.back() : op;
    if (postOps.size() > 1) {
      postOps.pop_back();
      deleteOperation(op);
      for (auto &deleteOp : postOps)
        deleteOperation(deleteOp);
    }
    deleteOperands(replaced_op);
    rewriter.replaceOp(replaced_op, newop);
    postOps = {};
    llvm::outs() << "after duplicate, postOps num: " << postOps.size() << "\n";
  } else {
    SplitMMonK(op, splites_res, input_tensors, resultTy, loc, rewriter);
    if (splites_res.size() != NUM_OF_NUMA) {
      llvm::outs() << "not getting the expected splited outputs\n";
      return failure();
    }
    // Add the two halves back together
    // Create linalg.map operation
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultTy.getElementType()));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                                   resultTy.getElementType());
    Value initTensor =
        rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    auto newop = addN(initTensor, splites_res, resultTy, loc, rewriter);
    // Replace the original operation with the new linalg.map operation
    rewriter.replaceOp(op, newop);
  }
  llvm::outs() << "exit duplicate mm.\n";
  llvm::outs() << "==================================================\n";
  return success();
}

class SplitMatmulRewriter : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const final {
    // Check if the operation has already been processed
    if (op->hasAttr("splited"))
      return failure();
    return splitSingleMM(op, rewriter);
  }
};

class SplitMatmulTransposeBRewriter
    : public OpRewritePattern<linalg::MatmulTransposeBOp> {
public:
  using OpRewritePattern<linalg::MatmulTransposeBOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulTransposeBOp op,
                                PatternRewriter &rewriter) const final {
    // Check if the operation has already been processed
    llvm::outs() << "get into mm transpose b\n";
    if (op->hasAttr("splited"))
      return failure();
    return splitSingleMM(op, rewriter);
  }
};

namespace gc {
class SplitComputeIntensivePatterns
    : public impl::SplitComputeIntensivePatternsBase<
          SplitComputeIntensivePatterns> {
public:
  using impl::SplitComputeIntensivePatternsBase<
      SplitComputeIntensivePatterns>::SplitComputeIntensivePatternsBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.insert<SplitMatmulRewriter>(&getContext());
    patterns.insert<SplitMatmulTransposeBRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    SmallVector<Operation *> ops;
    getOperation()->walk([&](Operation *op) {
      if (isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(op))
        ops.push_back(op);
    });
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    bool erased;
    std::cout << "ops.size(): " << ops.size() << std::endl;
    if (failed(applyOpPatternsAndFold(ops, patternSet, config,
                                      /*changed=*/nullptr, &erased)))
      signalPassFailure();
    return;
  }
};

} // namespace gc
} // namespace mlir
