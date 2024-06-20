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

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Location.h"

#include "gc/Transforms/Passes.h"

#include <iostream>

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_SPLITCOMPUTEINTENSIVEPATTERNS
#include "gc/Transforms/Passes.h.inc"
} // namespace gc

size_t NUM_OF_NUMA = 3;
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

void getSplitedTensors(SmallVector<Value>& outputs, Location& loc, Value tensor, int64_t target_dim, PatternRewriter &rewriter) {
  if (auto definingOp = tensor.getDefiningOp()) {
    std::cout << "tensor operation name: " << definingOp->getName().getStringRef().str() << std::endl;
  } else {
    std::cout << "tensor does not have a defining operation." << std::endl;
  }


  auto Type = tensor.getType().cast<RankedTensorType>();
  int64_t rank = Type.getRank();
  if (!Type || Type.getRank() != SUPPORTED_RANK) {
    return;
  }

  int64_t M = Type.getDimSize(0);
  int64_t N = Type.getDimSize(1);
  std::cout << "M: " << M << ", N: " << N << std::endl;
  bool has_tail = target_dim == 1 ? N % NUM_OF_NUMA != 0 : M % NUM_OF_NUMA != 0;
  int64_t split_length = target_dim == 1 ? (N + NUM_OF_NUMA - 1) / NUM_OF_NUMA : (M + NUM_OF_NUMA - 1) / NUM_OF_NUMA;
  // Split the weight tensor into NUM_OF_NUMA parts
  auto splitEvenType = target_dim == 1
                  ? RankedTensorType::get({M, split_length}, Type.getElementType())
                  : RankedTensorType::get({split_length, N}, Type.getElementType());
  auto splitTailType = splitEvenType;
  if (has_tail) splitTailType = target_dim == 1
                  ? RankedTensorType::get({M, int64_t(N % split_length)}, Type.getElementType())
                  : RankedTensorType::get({int64_t(M % split_length), N}, Type.getElementType());
  for (auto split_idx : llvm::seq<unsigned>(0, NUM_OF_NUMA)) {
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    for (auto i : llvm::seq<unsigned>(0, rank)) {
      sizes.push_back(rewriter.getIndexAttr((split_idx == (NUM_OF_NUMA-1)) ? splitTailType.getShape()[i] : splitEvenType.getShape()[i]));
      offsets.push_back(rewriter.getIndexAttr((split_idx == 0 || i != target_dim) ? 0 : splitEvenType.getShape()[i] * split_idx));
    }
    Value res = rewriter.create<tensor::ExtractSliceOp>(
            loc, split_idx == (NUM_OF_NUMA-1) ? splitTailType : splitEvenType, tensor, offsets, sizes, strides)->getResult(0);
    auto res_type = res.getType().cast<RankedTensorType>();
    std::cout << split_idx << ", res_type M: " << res_type.getDimSize(0) << ", N: " << res_type.getDimSize(1) << std::endl;
    outputs.push_back(res);
    std::cout << outputs.size() << std::endl;
  }
}

void SplitMMonN(SmallVector<Value>& outputs, SmallVector<Value>& inputs, TensorType& resultTy, Location& loc, PatternRewriter &rewriter) {
  /*Split on N axis*/
  std::cout << "split on N" << std::endl;
  int64_t M = inputs[0].getType().cast<RankedTensorType>().getDimSize(0);
  int64_t N = inputs[1].getType().cast<RankedTensorType>().getDimSize(1);
  int64_t K = inputs[0].getType().cast<RankedTensorType>().getDimSize(1);
  SmallVector<Value> splited_weights;
  getSplitedTensors(splited_weights, loc, inputs[1], /*target_dim*/1, rewriter);
  if (splited_weights.size() != NUM_OF_NUMA) return;

  for (Value weight : splited_weights) {
    Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultTy.getElementType()));
    std::cout << "weight.getType().cast<RankedTensorType>().getDimSize(1): " << weight.getType().cast<RankedTensorType>().getDimSize(1) << std::endl;
    Value empty = rewriter.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t> {M, weight.getType().cast<RankedTensorType>().getDimSize(1)}, resultTy.getElementType());
    Value tensor =
        rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    outputs.push_back(rewriter.create<linalg::MatmulOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/tensor.getType().cast<RankedTensorType>(),
          /*inputs=*/ValueRange{inputs[0], weight},
          /*outputs=*/tensor)->getResult(0));
  }
}

void SplitMMonK(SmallVector<Value>& outputs, SmallVector<Value>& inputs, TensorType& resultTy, Location& loc, PatternRewriter &rewriter) {
  /*Split on K axis*/
  std::cout << "split on K" << std::endl;
  int64_t M = inputs[0].getType().cast<RankedTensorType>().getDimSize(0);
  int64_t N = inputs[1].getType().cast<RankedTensorType>().getDimSize(1);
  int64_t K = inputs[0].getType().cast<RankedTensorType>().getDimSize(1);
  SmallVector<Value> splited_data, splited_weights;
  getSplitedTensors(splited_data, loc, inputs[0], /*target_dim*/1, rewriter);
  std::cout << "splited_data size: " << splited_data.size() << std::endl;
  if (splited_data.size() != NUM_OF_NUMA) return;
  getSplitedTensors(splited_weights, loc, inputs[1], /*target_dim*/0, rewriter);
  std::cout << "splited_weights size: " << splited_weights.size() << std::endl;
  if (splited_weights.size() != NUM_OF_NUMA) return;

  for (auto [data, weight] :
       llvm::zip_equal(splited_data, splited_weights)) {
    Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultTy.getElementType()));
    Value empty = rewriter.create<tensor::EmptyOp>(
      loc, resultTy.getShape(), resultTy.getElementType());
    Value tensor =
        rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    outputs.push_back(rewriter.create<linalg::MatmulOp>(
          /*location=*/loc,
          /*resultTensorTypes=*/tensor.getType().cast<RankedTensorType>(),
          /*inputs=*/ValueRange{data, weight},
          /*outputs=*/tensor)->getResult(0));
  }
}

bool isSupportedPostOp(Operation *op) {
  // Check if the operation is a linalg operation
  if (!isa<linalg::LinalgOp>(op))
    return false;

  // Get the inputs and outputs of the linalg operation
  bool ismax = isa<linalg::MaxOp>(op);
  bool isadd = isa<linalg::AddOp>(op);
  bool ismul = isa<linalg::MulOp>(op);
  return ismax || isadd || ismul;
}

// Helper function to get all post ops following the given operation
void getUnOps(Operation *op, SmallVectorImpl<Operation *> &postOps) {
  for (auto user : op->getUsers()) {
    if (isSupportedPostOp(user)) postOps.push_back(user);
    // Recursively search for unary ops
    getUnOps(user, postOps);
  }
}

template <typename opType>
void duplicateBinary(SmallVector<Value>& outputs,std::vector<SmallVector<Value>>& inputs, TensorType& resultTy, Location& loc, PatternRewriter &rewriter) {
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultTy.getElementType()));
  for (int i = 0; i < NUM_OF_NUMA; ++i) {
    TensorType type = inputs[i][0].getType().cast<RankedTensorType>();
    Value Empty = rewriter.create<tensor::EmptyOp>(
        loc, type.getShape(), type.getElementType());
    auto tmpOp = rewriter.create<opType>(loc, inputs[i], ValueRange {Empty});
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
    if (!operand) continue;
    if (operand.use_empty()) continue;  // Skip if operand has no uses

    // If the operand is an operation and is either emptyOp or fillOp
    if (auto definingOp = operand.getDefiningOp()) {
      if (isa<tensor::EmptyOp>(definingOp) || isa<linalg::FillOp>(definingOp)) {
        llvm::outs() << "is empty \n";
        // Recursively delete the operand operation if it has only one use
        if (definingOp->hasOneUse()) {
          deleteOperation(definingOp);
        }
      }
    }
  }

  // Step 3: Disconnect the operation from its operands and users
  op->dropAllUses();
  op->dropAllReferences();

  // Step 4: Erase the operation from its parent block
  op->erase();
}

Value addN(Value& initTensor, SmallVector<Value>& ins, TensorType& resultTy, Location& loc, PatternRewriter &rewriter) {
  llvm::outs() << "start addN \n";
  // Create indexing maps (for input tensors and output tensor)
  int num_of_args = int(ins.size()) + 1;
  MLIRContext *context = rewriter.getContext();
  SmallVector<AffineMap> indexingMaps(num_of_args,
                                AffineMap::getMultiDimIdentityMap(resultTy.getRank(), context));
  llvm::outs() << "created affinemap \n";
  // Create iterator types (parallel for all dimensions)
  // ArrayRef<StringRef> iteratorTypes(resultTy.getRank(), "parallel");
  SmallVector<utils::IteratorType> iteratorTypes(resultTy.getRank(), utils::IteratorType::parallel);
  llvm::outs() << "created IteratorType \n";
  // Create the linalg.generic op
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, resultTy, ValueRange{ins}, ValueRange{initTensor},
      indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        // Define the body of the linalg.generic operation (elementwise addition)
        Value sum = nestedBuilder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
        for (auto i = 2; i < num_of_args - 1; ++i)
          sum = nestedBuilder.create<arith::AddFOp>(nestedLoc, sum, args[i]); // Add more if more inputs
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
      });

  // Mark the output as the result of the function (for demonstration purposes)
  return genericOp.getResults().front();;
}

LogicalResult splitSingleMM(linalg::MatmulOp& op,
                                PatternRewriter &rewriter) {
  SmallVector<Operation *> postOps;
  getUnOps(op, postOps);
  auto loc = op->getLoc();
  auto resultTy = dyn_cast<TensorType>(op->getResultTypes().front());
  auto input_operands = op.getInputs();
  SmallVector<Value> input_tensors;
  for (Value operand : input_operands) {
    if (!operand.getType().isa<TensorType>()) {
      continue;
    }
    input_tensors.push_back(operand);
  }

  int64_t M = input_tensors[0].getType().cast<RankedTensorType>().getDimSize(0);
  int64_t N = input_tensors[1].getType().cast<RankedTensorType>().getDimSize(1);
  int64_t K = input_tensors[0].getType().cast<RankedTensorType>().getDimSize(1);
  std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

  int64_t target_dim = N / K >= 2 ? 1 : 0;
  SmallVector<Value> splites_res;
  if (target_dim == 1) {
    SplitMMonN(splites_res, input_tensors, resultTy, loc, rewriter);
    if (splites_res.size() != NUM_OF_NUMA) return failure();
    SmallVector<Value> Outputs = splites_res;
    auto lastInput = op->getResult(0);
    for (auto postOp : postOps) {
      llvm::outs() << "Operation name: " << postOp->getName().getStringRef() << "\n";
      auto opInputs = postOp->getOperands().drop_back();
      llvm::outs() << "inputs: " << opInputs.size() << "\n";
      auto opOutputs = postOp->getResults();
      llvm::outs() << "outputs: " << opOutputs.size() << "\n";

      std::vector<SmallVector<Value>> Inputs;
      for (auto input : opInputs) {
        if (auto definingOp = input.getDefiningOp()) {
        std::cout << "Input operation name: " << definingOp->getName().getStringRef().str() << std::endl;
        } else {
        std::cout << "Input does not have a defining operation." << std::endl;
        }
        if (input == lastInput) {
          std::cout << "enter mm output" << std::endl;
          for (size_t i = 0; i < NUM_OF_NUMA; ++i) {
            SmallVector<Value> innerVector;
            innerVector.push_back(Outputs[0]);
            Inputs.push_back(innerVector);
            Outputs.erase(Outputs.begin());
            llvm::outs() << "inputs[0].size: " << Inputs[0].size() <<" \n";
          }
        } else {
          llvm::outs() << "doesnot match anything \n";
          SmallVector<Value> splited_inputs;
          getSplitedTensors(splited_inputs, loc, input, /*target_dim*/1, rewriter);
          llvm::outs() << "inputs[0].size: " << Inputs[0].size() <<" \n";
          int i = 0;
          for (const auto &splited_input : splited_inputs) {
              Inputs[i].push_back(splited_input);
              i++;
          }
          llvm::outs() << "split input done \n";
        }
      }
      if (auto postOpType = llvm::dyn_cast<linalg::AddOp>(postOp))
        duplicateBinary<linalg::AddOp>(Outputs, Inputs, resultTy, loc, rewriter);
      else if (auto postOpType = llvm::dyn_cast<linalg::MulOp>(postOp))
        duplicateBinary<linalg::MulOp>(Outputs, Inputs, resultTy, loc, rewriter);
      else if (auto postOpType = llvm::dyn_cast<linalg::MaxOp>(postOp))
        duplicateBinary<linalg::MaxOp>(Outputs, Inputs, resultTy, loc, rewriter);
      llvm::outs() << "post op creation and deletion done \n";
      lastInput = postOp->getResult(0);
    }
    // Concatenate the two halves back together on N axis
    auto newop = rewriter.create<tensor::ConcatOp>(
    loc, target_dim, Outputs);
    llvm::outs() << "created concat \n";
    auto replaced_op = postOps.size() ? postOps.back() : op;
    if (postOps.size() > 1) {
      postOps.pop_back();
      deleteOperation(op);
      for (auto &deleteOp : postOps)
        deleteOperation(deleteOp);
    }
    rewriter.replaceOp(replaced_op, newop);
  } else {
    SplitMMonK(splites_res, input_tensors, resultTy, loc, rewriter);
    if (splites_res.size() != NUM_OF_NUMA) return failure();
    // Add the two halves back together
    // Create linalg.map operation
    Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultTy.getElementType()));
    Value empty = rewriter.create<tensor::EmptyOp>(
      loc, resultTy.getShape(), resultTy.getElementType());
    Value initTensor =
        rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    auto newop = addN(initTensor, splites_res, resultTy, loc, rewriter);
    // Replace the original operation with the new linalg.map operation
    rewriter.replaceOp(op, newop);
  }
  return success();
}

LogicalResult splitSingleMMwithUnary(linalg::MatmulOp& op,
                                PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto resultTy = dyn_cast<TensorType>(op->getResultTypes().front());
  auto input_operands = op.getInputs();
  SmallVector<Value> input_tensors;
  for (Value operand : input_operands) {
    if (!operand.getType().isa<TensorType>()) {
      continue;
    }
    input_tensors.push_back(operand);
  }

  int64_t M = input_tensors[0].getType().cast<RankedTensorType>().getDimSize(0);
  int64_t N = input_tensors[1].getType().cast<RankedTensorType>().getDimSize(1);
  int64_t K = input_tensors[0].getType().cast<RankedTensorType>().getDimSize(1);
  std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

  int64_t target_dim = N / K >= 2 ? 1 : 0;
  SmallVector<Value> splites_res;
  if (target_dim == 1) {
    SplitMMonN(splites_res, input_tensors, resultTy, loc, rewriter);
    if (splites_res.size() != NUM_OF_NUMA) return failure();


    // Concatenate the two halves back together on N axis
    auto newop = rewriter.create<tensor::ConcatOp>(
        loc, target_dim, splites_res);
    rewriter.replaceOp(op, newop);
  } else {
    SplitMMonK(splites_res, input_tensors, resultTy, loc, rewriter);
    if (splites_res.size() != NUM_OF_NUMA) return failure();
    // Add the two halves back together
    // Create linalg.map operation
    Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultTy.getElementType()));
    Value empty = rewriter.create<tensor::EmptyOp>(
      loc, resultTy.getShape(), resultTy.getElementType());
    Value initTensor =
        rewriter.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    auto newop = rewriter.create<linalg::AddOp>(
        loc, resultTy, splites_res, ValueRange{initTensor});

    // Replace the original operation with the new linalg.map operation
    rewriter.replaceOp(op, newop);
  }
  return success();
}

LogicalResult splitMLP(linalg::MatmulOp& op,
                                PatternRewriter &rewriter) {
  return success();
}

class SplitComputeIntensivePatternsRewriter
    : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const final {
    // Check if the operation has already been processed
    if (op->hasAttr("splited"))
      return failure();
    // Ensure the operation is followed by relu and another matmul.
    // auto nextOp = op->getNextNode();
    // while (isa<linalg::FillOp>(nextOp) || isa<arith::ConstantOp>(nextOp)) nextOp = nextOp->getNextNode();
    // std::cout << !isa<linalg::MaxOp>(nextOp) << std::endl;
    // need to break when encounters computational op
    // if (!nextOp || !isa<linalg::MaxOp>(nextOp))
    return splitSingleMM(op, rewriter);
    // auto reluOp = cast<linalg::MaxOp>(nextOp);
    // auto nextNextOp = reluOp->getNextNode();
    // while (isa<linalg::FillOp>(nextNextOp) || isa<arith::ConstantOp>(nextNextOp)) nextNextOp = nextNextOp->getNextNode();
    // // need to break when encounters binary op
    // if (!nextNextOp || !isa<linalg::MatmulOp>(nextNextOp))
    //   return splitSingleMMwithUnary(op, rewriter);
    // auto nextMatmulOp = cast<linalg::MatmulOp>(nextNextOp);
    // return splitMLP(op, rewriter);
  }
};

namespace gc {
class SplitComputeIntensivePatterns
    : public impl::SplitComputeIntensivePatternsBase<SplitComputeIntensivePatterns> {
public: 
  using impl::SplitComputeIntensivePatternsBase<
      SplitComputeIntensivePatterns>::SplitComputeIntensivePatternsBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.insert<SplitComputeIntensivePatternsRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    SmallVector<Operation *> ops;
    getOperation()->walk([&](Operation *op) {
      if (isa<linalg::MatmulOp>(op))
        ops.push_back(op);
    });
    GreedyRewriteConfig config;
        config.strictMode = GreedyRewriteStrictness::ExistingOps;
    bool erased;
    std::cout << "ops.size(): " << ops.size() << std::endl;
    if (failed(applyOpPatternsAndFold(ops, patternSet,
                                     config, /*changed=*/nullptr, &erased)))
      signalPassFailure();
    return;
  }
};

} // namespace gc
} // namespace mlir
