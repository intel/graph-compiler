#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::linalgx;

Value createAndFillTensor(OpBuilder &b, Location loc, ArrayRef<int64_t> shape,
                          Type elementType, Value fillValue) {
  auto empty = tensor::EmptyOp::create(b, loc, shape, elementType);
  return linalg::FillOp::create(b, loc, ValueRange{fillValue},
                                ValueRange{empty})
      .getResult(0);
}

SmallVector<Value> scfForInitValues(OpBuilder &b, Location loc, Value query,
                                    Type elemType) {
  // Get query tensor shape
  auto queryType = cast<ShapedType>(query.getType());
  ArrayRef<int64_t> queryShape = queryType.getShape();

  // Use the provided element type for accumulator types
  Type accType = elemType;

  int64_t qDimMinus2 = queryShape[queryShape.size() - 2]; // q[-2]
  int64_t qDimMinus1 = queryShape[queryShape.size() - 1]; // q[-1]

  // Create -inf constant for m_i_row_in
  Value negInf = arith::ConstantOp::create(
      b, loc, accType,
      b.getFloatAttr(
          accType, APFloat::getInf(cast<FloatType>(accType).getFloatSemantics(),
                                   /*Negative=*/true)));

  // Create 0 constant for l_i_row_in and acc_in
  Value zero =
      arith::ConstantOp::create(b, loc, accType, b.getFloatAttr(accType, 0.0));

  // Create m_i_row_in: shape [q[-2]], value = -inf
  Value m_i_row_in = createAndFillTensor(b, loc, {qDimMinus2}, accType, negInf);

  // Create l_i_row_in: shape [q[-2]], value = 0
  Value l_i_row_in = createAndFillTensor(b, loc, {qDimMinus2}, accType, zero);

  // Create acc_in: shape [q[-2], q[-1]], value = 0
  Value acc_in =
      createAndFillTensor(b, loc, {qDimMinus2, qDimMinus1}, accType, zero);
  return {m_i_row_in, l_i_row_in, acc_in};
}

FailureOr<SmallVector<Value>> AttentionOp::decomposeOperation(OpBuilder &b) {
  Location loc = getLoc();

  // Get input operands
  Value query = getQuery();
  Value key = getKey();
  Value value = getValue();
  Value scale = getScale();
  Value mask = getMask(); // May be null if no mask provided
  Value output = getOutput();

  auto queryType = cast<ShapedType>(query.getType());
  auto keyType = cast<ShapedType>(key.getType());
  auto outputType = cast<ShapedType>(output.getType());

  ArrayRef<int64_t> queryShape = queryType.getShape();
  ArrayRef<int64_t> keyShape = keyType.getShape();

  Type queryElemType = queryType.getElementType();
  Type outputElemType = outputType.getElementType();
  Type accElemType = b.getF32Type();

  // Basic validation
  assert((queryShape.size() == 3 || queryShape.size() == 4) &&
         "Query must be 3D or 4D");
  assert(queryShape.size() == keyShape.size() && "Q and K must have same rank");
  assert((outputElemType.isF32() || outputElemType.isF16()) &&
         "Output must be f32 or f16");

  // Get dimensions - last 2 dimensions are sequence and head dimensions
  int64_t seqDim = queryShape[queryShape.size() - 2];
  int64_t headDim = queryShape[queryShape.size() - 1];
  int64_t keySeqDim = keyShape[keyShape.size() - 2];

  // Collapse Q to 2D (assert leading dims are 1)
  SmallVector<ReassociationIndices> reassociation;
  if (queryShape.size() == 3) {
    // [batch, seq, head] -> [[batch, seq], [head]]
    assert(queryShape[0] == 1 && "Leading dimension must be 1");
    reassociation.push_back({0, 1});
    reassociation.push_back({2});
  } else {
    // [b1, b2, seq, head] -> [[b1, b2, seq], [head]]
    assert(queryShape[0] == 1 && queryShape[1] == 1 &&
           "Leading dimensions must be 1");
    reassociation.push_back({0, 1, 2});
    reassociation.push_back({3});
  }

  Value Q_block = tensor::CollapseShapeOp::create(b, loc, query, reassociation);

  // Create constants (always in f32 for accumulator precision)
  Value minusInf = arith::ConstantOp::create(
      b, loc, accElemType,
      b.getFloatAttr(
          accElemType,
          APFloat::getInf(cast<FloatType>(accElemType).getFloatSemantics(),
                          /*Negative=*/true)));
  Value zeroF32 = arith::ConstantOp::create(b, loc, accElemType,
                                            b.getFloatAttr(accElemType, 0.0));

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value stepSize = arith::ConstantIndexOp::create(b, loc, 64);
  Value seqLen = arith::ConstantIndexOp::create(b, loc, keySeqDim);

  // Prepare scale: default is 1/sqrt(headDim), cast to accumulator type
  Value scaleF32 = scale;
  if (scale.getType() != accElemType) {
    scaleF32 = arith::ExtFOp::create(b, loc, accElemType, scale);
  }
  //   scaleF32 = arith::ConstantOp::create(b, loc, accElemType,
  //       b.getFloatAttr(accElemType, 10.0f));

  // Initialize accumulator tensors
  SmallVector<Value> initValues =
      scfForInitValues(b, loc, Q_block, accElemType);
  Value m_i_row_in = initValues[0];
  Value l_i_row_in = initValues[1];
  Value acc_in = initValues[2];

  // Main flash attention loop: for k in range(0, N_CTX, STEP)
  auto forOp = scf::ForOp::create(
      b, loc, c0, seqLen, stepSize, ValueRange{acc_in, m_i_row_in, l_i_row_in},
      [&](OpBuilder &builder, Location loc, Value k, ValueRange iterArgs) {
        Value acc_in_iter = iterArgs[0];
        Value m_i_row = iterArgs[1];
        Value l_i_row = iterArgs[2];

        // Extract K block and collapse to 2D
        SmallVector<OpFoldResult> kOffsets, kSizes, kStrides;
        for (size_t i = 0; i < keyShape.size(); i++) {
          if (i == keyShape.size() - 2) {
            kOffsets.push_back(k);
            kSizes.push_back(builder.getIndexAttr(64));
          } else if (i == keyShape.size() - 1) {
            kOffsets.push_back(builder.getIndexAttr(0));
            kSizes.push_back(builder.getIndexAttr(headDim));
          } else {
            kOffsets.push_back(builder.getIndexAttr(0));
            kSizes.push_back(builder.getIndexAttr(keyShape[i]));
          }
          kStrides.push_back(builder.getIndexAttr(1));
        }

        Value K_block_high_dim = tensor::ExtractSliceOp::create(
            builder, loc, key, kOffsets, kSizes, kStrides);
        Value K_block = tensor::CollapseShapeOp::create(
            builder, loc, K_block_high_dim, reassociation);

        // Transpose K: K_T = K^T
        auto kBlockType = cast<ShapedType>(K_block.getType());
        RankedTensorType transposedType =
            RankedTensorType::get({headDim, 64}, kBlockType.getElementType());
        Value K_T_empty =
            tensor::EmptyOp::create(builder, loc, transposedType, ValueRange{});
        Value K_T =
            linalg::TransposeOp::create(builder, loc, K_block, K_T_empty,
                                        ArrayRef<int64_t>{1, 0})
                .getResult()[0];

        // QK = Q @ K^T
        RankedTensorType qkType =
            RankedTensorType::get({seqDim, 64}, accElemType);
        Value qk_empty =
            tensor::EmptyOp::create(builder, loc, qkType, ValueRange{});
        Value qk = linalg::FillOp::create(builder, loc, ValueRange{zeroF32},
                                          ValueRange{qk_empty})
                       .getResult(0);
        Value qk_out =
            linalg::MatmulOp::create(builder, loc, ValueRange{Q_block, K_T},
                                     ValueRange{qk})
                .getResult(0);

        // Apply scale: qk_scaled = qk_out * scale
        Value scale_filled = createAndFillTensor(builder, loc, {seqDim, 64},
                                                 accElemType, scaleF32);
        Value qk_scaled_empty =
            tensor::EmptyOp::create(builder, loc, qkType, ValueRange{});
        Value qk_out_scaled =
            linalg::MulOp::create(
                builder, loc, ValueRange{qk_out, scale_filled}, qk_scaled_empty)
                .getResult(0);

        // Apply attention mask if present: qk_out_scaled += mask_block
        if (mask) {
          // Extract mask block for current k position: mask[:, k:k+64]
          Value mask_block = tensor::ExtractSliceOp::create(
              builder, loc, mask,
              ArrayRef<OpFoldResult>{builder.getIndexAttr(0), k},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(seqDim),
                                     builder.getIndexAttr(64)},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(1),
                                     builder.getIndexAttr(1)});

          // Cast mask to accumulator type (f32) if needed
          auto maskElemType = cast<ShapedType>(mask.getType()).getElementType();
          if (maskElemType != accElemType) {
            RankedTensorType maskBlockF32Type =
                RankedTensorType::get({seqDim, 64}, accElemType);
            Value mask_f32_empty = tensor::EmptyOp::create(
                builder, loc, maskBlockF32Type, ValueRange{});
            mask_block =
                linalg::GenericOp::create(
                    builder, loc, TypeRange{maskBlockF32Type},
                    ValueRange{mask_block}, ValueRange{mask_f32_empty},
                    ArrayRef<AffineMap>{builder.getMultiDimIdentityMap(2),
                                        builder.getMultiDimIdentityMap(2)},
                    ArrayRef<utils::IteratorType>{
                        utils::IteratorType::parallel,
                        utils::IteratorType::parallel},
                    [&](OpBuilder &b, Location loc, ValueRange args) {
                      Value extended =
                          arith::ExtFOp::create(b, loc, accElemType, args[0]);
                      linalg::YieldOp::create(b, loc, extended);
                    })
                    .getResult(0);
          }

          // Add mask to scaled QK scores
          Value qk_masked_empty =
              tensor::EmptyOp::create(builder, loc, qkType, ValueRange{});
          qk_out_scaled =
              linalg::AddOp::create(builder, loc,
                                    ValueRange{qk_out_scaled, mask_block},
                                    qk_masked_empty)
                  .getResult(0);
        }

        // qk_max = max(qk, dim=1)
        RankedTensorType qkMaxType =
            RankedTensorType::get({seqDim}, accElemType);
        Value qk_max_empty =
            tensor::EmptyOp::create(builder, loc, qkMaxType, ValueRange{});
        Value qk_min =
            linalg::FillOp::create(builder, loc, ValueRange{minusInf},
                                   ValueRange{qk_max_empty})
                .getResult(0);
        Value qk_out_max =
            linalg::ReduceOp::create(
                builder, loc, qk_out_scaled, qk_min, ArrayRef<int64_t>{1},
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value result =
                      arith::MaximumFOp::create(b, loc, args[0], args[1]);
                  linalg::YieldOp::create(b, loc, result);
                })
                .getResult(0);

        // m_ij = max(m_i, qk_max)
        Value m_ij_empty =
            tensor::EmptyOp::create(builder, loc, qkMaxType, ValueRange{});
        Value m_ij_row =
            linalg::MaxOp::create(builder, loc, ValueRange{m_i_row, qk_out_max},
                                  m_ij_empty)
                .getResult(0);

        // Broadcast m_ij for subtraction
        Value m_ij_broadcasted_empty =
            tensor::EmptyOp::create(builder, loc, qkType, ValueRange{});
        Value m_ij_broadcasted =
            linalg::BroadcastOp::create(builder, loc, m_ij_row,
                                        m_ij_broadcasted_empty,
                                        ArrayRef<int64_t>{1})
                .getResult()[0];

        // qk_centered = qk - m_ij
        Value qk_centered_empty =
            tensor::EmptyOp::create(builder, loc, qkType, ValueRange{});
        Value qk_centered =
            linalg::SubOp::create(builder, loc,
                                  ValueRange{qk_out_scaled, m_ij_broadcasted},
                                  qk_centered_empty)
                .getResult(0);

        // qk_exp = exp(qk_centered)
        Value qk_exp_empty =
            tensor::EmptyOp::create(builder, loc, qkType, ValueRange{});
        Value qk_exp =
            linalg::ExpOp::create(builder, loc, qk_centered, qk_exp_empty)
                .getResult(0);

        // l_ij = sum(qk_exp, dim=1)
        Value l_ij_empty =
            tensor::EmptyOp::create(builder, loc, qkMaxType, ValueRange{});
        Value l_ij_zero =
            linalg::FillOp::create(builder, loc, ValueRange{zeroF32},
                                   ValueRange{l_ij_empty})
                .getResult(0);
        Value l_ij_row =
            linalg::ReduceOp::create(
                builder, loc, qk_exp, l_ij_zero, ArrayRef<int64_t>{1},
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value result =
                      arith::AddFOp::create(b, loc, args[0], args[1]);
                  linalg::YieldOp::create(b, loc, result);
                })
                .getResult(0);

        // alpha = exp(m_i - m_ij)
        Value alpha_temp =
            tensor::EmptyOp::create(builder, loc, qkMaxType, ValueRange{});
        Value alpha_diff =
            linalg::SubOp::create(builder, loc, ValueRange{m_i_row, m_ij_row},
                                  alpha_temp)
                .getResult(0);
        Value alpha_empty =
            tensor::EmptyOp::create(builder, loc, qkMaxType, ValueRange{});
        Value alpha_row =
            linalg::ExpOp::create(builder, loc, alpha_diff, alpha_empty)
                .getResult(0);

        // l_i_new = alpha * l_i + l_ij
        Value l_i_temp =
            tensor::EmptyOp::create(builder, loc, qkMaxType, ValueRange{});
        Value l_i_scaled =
            linalg::MulOp::create(builder, loc, ValueRange{l_i_row, alpha_row},
                                  l_i_temp)
                .getResult(0);
        Value l_i_new_empty =
            tensor::EmptyOp::create(builder, loc, qkMaxType, ValueRange{});
        Value l_i_row_new =
            linalg::AddOp::create(
                builder, loc, ValueRange{l_i_scaled, l_ij_row}, l_i_new_empty)
                .getResult(0);

        // Broadcast alpha for acc scaling
        RankedTensorType accType =
            RankedTensorType::get({seqDim, headDim}, accElemType);
        Value alpha_broadcasted_empty =
            tensor::EmptyOp::create(builder, loc, accType, ValueRange{});
        Value alpha_broadcasted =
            linalg::BroadcastOp::create(builder, loc, alpha_row,
                                        alpha_broadcasted_empty,
                                        ArrayRef<int64_t>{1})
                .getResult()[0];

        // acc_scaled = acc * alpha
        Value acc_scaled_empty =
            tensor::EmptyOp::create(builder, loc, accType, ValueRange{});
        Value acc_scaled =
            linalg::MulOp::create(builder, loc,
                                  ValueRange{acc_in_iter, alpha_broadcasted},
                                  acc_scaled_empty)
                .getResult(0);

        // Truncate qk_exp to f16 for matmul with V
        RankedTensorType qkF16Type =
            RankedTensorType::get({seqDim, 64}, queryElemType);
        Value qk_f16_empty =
            tensor::EmptyOp::create(builder, loc, qkF16Type, ValueRange{});
        Value qk_f16 =
            linalg::GenericOp::create(
                builder, loc, TypeRange{qkF16Type}, ValueRange{qk_exp},
                ValueRange{qk_f16_empty},
                ArrayRef<AffineMap>{builder.getMultiDimIdentityMap(2),
                                    builder.getMultiDimIdentityMap(2)},
                ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                              utils::IteratorType::parallel},
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value truncated =
                      arith::TruncFOp::create(b, loc, queryElemType, args[0]);
                  linalg::YieldOp::create(b, loc, truncated);
                })
                .getResult(0);

        // Extract V block and collapse to 2D
        Value V_block_high_dim = tensor::ExtractSliceOp::create(
            builder, loc, value, kOffsets, kSizes, kStrides);
        Value V_block = tensor::CollapseShapeOp::create(
            builder, loc, V_block_high_dim, reassociation);

        // new_acc = qk_f16 @ V + acc_scaled
        Value new_acc =
            linalg::MatmulOp::create(builder, loc, ValueRange{qk_f16, V_block},
                                     ValueRange{acc_scaled})
                .getResult(0);

        scf::YieldOp::create(builder, loc,
                             ValueRange{new_acc, m_ij_row, l_i_row_new});
      });

  // Final normalization: acc / l_i
  Value final_acc = forOp.getResult(0);
  Value l_i_final = forOp.getResult(2);

  RankedTensorType accType =
      RankedTensorType::get({seqDim, headDim}, accElemType);
  Value l_i_broadcasted_empty =
      tensor::EmptyOp::create(b, loc, accType, ValueRange{});
  Value l_i_broadcasted =
      linalg::BroadcastOp::create(b, loc, l_i_final, l_i_broadcasted_empty,
                                  ArrayRef<int64_t>{1})
          .getResult()[0];

  Value normalized_empty =
      tensor::EmptyOp::create(b, loc, accType, ValueRange{});
  Value normalized =
      linalg::DivOp::create(b, loc, ValueRange{final_acc, l_i_broadcasted},
                            normalized_empty)
          .getResult(0);

  // Truncate to output element type if needed (e.g., f32 -> f16)
  Value toExpand = normalized;
  if (outputElemType != accElemType) {
    RankedTensorType truncType =
        RankedTensorType::get({seqDim, headDim}, outputElemType);
    Value trunc_empty =
        tensor::EmptyOp::create(b, loc, truncType, ValueRange{});
    toExpand = linalg::GenericOp::create(
                   b, loc, TypeRange{truncType}, ValueRange{normalized},
                   ValueRange{trunc_empty},
                   ArrayRef<AffineMap>{b.getMultiDimIdentityMap(2),
                                       b.getMultiDimIdentityMap(2)},
                   ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                                 utils::IteratorType::parallel},
                   [&](OpBuilder &nb, Location nloc, ValueRange args) {
                     Value truncated = arith::TruncFOp::create(
                         nb, nloc, outputElemType, args[0]);
                     linalg::YieldOp::create(nb, nloc, truncated);
                   })
                   .getResult(0);
  }

  // Expand back to original shape
  Value result = tensor::ExpandShapeOp::create(b, loc, outputType, toExpand,
                                               reassociation);

  return SmallVector<Value>{result};
}
