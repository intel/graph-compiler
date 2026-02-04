#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::linalgx;

FailureOr<SmallVector<Value>> AttentionOp::decomposeOperation(OpBuilder &b) {
  Value sZero = arith::ConstantOp::create(b, getLoc(), b.getZeroAttr(mlir::IntegerType::get(b.getContext(), 64)));
  return SmallVector<Value>{sZero};
}