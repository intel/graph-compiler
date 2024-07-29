#include "gc/Dialect/LLVMIR/GENDialect.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace gen;

#include "gc/Dialect/LLVMIR/GenOpsDialect.cpp.inc"

LogicalResult
GenTargetAttr::verify(function_ref<InFlightDiagnostic()> emitError, int O,
                      StringRef triple) {
  if (O < 0 || O > 3) {
    emitError() << "The optimization level must be a number between 0 and 3.";
    return failure();
  }
  if (triple.empty()) {
    emitError() << "The target triple cannot be empty.";
    return failure();
  }
  return success();
}

LogicalResult GENDialect::verifyOperationAttribute(Operation *op,
                                                   NamedAttribute attr) {
  return success();
}

void GENDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "gc/Dialect/LLVMIR/GenOpsAttributes.cpp.inc"
      >();

  allowUnknownOperations();
  declarePromisedInterface<gpu::TargetAttrInterface, GenTargetAttr>();
}

#define GET_ATTRDEF_CLASSES
#include "gc/Dialect/LLVMIR/GenOpsAttributes.cpp.inc"
