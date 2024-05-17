#include "gc/Dialect/Microkernel/MicrokernelDialect.h"
#include "gc/Dialect/Microkernel/MicrokernelOps.h"

using namespace mlir;
using namespace mlir::microkernel;

void MicrokernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gc/Dialect/Microkernel/MicrokernelOps.cpp.inc"
      >();
}
