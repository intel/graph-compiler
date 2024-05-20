#include "gc/Transforms/Passes.h"
#include "mlir-c/Pass.h"
#include "mlir/CAPI/Pass.h"

#include "gc/Transforms/Passes.capi.h.inc"
using namespace mlir::gc;

#ifdef __cplusplus
extern "C" {
#endif

#include "gc/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif