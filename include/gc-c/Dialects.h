#ifndef GC_MLIR_C_DIALECTS_H
#define GC_MLIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(OneDNNGraph, onednn_graph);

#ifdef __cplusplus
}
#endif
#endif // GC_MLIR_C_DIALECTS_H