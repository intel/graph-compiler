#include "gc-c/Dialects.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(OneDNNGraph, onednn_graph,
                                      mlir::onednn_graph::OneDNNGraphDialect)