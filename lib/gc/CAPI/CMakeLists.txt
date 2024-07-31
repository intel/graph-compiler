set(GC_ALL_LIBS 
  MLIROneDNNGraph
  MLIRCPURuntimeDialect
  GCPasses
  MLIRCPURuntimeTransforms)

list(APPEND GC_ALL_LIBS GCGPUPasses)

add_mlir_public_c_api_library(GcCAPI
  Dialects.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
  ${GC_ALL_LIBS}
)