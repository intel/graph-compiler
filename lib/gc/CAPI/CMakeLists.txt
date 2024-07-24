set(GC_ALL_LIBS 
  MLIROneDNNGraph
  ${GC_ONEDNN_DIALECT_LIB_NAME}
  GcPasses
  MLIRCPURuntimeTransforms)

if(GC_USE_GPU)
  list(APPEND GC_ALL_LIBS GcGPUPasses)
endif()

add_mlir_public_c_api_library(GcCAPI
  Dialects.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
  ${GC_ALL_LIBS}
)
