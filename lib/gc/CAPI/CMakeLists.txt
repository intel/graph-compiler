set(GC_ALL_LIBS 
  MLIROneDNNGraph
  MLIRCPURuntimeDialect
  GCPasses
  MLIRCPURuntimeTransforms)

get_property(gc_passes_libs GLOBAL PROPERTY GC_PASS_LIBS)
list(APPEND GC_ALL_LIBS ${gc_passes_libs})

add_mlir_public_c_api_library(GcCAPI
  Dialects.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
  ${GC_ALL_LIBS}
)