add_mlir_public_c_api_library(GcCAPI
  Dialects.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
  MLIROneDNNGraph
  MLIRCPURuntimeDialect
  GCPasses
  MLIRCPURuntimeTransforms
  MLIRMicrokernel MLIRMicrokernelTransforms GCMLIRUtils MLIRLinalgx GCAnalysis MLIRLinalgxTransforms
)