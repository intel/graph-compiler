gc_add_mlir_library(GcGpuPasses
  LinalgToXeGPU.cpp
  GPULegalizeModule.cpp
  ConvertGpuSignaturesToLLVM.cpp

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    MLIRGPUDialect
    MLIRXeGPUDialect
    MLIRGPUTransforms
    MLIRGPUToSPIRV
    MLIRSCFToGPU
    MLIRSCFToSPIRV
    MLIRMathToSPIRV
    MLIRControlFlowToSPIRV
    MLIRMemRefTransforms
    GcInterface
    GcUtilsIR
)

