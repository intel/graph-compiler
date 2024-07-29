gc_add_mlir_library(GcGpuPasses
  LinalgToXeGPU.cpp
  GPULegalizeModule.cpp
  ConvertGpuSignaturesToLLVM.cpp
  GPUAttachGenTarget.cpp

  DEPENDS
    GraphCompilerPassIncGen

  LINK_LIBS PUBLIC
    MLIRGENDialect
    MLIRGENTarget
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

