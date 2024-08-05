gc_add_mlir_dialect_library(MLIRGENTarget
  GEN/Target.cpp

  OBJECT

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR
  ${PROJECT_SOURCE_DIR}/include/gc/Dialect/LLVMIR

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
  SPIRVCodeGen
  MLIRGENToLLVMIRTranslation
  GcInterface
  )
