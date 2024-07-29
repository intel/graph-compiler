add_mlir_dialect_library(MLIRXETarget
  XE/Target.cpp

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
  )