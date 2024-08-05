gc_add_mlir_dialect_library(MLIRGENDialect
  IR/GENDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR
  ${PROJECT_SOURCE_DIR}/include/gc/Dialect/LLVMIR

  DEPENDS
  MLIRGENConversionsIncGen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  GcInterface
)
