add_mlir_dialect_library(MLIRGENDialect
  IR/GENDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR
  ${PROJECT_SOURCE_DIR}/include/gc/Dialect/LLVMIR

  DEPENDS
#   MLIRGPUCompilationAttrInterfacesIncGen
#   MLIRGENOpsIncGen
  MLIRGENConversionsIncGen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )
set_property(GLOBAL APPEND PROPERTY GC_DIALECT_LIBS MLIRGENDialect)
