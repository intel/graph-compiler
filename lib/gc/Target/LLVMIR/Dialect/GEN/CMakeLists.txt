gc_add_mlir_translation_library(MLIRGENToLLVMIRTranslation
  GENToLLVMIRTranslation.cpp

  DEPENDS
  MLIRGENConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRGENDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  GcInterface
  )
