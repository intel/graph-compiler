if(GC_DEV_LINK_LLVM_DYLIB)
  set(LLVM_LINK_COMPONENTS
    LLVM
    )
  get_property(dialect_libs GLOBAL PROPERTY GC_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY GC_PASS_LIBS)
  set(MLIR_LINK_COMPONENTS
    MLIR
    MLIRExecutionEngineShared
  )
else()
  set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    native
  )
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(MLIR_LINK_COMPONENTS
    MLIRBuiltinToLLVMIRTranslation
    MLIRExecutionEngine
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRToLLVMIRTranslationRegistration
  )
endif()

set(GC_PASSES GcInterface GcPasses)
if(GC_UNABLE_GPU)
  list(APPEND GC_PASSES GcGpuPasses)
endif()

gc_add_mlir_library(GcJitWrapper
  Driver.cpp

  LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
    ${dialect_libs}
    ${conversion_libs}
    ${GC_PASSES}
    GcAnalysis
  )
