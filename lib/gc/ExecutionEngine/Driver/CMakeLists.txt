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

get_property(gc_passes_libs GLOBAL PROPERTY GC_PASS_LIBS)

add_mlir_library(GCJitWrapper
  Driver.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

  LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
    ${dialect_libs}
    ${conversion_libs}
    ${gc_passes_libs}
  )

