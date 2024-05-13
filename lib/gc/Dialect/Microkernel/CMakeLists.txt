if(GC_LINK_DYNAMIC_LLVM)
  set(MLIR_LINK_COMPONENTS
    MLIR
  )
else()
  set(MLIR_LINK_COMPONENTS
    MLIRIR
  )
endif()

add_mlir_dialect_library(MLIRMicrokernel
    MicrokernelDialect.cpp
    MicrokernelOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/Microkernel
    
    DEPENDS
    MLIRMicrokernelOpsIncGen
    
    LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
)