if(GC_LINK_DYNAMIC_LLVM)
  set(MLIR_LINK_COMPONENTS
    MLIR
  )
else()
  set(MLIR_LINK_COMPONENTS
    MLIRIR
  )
endif()

add_mlir_dialect_library(MLIRLinalgx
    LinalgxDialect.cpp
    LinalgxOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/Linalgx
    
    DEPENDS
    MLIRLinalgxOpsIncGen
    
    LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
)
set_property(GLOBAL APPEND PROPERTY GC_DIALECT_LIBS MLIRLinalgx)