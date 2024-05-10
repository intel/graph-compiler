add_mlir_dialect_library(MLIRLinalgx
    LinalgxDialect.cpp
    LinalgxOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/Linalgx
    
    DEPENDS
    MLIRLinalgxOpsIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
)