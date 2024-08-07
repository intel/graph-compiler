gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRIR)

gc_add_mlir_dialect_library(MLIRLinalgx
    LinalgxDialect.cpp
    LinalgxOps.cpp
    Utils.cpp
    

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/Linalgx

    DEPENDS
    MLIRLinalgxOpsIncGen
    MLIRLinalgxStructuredOpsIncGen
    
    LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
    GcInterface
)