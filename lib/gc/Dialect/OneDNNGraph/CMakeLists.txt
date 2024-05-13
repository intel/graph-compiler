add_mlir_dialect_library(MLIROneDNNGraph
    OneDNNGraphDialect.cpp
    OneDNNGraphOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/OneDNNGraph
    
    DEPENDS
    MLIROneDNNGraphOpsIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
)