add_mlir_dialect_library(MLIROnednnGraph
    OnednnGraphDialect.cpp
    OnednnGraphOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc-dialects/OnednnGraph
    
    DEPENDS
    MLIROnednnGraphOpsIncGen
    
    LINK_LIBS PUBLIC
    MLIRIR
)