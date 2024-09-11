if (NOT GC_ENABLE_DNNL_API)
    return()
endif ()

target_compile_options(GcInterface INTERFACE -DGC_HAS_ONEDNN_DIALECT)

gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRIR)
gc_add_mlir_dialect_library(${GC_ONEDNN_DIALECT_LIB_NAME}
    OneDNNGraphDialect.cpp
    OneDNNGraphOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/OneDNNGraph
    
    DEPENDS
    ${GC_ONEDNN_DIALECT_LIB_NAME}OpsIncGen
    
    LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
    GcInterface
)
