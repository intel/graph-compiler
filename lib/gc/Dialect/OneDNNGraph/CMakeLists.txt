if (NOT GC_ENABLE_TEST_DNNL)
    return()
endif ()

gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRIR)
add_mlir_dialect_library(${GC_ONEDNN_DIALECT_LIB_NAME}
    OneDNNGraphDialect.cpp
    OneDNNGraphOps.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Dialect/OneDNNGraph
    
    DEPENDS
    ${GC_ONEDNN_DIALECT_LIB_NAME}OpsIncGen
    
    LINK_LIBS PUBLIC
    ${MLIR_LINK_COMPONENTS}
)
set_property(GLOBAL APPEND PROPERTY GC_DIALECT_LIBS ${GC_ONEDNN_DIALECT_LIB_NAME})
