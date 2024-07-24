if (NOT GC_ENABLE_TEST_DNNL)
    message(STATUS "OneDNNGraphDialect is not enabled.")
    return()
endif ()

add_mlir_dialect(OneDNNGraphOps onednn_graph)
add_mlir_doc(OneDNNGraphOps OneDNNGraphOps gc/Dialect/OneDNNGraph/ -gen-op-doc)
add_mlir_doc(OneDNNGraphDialect OneDNNGraphDialect gc/Dialect/OneDNNGraph/ -gen-dialect-doc)
