
set(LLVM_TARGET_DEFINITIONS OneDNNGraphTypes.td)
mlir_tablegen(OneDNNGraphOpsEnums.h.inc  -gen-enum-decls)
mlir_tablegen(OneDNNGraphOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(OneDNNGraphOpsAttributes.h.inc 
              -gen-attrdef-decls -attrdefs-dialect=onednn_graph)
mlir_tablegen(OneDNNGraphOpsAttributes.cpp.inc 
              -gen-attrdef-defs -attrdefs-dialect=onednn_graph)
add_mlir_dialect(OneDNNGraphOps onednn_graph)
add_mlir_doc(OneDNNGraphOps OneDNNGraphOps gc/Dialect/OneDNNGraph/ -gen-op-doc)
add_mlir_doc(OneDNNGraphDialect OneDNNGraphDialect gc/Dialect/OneDNNGraph/ -gen-dialect-doc)
