set(LLVM_TARGET_DEFINITIONS LinalgxDialect.td)
mlir_tablegen(LinalgxOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=linalgx)
mlir_tablegen(LinalgxOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=linalgx)

add_mlir_dialect(LinalgxOps linalgx)
set(LLVM_TARGET_DEFINITIONS LinalgxStructuredOps.td)
mlir_tablegen(LinalgxStructuredOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgxStructuredOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLinalgxStructuredOpsIncGen)

add_mlir_doc(LinalgxOps LinalgxOps gc/Dialect/Linalgx/ -gen-op-doc)
add_mlir_doc(LinalgxDialect LinalgxDialect gc/Dialect/Linalgx/ -gen-dialect-doc)
add_mlir_doc(LinalgxStructuredOps LinalgxStructuredOps gc/Dialect/Linalgx/ -gen-dialect-doc)
