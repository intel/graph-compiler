set(LLVM_TARGET_DEFINITIONS MicrokernelEnum.td)
mlir_tablegen(MicrokernelEnum.h.inc -gen-enum-decls)
mlir_tablegen(MicrokernelEnum.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMicrokernelAttrDefIncGen)

add_mlir_dialect(MicrokernelOps microkernel)
add_mlir_doc(MicrokernelOps MicrokernelOps gc/Dialect/Microkernel/ -gen-op-doc)
add_mlir_doc(MicrokernelDialect MicrokernelDialect gc/Dialect/Microkernel/ -gen-dialect-doc)
