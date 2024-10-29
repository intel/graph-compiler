set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GCConversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GCConversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GCConversion)
add_public_tablegen_target(GCConversionPassIncGen)

add_mlir_doc(Passes GCConversionPasses ./ -gen-pass-doc)