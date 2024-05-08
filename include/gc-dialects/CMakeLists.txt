add_subdirectory(OnednnGraph)
add_subdirectory(Microkernel)
add_subdirectory(Linalgx)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GraphCompiler)
add_public_tablegen_target(GraphCompilerPassIncGen)
add_mlir_doc(Passes GraphCompilerPasses ./ -gen-pass-doc)
