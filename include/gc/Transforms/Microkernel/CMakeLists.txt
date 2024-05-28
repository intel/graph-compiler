set(LLVM_TARGET_DEFINITIONS MicrokernelPasses.td)
mlir_tablegen(MicrokernelPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRMicrokernelPassesIncGen)
add_mlir_doc(MicrokernelPasses GraphCompilerMicrokernelPasses ./ -gen-pass-doc)

set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS MLIRMicrokernelPasses)
