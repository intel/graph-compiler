gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRFuncDialect)

add_mlir_dialect_library(MLIRCPURuntimeTransforms
        CPURuntimeToLLVM.cpp

        DEPENDS
        MLIRCPURuntimePassesIncGen

        LINK_LIBS PUBLIC
        ${MLIR_LINK_COMPONENTS}
        MLIRCPURuntimeDialect
        GcInterface
        )
target_link_libraries(obj.MLIRCPURuntimeTransforms PUBLIC GcInterface)
set_property(GLOBAL APPEND PROPERTY GC_PASS_LIBS MLIRCPURuntimeTransforms)