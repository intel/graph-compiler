gc_set_mlir_link_components(MLIR_LINK_COMPONENTS MLIRFuncDialect)

gc_add_mlir_dialect_library(MLIRCPURuntimeDialect
        CPURuntimeDialect.cpp
        CPURuntimeOps.cpp

        DEPENDS
        MLIRCPURuntimeOpsIncGen
        MLIRCPURuntimePassesIncGen

        LINK_LIBS PUBLIC
        ${MLIR_LINK_COMPONENTS}
        GcInterface
        )
