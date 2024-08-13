include_guard()

get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
if (NOT DEFINED IMEX_INCLUDES)
    if(GC_DEV_LINK_LLVM_DYLIB)
        message(WARN "GPU backend may not be compatible with dynamic linking to LLVM")
    endif()

    # TODO: Change to main https://github.com/intel/mlir-extensions when all the
    # required functionality is merged.
    gc_fetch_content(imex d5bbd635dee500b8cff138686833bacfac5ade78 https://github.com/Menooker/mlir-extensions
            SET IMEX_CHECK_LLVM_VERSION=ON IMEX_ENABLE_L0_RUNTIME=0
    )

    set(IMEX_INCLUDES
            ${imex_BINARY_DIR}/include
            ${imex_SOURCE_DIR}/include
            ${imex_SOURCE_DIR}/src
    )
    set_property(GLOBAL PROPERTY IMEX_INCLUDES ${IMEX_INCLUDES})
endif ()
