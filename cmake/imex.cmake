include_guard()

get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
if (NOT DEFINED IMEX_INCLUDES)
    if(GC_DEV_LINK_LLVM_DYLIB)
        message(WARN "GPU backend may not be compatible with dynamic linking to LLVM")
    endif()

    # Read the content of imex-version.txt
    file(READ "${CMAKE_CURRENT_LIST_DIR}/imex-version.txt" IMEX_HASH)

    # Strip any extra whitespace or newlines
    string(STRIP "${IMEX_HASH}" IMEX_HASH)

    # TODO: Change to main https://github.com/intel/mlir-extensions when all the
    # required functionality is merged.
    gc_fetch_content(imex "${IMEX_HASH}" https://github.com/intel/mlir-extensions
            SET IMEX_CHECK_LLVM_VERSION=ON IMEX_ENABLE_L0_RUNTIME=0
    )

    set(IMEX_INCLUDES
            ${imex_BINARY_DIR}/include
            ${imex_SOURCE_DIR}/include
            ${imex_SOURCE_DIR}/src
    )
    set_property(GLOBAL PROPERTY IMEX_INCLUDES ${IMEX_INCLUDES})
    target_compile_options(GcInterface INTERFACE -DGC_USE_IMEX)
endif ()
