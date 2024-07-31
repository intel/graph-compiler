include_guard()

get_property(GC_DNNL_INCLUDES GLOBAL PROPERTY GC_DNNL_INCLUDES)
if (NOT DEFINED GC_DNNL_INCLUDES)
    # TODO: Change to main https://github.com/oneapi-src/oneDNN.git when all the
    # required functionality is merged.
    gc_fetch_content(dnnl dev https://github.com/kurapov-peter/oneDNN.git
            SKIP_ADD
    )

    set(GC_DNNL_INCLUDES
        $<BUILD_INTERFACE:${dnnl_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${dnnl_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${dnnl_SOURCE_DIR}/src>
    )
    set_property(GLOBAL PROPERTY GC_DNNL_INCLUDES ${GC_DNNL_INCLUDES})

    # This allows to generate headers from *.in without adding the library to the build.
    # If the build is required, remove this and the SKIP_ADD option above.
    if (DEFINED CMAKE_GENERATOR)
        set(GENERATOR_FLAG "-G ${CMAKE_GENERATOR}")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} ${GENERATOR_FLAG}
            -Wno-dev
            -S ${dnnl_SOURCE_DIR}
            -B ${dnnl_BINARY_DIR}
            -DDNNL_IS_MAIN_PROJECT=FALSE -DDNNL_BUILD_TESTS=FALSE -DDNNL_BUILD_EXAMPLES=FALSE
            COMMAND_ERROR_IS_FATAL ANY
    )
endif ()
