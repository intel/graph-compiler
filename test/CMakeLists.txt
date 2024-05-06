if (NOT GC_TEST_ENABLE)
    message(STATUS "The tests are not enabled.")
    return()
endif ()

include(gtest)

# Static library to be used in tests
add_library(graph_compiler_static STATIC ${GC_LIB_SOURCES})
target_include_directories(graph_compiler_static PUBLIC ${GC_LIB_INCLUDES})

add_subdirectory(dnnl)
