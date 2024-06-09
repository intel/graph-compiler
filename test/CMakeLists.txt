if (NOT GC_TEST_ENABLE)
    message(STATUS "The tests are not enabled.")
    return()
endif ()

include(gtest)

# Static graph compiler library to be used in tests
add_mlir_library(graph_compiler_static STATIC ${GC_LIB_SOURCES})
target_compile_options(obj.graph_compiler_static PUBLIC -fexceptions)
target_include_directories(graph_compiler_static PUBLIC ${GC_LIB_INCLUDES})
target_link_libraries(graph_compiler_static PUBLIC ${GC_LIB_LINKED_LIBS})

add_subdirectory(dnnl)
add_subdirectory(mlir)
