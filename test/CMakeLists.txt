if (NOT GC_TEST_ENABLE)
    message(STATUS "The tests are not enabled.")
    return()
endif ()

include(gtest)

# Static library to be used in tests
add_library(graph_compiler_static STATIC ${GC_LIB_SOURCES})
target_include_directories(graph_compiler_static PUBLIC ${GC_LIB_INCLUDES})

add_subdirectory(dnnl)

llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)
configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/gc/Unit/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/gc/Unit/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/gc/Unit/lit.cfg.py
)

set(GC_OPT_TEST_DEPENDS
        FileCheck count not
        # mlir-gen
        gc-opt
        gc-cpu-runner
        GCUnitTests
        )

if(MLIR_ENABLE_BINDINGS_PYTHON)
        list(APPEND GC_OPT_TEST_DEPENDS GcPythonModules)
endif()

add_lit_testsuite(gc-check "Running the regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${GC_OPT_TEST_DEPENDS}
        )

# Looks that this property is suitable for IDE
# TODO: Check is this fine for IDE
set_target_properties(gc-check PROPERTIES FOLDER "Tests")

add_lit_testsuites(GC_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${GC_OPT_TEST_DEPENDS})

