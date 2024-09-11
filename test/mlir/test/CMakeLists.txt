if(WIN32)
    set(MLIR_RUNNER_UTILS_DIR ${LLVM_BINARY_DIR}/bin)
else()
    set(MLIR_RUNNER_UTILS_DIR ${LLVM_LIBRARY_DIR})
endif()

llvm_canonicalize_cmake_booleans(
        GC_ENABLE_BINDINGS_PYTHON
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
        FileCheck count not split-file
        # mlir-gen
        gc-opt
        gc-cpu-runner
        GCUnitTests
)

if(GC_ENABLE_IMEX)
        include(imex)
        list(APPEND GC_OPT_TEST_DEPENDS GcOpenclRuntime)
endif()

if(GC_ENABLE_BINDINGS_PYTHON)
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
