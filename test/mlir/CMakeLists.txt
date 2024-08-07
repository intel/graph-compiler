if (NOT GC_ENABLE_TEST_MLIR)
    message(STATUS "The mlir tests are not enabled.")
    return()
endif ()

add_subdirectory(test)
add_subdirectory(unittests)
