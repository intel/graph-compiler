if (NOT GC_ENABLE_TEST)
    message(STATUS "The tests are not enabled.")
    return()
endif ()

include(gtest)
add_subdirectory(dnnl)
add_subdirectory(mlir)

if(GC_BENCH_ENABLE)
    add_subdirectory(benchgc)
endif()