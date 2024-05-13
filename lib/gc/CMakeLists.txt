if(GC_MLIR_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GC_MLIR_CXX_FLAGS}")
endif()

include(../../cmake/functions.cmake)

add_subdirectory(Dialect)
add_subdirectory(Transforms)