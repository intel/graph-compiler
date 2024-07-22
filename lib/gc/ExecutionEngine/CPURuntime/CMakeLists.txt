find_package(OpenMP REQUIRED)

if ("iomp" IN_LIST OpenMP_C_LIB_NAMES OR "omp" IN_LIST OpenMP_C_LIB_NAMES OR "omp5" IN_LIST OpenMP_C_LIB_NAMES)
else()
  add_definitions("-DGC_NEEDS_OMP_WRAPPER=1")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_mlir_library(GCCpuRuntime
    SHARED
    Parallel.cpp
    Memorypool.cpp
    ThreadLocals.cpp
    EXCLUDE_FROM_LIBMLIR

    ADDITIONAL_HEADER_DIRS
      ${PROJECT_SOURCE_DIR}/include
  )
