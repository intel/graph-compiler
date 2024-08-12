find_package(OpenMP REQUIRED)

if ("iomp" IN_LIST OpenMP_C_LIB_NAMES OR "omp" IN_LIST OpenMP_C_LIB_NAMES OR "omp5" IN_LIST OpenMP_C_LIB_NAMES)
else()
  add_definitions("-DGC_NEEDS_OMP_WRAPPER=1")
endif()

file(GLOB_RECURSE MICROKERNEL_RUNTIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/*.c
    )

if (GC_RUNTIME_NAIVE_BRGEMM)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/BrgemmOnednn.cpp" "" MICROKERNEL_RUNTIME_SOURCES "${MICROKERNEL_RUNTIME_SOURCES}")
else()
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/BrgemmNaive.cpp" "" MICROKERNEL_RUNTIME_SOURCES "${MICROKERNEL_RUNTIME_SOURCES}")
endif()

include(onednn)

get_property(DNNL_INCLUDES GLOBAL PROPERTY DNNL_INCLUDES)
get_property(DNNL_LIB_DEPS GLOBAL PROPERTY DNNL_LIB_DEPS)

include_directories(${DNNL_INCLUDES})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
add_mlir_library(GCCpuRuntime
    SHARED
    Parallel.cpp
    MemoryPool.cpp
    UnifiedMemoryFramework.cpp
    ${MICROKERNEL_RUNTIME_SOURCES}

    DEPENDS
    dnnl_brgemm

    LINK_LIBS PRIVATE
    ${DNNL_LIB_DEPS}

    EXCLUDE_FROM_LIBMLIR
  )

include(umf)
get_property(UMF_INCLUDES GLOBAL PROPERTY UMF_INCLUDES)
target_include_directories(GCCpuRuntime PRIVATE ${UMF_INCLUDES})
target_link_libraries(GCCpuRuntime PRIVATE umf)
