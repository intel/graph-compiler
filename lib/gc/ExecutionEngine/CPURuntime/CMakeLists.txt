find_package(OpenMP REQUIRED)

file(GLOB_RECURSE MICROKERNEL_RUNTIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/*.c
    )

if (GC_ENABLE_RUNTIME_NAIVE_BRGEMM)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/BrgemmOnednn.cpp" "" MICROKERNEL_RUNTIME_SOURCES "${MICROKERNEL_RUNTIME_SOURCES}")
else()
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/BrgemmNaive.cpp" "" MICROKERNEL_RUNTIME_SOURCES "${MICROKERNEL_RUNTIME_SOURCES}")
endif()

include(onednn)

get_property(GC_DNNL_INCLUDES GLOBAL PROPERTY GC_DNNL_INCLUDES)
get_property(GC_DNNL_LIB_DEPS GLOBAL PROPERTY GC_DNNL_LIB_DEPS)

include_directories(${GC_DNNL_INCLUDES})

gc_add_mlir_library(GcCpuRuntime
    SHARED
    Parallel.cpp
    ${MICROKERNEL_RUNTIME_SOURCES}

    DEPENDS
    dnnl_brgemm

    LINK_LIBS PRIVATE
    ${GC_DNNL_LIB_DEPS}

    LINK_LIBS PUBLIC
    GcInterface

    EXCLUDE_FROM_LIBMLIR
)

if ("iomp" IN_LIST OpenMP_C_LIB_NAMES OR "omp" IN_LIST OpenMP_C_LIB_NAMES OR "omp5" IN_LIST OpenMP_C_LIB_NAMES)
else()
  target_compile_options(GcCpuRuntime PRIVATE "-DGC_NEEDS_OMP_WRAPPER")
endif()

string(REPLACE " " ";" flags "${OpenMP_C_FLAGS} ${OpenMP_CXX_FLAGS}")
target_compile_options(GcCpuRuntime PUBLIC ${flags})
target_link_options(GcCpuRuntime PUBLIC ${flags})
