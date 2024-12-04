find_package(OpenMP REQUIRED)

file(GLOB_RECURSE MICROKERNEL_RUNTIME_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Microkernel/*.c
    )

if (GC_ENABLE_RUNTIME_NAIVE_BRGEMM)
    add_definitions("-DGC_ENABLE_RUNTIME_NAIVE_BRGEMM=1")
endif()

include(onednn)

get_property(GC_DNNL_INCLUDES GLOBAL PROPERTY GC_DNNL_INCLUDES)
get_property(GC_DNNL_LIB_DEPS GLOBAL PROPERTY GC_DNNL_LIB_DEPS)

gc_add_mlir_library(GcCpuRuntime
    Parallel.cpp
    MemoryPool.cpp
    ${MICROKERNEL_RUNTIME_SOURCES}

    LINK_LIBS PUBLIC
    GcInterface
)

target_include_directories(GcCpuRuntime PRIVATE ${GC_DNNL_INCLUDES})

if ("iomp" IN_LIST OpenMP_C_LIB_NAMES OR "omp" IN_LIST OpenMP_C_LIB_NAMES OR "omp5" IN_LIST OpenMP_C_LIB_NAMES)
else()
  target_compile_options(GcCpuRuntime PRIVATE "-DGC_NEEDS_OMP_WRAPPER")
endif()

string(REPLACE " " ";" flags "${OpenMP_C_FLAGS} ${OpenMP_CXX_FLAGS}")
target_compile_options(GcCpuRuntime PUBLIC ${flags})
target_link_options(GcCpuRuntime PUBLIC ${flags})
