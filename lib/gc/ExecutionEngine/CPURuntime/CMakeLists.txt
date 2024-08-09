find_package(OpenMP REQUIRED)

gc_add_mlir_library(GcCpuRuntime
    SHARED
    Parallel.cpp
    MemoryPool.cpp

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

include(umf)
get_property(UMF_INCLUDES GLOBAL PROPERTY UMF_INCLUDES)
target_include_directories(GcCpuRuntime PRIVATE ${UMF_INCLUDES})