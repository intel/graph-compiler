find_package(OpenCL REQUIRED)

gc_add_mlir_library(GcGpuOclRuntime
  GpuOclRuntime.cpp
  LINK_LIBS PUBLIC
  GcInterface
  GcGpuPasses
)
target_include_directories(GcGpuOclRuntime PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(GcGpuOclRuntime PUBLIC ${OpenCL_LIBRARIES})
