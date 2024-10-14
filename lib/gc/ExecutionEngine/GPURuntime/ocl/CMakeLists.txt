find_package(OpenCL REQUIRED)

gc_add_mlir_library(GcGpuOclRuntime
  GpuOclRuntime.cpp
  LINK_LIBS PUBLIC
  GcInterface
  GcGpuPasses
)
target_include_directories(GcGpuOclRuntime PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(GcGpuOclRuntime PUBLIC ${OpenCL_LIBRARIES})

if(GC_ENABLE_GPU_PROFILE)
  include(ptigpu)
  get_property(GC_PTIGPU_BINARY_DIR GLOBAL PROPERTY GC_PTIGPU_BINARY_DIR)
  target_link_libraries(GcGpuOclRuntime PUBLIC ${GC_PTIGPU_BINARY_DIR}/lib/libonetrace_tool.so)
endif ()
