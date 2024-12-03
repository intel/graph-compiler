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
  find_package(Pti REQUIRED)
  target_link_libraries(GcGpuOclRuntime PRIVATE Pti::pti_view)
  add_definitions(-DGC_ENABLE_GPU_PROFILE)
endif ()
