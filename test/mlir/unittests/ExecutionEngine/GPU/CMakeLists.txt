add_mlir_unittest(GCExecutionEngineGpuTests
  GpuOclRuntimeTest.cpp
)
target_link_libraries(GCExecutionEngineGpuTests
  PRIVATE
  GcJitWrapper
  GcGpuOclRuntime
)
