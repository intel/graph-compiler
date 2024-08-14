add_mlir_unittest(GCExecutionEngineTests
  JitWrapper.cpp
)
target_link_libraries(GCExecutionEngineTests
  PRIVATE
  GcJitWrapper
  GcCpuRuntime)
