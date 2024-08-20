add_mlir_unittest(GCExecutionEngineTests
  JitWrapper.cpp
  BrgemmRuntime.cpp
)

target_link_libraries(GCExecutionEngineTests
  PRIVATE
  GcJitWrapper
  GcCpuRuntime)
