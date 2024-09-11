add_mlir_unittest(GCExecutionEngineTests
  JitWrapper.cpp
  BrgemmRuntime.cpp
)

get_property(GC_DNNL_INCLUDES GLOBAL PROPERTY GC_DNNL_INCLUDES)
target_include_directories(GCExecutionEngineTests PRIVATE ${GC_DNNL_INCLUDES})

target_link_libraries(GCExecutionEngineTests
  PRIVATE
  GcJitWrapper
  GcCpuRuntime)

