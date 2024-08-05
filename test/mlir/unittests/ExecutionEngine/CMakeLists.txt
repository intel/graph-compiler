add_mlir_unittest(GCExecutionEngineTests
  JitWrapper.cpp
)
# todo: remove once we have a gpu runtime library
get_property(gc_passes_libs GLOBAL PROPERTY GC_PASS_LIBS)
target_link_libraries(GCExecutionEngineTests
  PRIVATE
  GcJitWrapper
  GcCpuRuntime
  ${gc_passes_libs})
