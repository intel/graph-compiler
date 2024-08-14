add_mlir_unittest(GCTransformsTests
  StaticMemoryPlanning.cpp
)
target_link_libraries(GCTransformsTests
  PRIVATE
  GcPasses)
