add_mlir_unittest(GCAnalysisTests
  TargetDescriptionAnalysisTest.cpp
)
target_link_libraries(GCAnalysisTests
  PRIVATE
  GcAnalysis
  GcJitWrapper)
