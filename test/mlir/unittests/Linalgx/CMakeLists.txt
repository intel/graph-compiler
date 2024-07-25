add_mlir_unittest(GCLinalgxTests
  TestUtils.cpp
)
target_link_libraries(GCLinalgxTests
  PRIVATE
  MLIRLinalgx)
