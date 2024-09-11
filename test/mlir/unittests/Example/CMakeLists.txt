add_mlir_unittest(GCExampleTests
  Example.cpp
)
target_link_libraries(GCExampleTests
  PRIVATE
  MLIRLinalgx)
