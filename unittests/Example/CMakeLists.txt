add_mlir_unittest(GCExampleTests
  Example.cpp
  OneDNNGraph.cpp
)
target_link_libraries(GCExampleTests
  PRIVATE
  MLIROneDNNGraph
  MLIRLinalgx)
