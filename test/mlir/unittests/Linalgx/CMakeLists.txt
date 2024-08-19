add_mlir_unittest(GCLinalgxTests
  TestUtils.cpp
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(GCLinalgxTests
  PRIVATE
  MLIRLinalgx
  ${dialect_libs})
