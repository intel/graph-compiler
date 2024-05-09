add_mlir_unittest(MLIRArithTests
  EasyBuildTest.cpp)
target_link_libraries(MLIRArithTests
  PRIVATE
  MLIRFuncDialect
  MLIRArithDialect)
