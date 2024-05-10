add_mlir_unittest(MLIRSCFTests
  EasyBuildTest.cpp)
target_link_libraries(MLIRSCFTests
  PRIVATE
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRArithDialect)
