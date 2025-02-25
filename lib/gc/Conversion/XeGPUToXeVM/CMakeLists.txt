gc_add_mlir_conversion_library(MLIRXeGPUToXeVM
  XeGPUToXeVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/gc/Conversion/XeGPUToXeVM

  DEPENDS
  GCConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRXeVMDialect
  MLIRVectorDialect
  MLIRArithDialect
  MLIRXeGPUDialect
  MLIRPass
  MLIRTransforms
)