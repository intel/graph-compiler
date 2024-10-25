gc_add_mlir_conversion_library(MLIRXeVMToLLVM
  XeVMToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/gc/Conversion/XeVMToLLVM

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
  MLIRPass
  MLIRTransforms
)