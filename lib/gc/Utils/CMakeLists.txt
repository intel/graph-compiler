add_mlir_library(GCMLIRUtils
  StructuredOpMatcher.cpp
  ValueUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/gc/Utils
)
