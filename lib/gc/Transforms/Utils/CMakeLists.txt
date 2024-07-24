add_mlir_library(GcUtilsIR
  MatcherUtils.cpp
  StructuredOpMatcher.cpp
  ValueUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

  DEPENDS
    MLIRLinalgDialect
)
