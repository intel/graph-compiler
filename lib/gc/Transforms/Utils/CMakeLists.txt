add_mlir_library(GCUtilsIR
  MatcherUtils.cpp
  StructuredOpMatcher.cpp
  ValueUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include

  DEPENDS
    MLIRLinalgDialect
)
