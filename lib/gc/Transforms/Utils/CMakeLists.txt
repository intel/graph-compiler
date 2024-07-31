gc_add_mlir_library(GcUtilsIR
  MatcherUtils.cpp
  StructuredOpMatcher.cpp
  ValueUtils.cpp

  DEPENDS
    MLIRLinalgDialect

  LINK_LIBS PUBLIC
    GcInterface
)
