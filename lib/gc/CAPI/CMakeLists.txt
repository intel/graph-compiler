set(GC_ALL_LIBS
  ${GC_ONEDNN_DIALECT_LIB_NAME}
  GcPasses
  MLIRCPURuntimeTransforms)

if(GC_ENABLE_IMEX)
  list(APPEND GC_ALL_LIBS GcGpuPasses)
endif()

add_mlir_public_c_api_library(GcCAPI
  Dialects.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
  ${GC_ALL_LIBS}
)
target_link_libraries(obj.GcCAPI PUBLIC GcInterface)
set_property(GLOBAL APPEND PROPERTY GC_MLIR_LIBS GcCAPI)
