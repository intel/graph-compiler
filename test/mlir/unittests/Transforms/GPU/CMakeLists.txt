add_mlir_unittest(GpuTransformsTests
  GpuUtilsTest.cpp
)
target_link_libraries(GpuTransformsTests PRIVATE GcGpuPasses)
target_include_directories(GpuTransformsTests PRIVATE ${PROJECT_SOURCE_DIR}/lib)
