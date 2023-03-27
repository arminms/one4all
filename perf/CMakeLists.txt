include_directories(../include)

## defining target for benchmarks
#
if(${ONE4ALL_TARGET_API} STREQUAL cuda)
  add_executable(benchmarks benchmarks_${ONE4ALL_TARGET_API}.cu)
else()
  add_executable(benchmarks benchmarks_${ONE4ALL_TARGET_API}.cpp)
endif()
target_compile_features(benchmarks PRIVATE cxx_std_17)
target_compile_options(benchmarks PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},rocm>:-Wno-unused-result>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},oneapi>:-fsycl-targets=${SYCL_TARGETS}>
)
target_link_options(benchmarks PRIVATE
$<$<STREQUAL:${ONE4ALL_TARGET_API},oneapi>:-fsycl-targets=${SYCL_TARGETS}>
)
target_link_libraries(benchmarks PRIVATE
  benchmark::benchmark
  $<$<STREQUAL:${ONE4ALL_TARGET_API},oneapi>:oneDPL>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},rocm>:${HIP_TARGET}>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},rocm>:roc::rocthrust>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},stl>:OpenMP::OpenMP_CXX>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},stl>:TBB::tbb>
)
