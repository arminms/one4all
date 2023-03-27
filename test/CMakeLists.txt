include_directories(../include)

## add path to test data dir
#
add_definitions(-DONE4ALL_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")

## defining target for unit_tests
#
if(${ONE4ALL_TARGET_API} STREQUAL cuda)
  add_executable(unit_tests unit_tests_${ONE4ALL_TARGET_API}.cu)
else()
  add_executable(unit_tests unit_tests_${ONE4ALL_TARGET_API}.cpp)
endif()
target_compile_features(unit_tests PRIVATE cxx_std_17)
target_compile_options(unit_tests PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},rocm>:-Wno-unused-result>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},oneapi>:-fsycl-targets=${SYCL_TARGETS}>
)
target_link_options(unit_tests PRIVATE
  $<$<STREQUAL:${ONE4ALL_TARGET_API},oneapi>:-fsycl-targets=${SYCL_TARGETS}>
)
target_link_libraries(unit_tests PRIVATE
  Catch2::Catch2WithMain
  $<$<STREQUAL:${ONE4ALL_TARGET_API},oneapi>:oneDPL>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},rocm>:${HIP_TARGET}>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},rocm>:roc::rocthrust>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},stl>:OpenMP::OpenMP_CXX>
  $<$<STREQUAL:${ONE4ALL_TARGET_API},stl>:TBB::tbb>
)
catch_discover_tests(unit_tests)
