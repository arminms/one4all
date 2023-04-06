## add path to test data dir
#
add_definitions(-DONE4ALL_TEST_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")

## defining target for unit tests
#
add_executable(unit_tests
  unit_tests_${ONE4ALL_TARGET_API}.$<IF:$<STREQUAL:${ONE4ALL_TARGET_API},cuda>,cu,cpp>
)

## defining link libraries for unit tests
#
target_link_libraries(unit_tests PRIVATE
  ${PROJECT_NAME}::${ONE4ALL_TARGET_API}
  Catch2::Catch2WithMain
)

## finally adding unit tests
#
catch_discover_tests(unit_tests)
