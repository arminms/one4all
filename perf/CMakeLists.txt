## defining target for benchmarks
#
add_executable(benchmarks
  benchmarks_${ONE4ALL_TARGET_API}.$<IF:$<STREQUAL:${ONE4ALL_TARGET_API},cuda>,cu,cpp>
)

## defining link libraries for benchmarks
#
target_link_libraries(benchmarks PRIVATE
  ${PROJECT_NAME}::${ONE4ALL_TARGET_API}
  benchmark::benchmark
)
