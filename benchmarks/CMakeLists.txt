# Define the benchmark binary
add_executable(bench_distance bench_distance.cpp)

# Link your library and Google Benchmark
target_link_libraries(bench_distance PRIVATE secan_lib benchmark::benchmark)
