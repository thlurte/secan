# Define the benchmark binary
add_executable(bench_distance bench_distance.cpp)
target_link_libraries(bench_distance PRIVATE secan_lib benchmark::benchmark)

add_executable(bench_exact_scan bench_exact_scan.cpp)
target_link_libraries(bench_exact_scan PRIVATE secan_lib benchmark::benchmark)

add_executable(bench_kd_tree bench_kd_tree.cpp)
target_link_libraries(bench_kd_tree PRIVATE secan_lib benchmark::benchmark)

