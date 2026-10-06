# Define the benchmark binary
add_executable(bench_distance bench_distance.cpp)
target_link_libraries(bench_distance PRIVATE secan_lib benchmark::benchmark)

add_executable(bench_exact_scan bench_exact_scan.cpp)
target_link_libraries(bench_exact_scan PRIVATE secan_lib benchmark::benchmark)

add_executable(bench_kd_tree bench_kd_tree.cpp)
target_link_libraries(bench_kd_tree PRIVATE secan_lib benchmark::benchmark)

add_executable(bench_index_sift bench_index_sift.cpp)
target_link_libraries(bench_index_sift PRIVATE secan_lib benchmark::benchmark)

add_executable(bench_quantization bench_quantization.cpp)
target_link_libraries(bench_quantization PRIVATE secan_lib benchmark::benchmark)



