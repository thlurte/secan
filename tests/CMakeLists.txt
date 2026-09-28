add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance PRIVATE secan_lib)
target_include_directories(test_distance PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_linear_scan test_linear_scan.cpp)
target_link_libraries(test_linear_scan PRIVATE secan_lib)
target_include_directories(test_linear_scan PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_io test_io.cpp)
target_link_libraries(test_io PRIVATE secan_lib)
target_include_directories(test_io PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_ir_metrics test_ir_metrics.cpp)
target_link_libraries(test_ir_metrics PRIVATE secan_lib)
target_include_directories(test_ir_metrics PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_normalize test_normalize.cpp)
target_link_libraries(test_normalize PRIVATE secan_lib)
target_include_directories(test_normalize PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_kd_tree test_kd_tree.cpp)
target_link_libraries(test_kd_tree PRIVATE secan_lib)
target_include_directories(test_kd_tree PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_ivf_flat test_ivf_flat.cpp)
target_link_libraries(test_ivf_flat PRIVATE secan_lib)
target_include_directories(test_ivf_flat PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_flat test_flat.cpp)
target_link_libraries(test_flat PRIVATE secan_lib)
target_include_directories(test_flat PRIVATE ${CMAKE_SOURCE_DIR}/include)

enable_testing()
add_test(NAME distance_tests COMMAND test_distance)
add_test(NAME linear_scan_tests COMMAND test_linear_scan)
add_test(NAME io_tests COMMAND test_io)
add_test(NAME ir_metrics_tests COMMAND test_ir_metrics)
add_test(NAME normalize_tests COMMAND test_normalize)
add_test(NAME kd_tree_tests COMMAND test_kd_tree)
add_test(NAME flat_tests COMMAND test_flat)
add_test(NAME ivf_flat_tests COMMAND test_ivf_flat)

