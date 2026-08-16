add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance PRIVATE secan_lib)

add_executable(test_linear_scan test_linear_scan.cpp)
target_link_libraries(test_linear_scan PRIVATE secan_lib)

add_test(NAME distance_tests COMMAND test_distance)
add_test(NAME linear_scan_tests COMMAND test_linear_scan)
