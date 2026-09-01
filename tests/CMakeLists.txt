add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance PRIVATE secan_lib)
target_include_directories(test_distance PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_linear_scan test_linear_scan.cpp)
target_link_libraries(test_linear_scan PRIVATE secan_lib)
target_include_directories(test_linear_scan PRIVATE ${CMAKE_SOURCE_DIR}/include)


enable_testing()
add_test(NAME distance_tests COMMAND test_distance)
add_test(NAME linear_scan_tests COMMAND test_linear_scan)
