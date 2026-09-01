cmake_minimum_required(VERSION 3.15)
project(secan VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- library: all real logic, no main() ---
add_library(secan_lib
  src/utils/utils.cpp
  src/search/search.cpp
  src/search/distance.cpp
)
target_include_directories(secan_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# --- dependency: google benchmark, evaluates search algos ---
include(FetchContent)

# Disable internal Google Benchmark tests and docs (prevents downloading old GoogleTest)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.0
)
FetchContent_MakeAvailable(googlebenchmark)

add_subdirectory(benchmarks)

# --- executable: just main.cpp, links the library ---
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE secan_lib)

find_library(PROFILER_LIB profiler)
if(PROFILER_LIB)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${PROFILER_LIB})
endif()

# --- tests ---
enable_testing()
add_subdirectory(tests)
