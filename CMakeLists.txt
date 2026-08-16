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

# --- executable: just main.cpp, links the library ---
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE secan_lib)

# --- tests ---
enable_testing()
add_subdirectory(tests)
