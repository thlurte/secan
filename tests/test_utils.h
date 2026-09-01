#pragma once
#include <chrono>
#include <iostream>
#include <string>

inline int g_tests_run = 0;
inline int g_tests_failed = 0;

inline void print_clock_overhead() {
  std::chrono::time_point _t1 = std::chrono::system_clock::now();
  std::chrono::time_point _t2 = std::chrono::system_clock::now();

  std::cout
      << "Clock Overhead "
      << std::chrono::duration_cast<std::chrono::nanoseconds>(_t2 - _t1).count()
      << "ns" << std::endl;
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    g_tests_run++;                                                             \
    std::chrono::time_point t1 = std::chrono::system_clock::now();             \
    bool passed{(condition)};                                                  \
    std::chrono::time_point t2 = std::chrono::system_clock::now();             \
    if (!(passed)) {                                                           \
      g_tests_failed++;                                                        \
      std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                    \
                << "  CHECK(" #condition ") failed\n";                         \
    }                                                                          \
    std::cerr << "Test took: "                                                 \
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1) \
                     .count()                                                  \
              << "ns" << std::endl;                                            \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
  do {                                                                         \
    g_tests_run++;                                                             \
    std::chrono::time_point t1 = std::chrono::system_clock::now();             \
    bool failed{(std::abs((a) - (b)) > (eps))};                                \
    std::chrono::time_point t2 = std::chrono::system_clock::now();             \
                                                                               \
    if (failed) {                                                              \
      g_tests_failed++;                                                        \
      std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                    \
                << "  CHECK_NEAR(" #a ", " #b ") failed: " << (a) << " vs "    \
                << (b) << "\n";                                                \
    }                                                                          \
    std::cerr << "Test took: "                                                 \
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1) \
                     .count()                                                  \
              << "ns" << std::endl;                                            \
  } while (0)

inline int report_results(const std::string &suite_name) {
  std::cout << suite_name << ": " << (g_tests_run - g_tests_failed) << "/"
            << g_tests_run << " passed\n";
  print_clock_overhead();
  return g_tests_failed == 0 ? 0 : 1;
}
