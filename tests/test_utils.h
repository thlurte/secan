#pragma once
#include <iostream>
#include <string>

inline int g_tests_run = 0;
inline int g_tests_failed = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    g_tests_run++;                                                             \
    if (!(condition)) {                                                        \
      g_tests_failed++;                                                        \
      std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                    \
                << "  CHECK(" #condition ") failed\n";                         \
    }                                                                          \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
  do {                                                                         \
    g_tests_run++;                                                             \
    if (std::abs((a) - (b)) > (eps)) {                                         \
      g_tests_failed++;                                                        \
      std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                    \
                << "  CHECK_NEAR(" #a ", " #b ") failed: " << (a) << " vs "    \
                << (b) << "\n";                                                \
    }                                                                          \
  } while (0)

inline int report_results(const std::string &suite_name) {
  std::cout << suite_name << ": " << (g_tests_run - g_tests_failed) << "/"
            << g_tests_run << " passed\n";
  return g_tests_failed == 0 ? 0 : 1;
}
