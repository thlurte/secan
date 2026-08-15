#include <chrono>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "secan/search/search.h"
#include "secan/utils/utils.h"

int main() {
  std::cout << "Test Suite 1.0 | Generate Random Numbers \n";
  std::cout << "=====================================\n";

  std::cout << "Starting Test Suite 1.0\n";
  auto [data, shape] = generate(0.0f, 1.0f, {10000, 128});

  std::cout << "Completed Test Suite 1.0 | Dataset\n";
  auto [query, q_shape] = generate(0.0f, 1.0f, {128});
  std::cout << "Completed Test Suite 1.0 | Query\n\n";

  std::cout << "Test Suite 2.0 | Linear Scan w/ Cosine \n";
  std::cout << "=====================================\n";

  auto cos_start_time = std::chrono::high_resolution_clock::now();
  std::vector<SearchResult> cos_results = linear_scan(data, query, 15);
  std::cout << "First element found in top_k \n"
            << "Distance is " << std::to_string(cos_results[0].distance)
            << "\n Index is " << std::to_string(cos_results[0].index)
            << std::endl;
  auto cos_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cos_elapsed_duration =
      cos_end_time - cos_start_time;
  std::chrono::milliseconds cos_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          cos_elapsed_duration);
  std::chrono::nanoseconds cos_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          cos_elapsed_duration);
  std::cout << "\nMeasuring Execution Latency\n";
  std::cout << "Took " << cos_elapsed_duration << " Seconds to execute\n";
  std::cout << "Took " << cos_ms << " Milliseconds to execute\n";
  std::cout << "Took " << cos_ns << " Nanoseconds to execute\n";

  std::cout << "Completed Test Suite 2.0 | Cosine Distance\n\n";

  std::cout << "Test Suite 3.0 | Linear Scan w/ L2 Distance \n";
  std::cout << "=====================================\n";

  auto l2_start_time = std::chrono::high_resolution_clock::now();
  std::vector<SearchResult> l2_results = linear_scan(data, query, 15, "l2");
  auto l2_end_time = std::chrono::high_resolution_clock::now();
  std::cout << "First element found in top_k \n"
            << "Distance is " << std::to_string(l2_results[0].distance)
            << "\n Index is " << std::to_string(l2_results[0].index)
            << std::endl;
  std::chrono::duration<double> l2_elapsed_duration =
      l2_end_time - l2_start_time;
  std::chrono::milliseconds l2_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          l2_elapsed_duration);
  std::chrono::nanoseconds l2_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(l2_elapsed_duration);

  std::cout << "\nMeasuring Execution Latency\n";
  std::cout << "Took " << l2_elapsed_duration << " Seconds to execute\n";
  std::cout << "Took " << l2_ms << " Milliseconds to execute\n";
  std::cout << "Took " << l2_ns << " Nanoseconds to execute\n";

  std::cout << "Completed Test Suite 3.0 | L2 Distance\n\n";

  std::cout << "Test Suite 4.0 | Display Shapes \n";
  std::cout << "=====================================\n";

  std::cout << std::to_string(data.size()) << std::endl;
  std::cout << std::to_string(shape[0]) << " x " << std::to_string(shape[1])
            << std::endl;

  return 0;
}
