#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "secan/utils/utils.h"

float l2_squared(const float *row, const float *query, int dim) {
  float diff = 0.0f;
  for (size_t i = 0; i < static_cast<size_t>(dim); i++) {
    diff += (row[i] - query[i]) * (row[i] - query[i]);
  }
  return diff;
}

float cosine_distance(const float *row, const float *query, int dim) {
  float dot_prod = 0.0f;
  float norm_query = 0.0f;
  float norm_row = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    dot_prod = dot_prod + (row[i] * query[i]);
    norm_query += query[i] * query[i];
    norm_row += row[i] * row[i];
  }
  float similarity =
      dot_prod / std::sqrt(norm_query * norm_row); // overflow risk is possible
  return 1.0f - similarity;
}

std::vector<SearchResult> linear_scan(const std::vector<float> &dataset,
                                      const std::vector<float> &query,
                                      int top_k,
                                      std::string method = "cosine") {

  int dim = query.size();
  int N = dataset.size() / dim;
  std::vector<SearchResult> results;
  results.reserve(N);
  for (size_t i = 0; static_cast<size_t>(i) < N; i++) {
    const float *row = &dataset[i * dim];
    if (method == "cosine") {
      float dist = cosine_distance(row, query.data(), dim);
      results.push_back({static_cast<int>(i), dist});

    } else {
      float dist = l2_squared(row, query.data(), dim);
      results.push_back({static_cast<int>(i), dist});
    }
  }
  std::partial_sort(results.begin(), results.begin() + top_k, results.end(),
                    [](const SearchResult &a, const SearchResult &b) {
                      return a.distance < b.distance;
                    });
  results.resize(top_k);
  return results;
}

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
