#include "secan/index/flat.h"
#include "test_utils.h"
#include <cmath>
#include <vector>

void test_flat2d_initialization() {
  std::cout << "\n--- Testing Flat2DIndex Initialization ---" << std::endl;

  secan::Flat2DIndex index(128, secan::MetricType::L2);
  CHECK(index.dim() == 128);
  CHECK(index.size() == 0);
  CHECK(index.empty());
  CHECK(index.metric() == secan::MetricType::L2);
}

void test_flat2d_add_and_search_exact() {
  std::cout << "\n--- Testing Flat2DIndex Add & Exact Search ---" << std::endl;

  const size_t num_vectors = 500;
  const size_t dim = 64;
  const size_t k = 5;

  secan::Flat2DIndex index(dim, secan::MetricType::L2);
  index.reserve(num_vectors);

  std::vector<float> data(num_vectors * dim);
  for (size_t i = 0; i < num_vectors; ++i) {
    for (size_t d = 0; d < dim; ++d) {
      data[i * dim + d] = static_cast<float>((i + 1) * (d + 1)) * 0.001f;
    }
  }

  index.add(num_vectors, data.data(), nullptr);
  CHECK(index.size() == num_vectors);
  CHECK(!index.empty());

  // Search exact target vectors at indices 0, 42, 128, 499
  std::vector<int32_t> targets = {0, 42, 128, 499};
  for (int32_t target_idx : targets) {
    const float *query = index.get(target_idx);
    auto results = index.search(query, k);

    CHECK(results.size() == k);
    // Nearest neighbor must be the exact vector with distance 0.0f
    CHECK(results[0].index == target_idx);
    CHECK(std::abs(results[0].distance) < 1e-5f);

    // Verify ordering is strictly ascending
    for (size_t i = 1; i < k; ++i) {
      CHECK(results[i].distance >= results[i - 1].distance);
    }
  }
}

void test_flat2d_custom_ids() {
  std::cout << "\n--- Testing Flat2DIndex Custom IDs ---" << std::endl;

  secan::Flat2DIndex index(4, secan::MetricType::L2);

  std::vector<float> data = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f
  };
  std::vector<int32_t> custom_ids = {1001, 1002, 1003, 1004};

  index.add(4, data.data(), custom_ids.data());
  CHECK(index.size() == 4);

  // Search for vector corresponding to custom ID 1003
  std::vector<float> query = {0.0f, 0.0f, 1.0f, 0.0f};
  auto results = index.search(query, 2);

  CHECK(results.size() == 2);
  CHECK(results[0].index == 1003);
  CHECK(std::abs(results[0].distance) < 1e-5f);
}

void test_flat2d_batch_search() {
  std::cout << "\n--- Testing Flat2DIndex Batch Search (Tiled GEMM) ---" << std::endl;

  const size_t num_vectors = 300;
  const size_t dim = 32;
  const size_t num_queries = 6;
  const size_t k = 4;

  secan::Flat2DIndex index(dim, secan::MetricType::L2);

  std::vector<float> data(num_vectors * dim);
  for (size_t i = 0; i < num_vectors; ++i) {
    for (size_t d = 0; d < dim; ++d) {
      data[i * dim + d] = static_cast<float>(i * 3 + d);
    }
  }
  index.add(num_vectors, data.data());

  std::vector<int32_t> query_targets = {5, 25, 75, 120, 210, 299};
  std::vector<float> queries(num_queries * dim);
  for (size_t q = 0; q < num_queries; ++q) {
    const int32_t target_idx = query_targets[q];
    std::copy(data.begin() + target_idx * dim, data.begin() + (target_idx + 1) * dim,
              queries.begin() + q * dim);
  }

  auto batch_res = index.batch_search(num_queries, queries.data(), k, /*tile_size=*/64);
  CHECK(batch_res.size() == num_queries);

  for (size_t q = 0; q < num_queries; ++q) {
    CHECK(batch_res[q].size() == k);
    CHECK(batch_res[q][0].index == query_targets[q]);
    CHECK(std::abs(batch_res[q][0].distance) < 1e-5f);
  }
}

void test_flat2d_reset() {
  std::cout << "\n--- Testing Flat2DIndex Reset ---" << std::endl;

  secan::Flat2DIndex index(8, secan::MetricType::L2);
  std::vector<float> data(16, 1.0f);
  index.add(2, data.data());
  CHECK(index.size() == 2);

  index.reset();
  CHECK(index.size() == 0);
  CHECK(index.empty());
}

int main() {
  print_clock_overhead();

  test_flat2d_initialization();
  test_flat2d_add_and_search_exact();
  test_flat2d_custom_ids();
  test_flat2d_batch_search();
  test_flat2d_reset();

  return report_results("flat_index_tests");
}
