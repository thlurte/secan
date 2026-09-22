#include "secan/search/search.h"
#include "test_utils.h"

void test_linear_scan_finds_exact_match() {
  // dataset: 3 vectors of dim 2
  std::vector<float> dataset = {1.0f, 1.0f, 5.0f, 5.0f, 9.0f, 9.0f};
  std::vector<float> query = {5.0f, 5.0f};

  auto results = linear_scan(dataset, query, 1, "l2");
  CHECK(results[0].index == 1); // exact match at index 1
  CHECK(results[0].distance == 0.0f);
}

void test_batch_linear_scan_tiled() {
  // Create 500 vectors with dim=128
  const size_t num_base = 500;
  const size_t dim = 128;
  const size_t num_queries = 8;
  const size_t k = 5;

  std::vector<float> base(num_base * dim, 0.0f);
  for (size_t i = 0; i < num_base; ++i) {
    for (size_t d = 0; d < dim; ++d) {
      base[i * dim + d] = static_cast<float>((i + 1) * (d + 1)) * 0.001f;
    }
  }

  // Queries: match vectors at indices 10, 50, 100, 200, 300, 400, 450, 499 exactly
  std::vector<int32_t> target_indices = {10, 50, 100, 200, 300, 400, 450, 499};
  std::vector<float> queries(num_queries * dim);
  for (size_t q = 0; q < num_queries; ++q) {
    const int32_t idx = target_indices[q];
    std::copy(base.begin() + idx * dim, base.begin() + (idx + 1) * dim,
              queries.begin() + q * dim);
  }

  // Test with small tile size to exercise multiple tile transitions
  auto tiled_results = batch_linear_scan_tiled(
      queries.data(), num_queries, base.data(), num_base, dim, k, /*tile_size=*/64);

  CHECK(tiled_results.size() == num_queries);

  for (size_t q = 0; q < num_queries; ++q) {
    CHECK(tiled_results[q].indices.size() == k);
    CHECK(tiled_results[q].distances.size() == k);
    // Nearest neighbor must be the exact match with distance 0.0f
    CHECK(tiled_results[q].indices[0] == target_indices[q]);
    CHECK(std::abs(tiled_results[q].distances[0]) < 1e-5f);

    // Verify ordering is strictly ascending by distance
    for (size_t i = 1; i < k; ++i) {
      CHECK(tiled_results[q].distances[i] >= tiled_results[q].distances[i - 1]);
    }
  }
}

int main() {
  test_linear_scan_finds_exact_match();
  test_batch_linear_scan_tiled();
  return report_results("linear_scan_tests");
}
