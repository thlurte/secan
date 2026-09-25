#include "secan/tree/kd_tree.h"
#include "secan/search/search.h"
#include "test_utils.h"
#include <cmath>
#include <vector>

void test_kd_tree_exact_match() {
  const size_t num_vectors = 1000;
  const size_t dim = 16;
  const size_t k = 5;

  secan::FloatDataset ds;
  ds.num_vectors = num_vectors;
  ds.dim = dim;
  ds.data.resize(num_vectors * dim);

  for (size_t i = 0; i < num_vectors; ++i) {
    for (size_t d = 0; d < dim; ++d) {
      ds.data[i * dim + d] = static_cast<float>(i * 7 + d * 13) * 0.01f;
    }
  }

  secan::KdTreeParams params;
  params.num_trees = 4;
  params.leaf_max_size = 16;
  params.top_variance_dims = 5;
  params.seed = 123;

  secan::RandomizedKdTree index(params);
  index.build(ds);

  CHECK(index.size() == num_vectors);
  CHECK(index.dim() == dim);
  CHECK(index.num_trees() == 4);

  // Search exact vectors at index 0, 100, 500, 999
  std::vector<int32_t> targets = {0, 100, 500, 999};
  secan::KdSearchParams search_params;
  search_params.max_checks = 512;
  search_params.metric = secan::MetricType::L2;

  for (int32_t target_idx : targets) {
    const float *query = ds.get(target_idx);
    auto results = index.search(query, k, search_params);

    CHECK(results.size() == k);
    // Nearest neighbor should be the exact target vector with 0 distance
    CHECK(results[0].index == target_idx);
    CHECK(std::abs(results[0].distance) < 1e-5f);
  }
}

void test_kd_tree_recall_vs_linear_scan() {
  const size_t num_vectors = 500;
  const size_t dim = 8;
  const size_t k = 10;

  secan::FloatDataset ds;
  ds.num_vectors = num_vectors;
  ds.dim = dim;
  ds.data.resize(num_vectors * dim);

  for (size_t i = 0; i < num_vectors * dim; ++i) {
    ds.data[i] = static_cast<float>((i * 19) % 251) * 0.05f;
  }

  secan::KdTreeParams params;
  params.num_trees = 4;
  params.leaf_max_size = 8;

  secan::RandomizedKdTree index(params);
  index.build(ds);

  std::vector<float> query(dim);
  for (size_t d = 0; d < dim; ++d) {
    query[d] = static_cast<float>((d * 23) % 100) * 0.05f;
  }

  // Exact linear scan ground truth
  auto ground_truth = secan::linear_scan(ds, query.data(), k, secan::MetricType::L2);

  // KD-Tree search with generous budget should achieve near 100% recall in 8 dimensions
  secan::KdSearchParams search_params;
  search_params.max_checks = num_vectors; // Search whole tree if needed
  search_params.metric = secan::MetricType::L2;

  auto kd_results = index.search(query.data(), k, search_params);

  CHECK(kd_results.size() == k);
  size_t matches = 0;
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = 0; j < k; ++j) {
      if (kd_results[i].index == ground_truth[j].index) {
        ++matches;
        break;
      }
    }
  }

  float recall = static_cast<float>(matches) / static_cast<float>(k);
  CHECK(recall >= 0.90f);
}

int main() {
  test_kd_tree_exact_match();
  test_kd_tree_recall_vs_linear_scan();
  return report_results("kd_tree_tests");
}
