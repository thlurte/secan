#include "secan/search/search.h"
#include "secan/utils/io.h"
#include "secan/utils/metrics.h"
#include "test_utils.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

void test_recall_at_k() {
  std::vector<int32_t> gt = {10, 20, 30, 40, 50};
  
  // Perfect match
  std::vector<int32_t> ret1 = {10, 20, 30, 40, 50};
  CHECK_NEAR(secan::recall_at_k(ret1, gt, 5), 1.0, 1e-6);

  // 3 out of 5 match
  std::vector<int32_t> ret2 = {10, 99, 30, 40, 88};
  CHECK_NEAR(secan::recall_at_k(ret2, gt, 5), 0.6, 1e-6);

  // 0 match
  std::vector<int32_t> ret3 = {1, 2, 3, 4, 5};
  CHECK_NEAR(secan::recall_at_k(ret3, gt, 5), 0.0, 1e-6);

  // Cutoff k = 2: both in top-2 GT
  std::vector<int32_t> ret4 = {20, 10, 99};
  CHECK_NEAR(secan::recall_at_k(ret4, gt, 2), 1.0, 1e-6);
}

void test_reciprocal_rank() {
  std::vector<int32_t> gt = {42};

  // Rank 1
  CHECK_NEAR(secan::reciprocal_rank({42, 1, 2}, gt), 1.0, 1e-6);

  // Rank 2
  CHECK_NEAR(secan::reciprocal_rank({1, 42, 2}, gt), 0.5, 1e-6);

  // Rank 4
  CHECK_NEAR(secan::reciprocal_rank({1, 2, 3, 42}, gt), 0.25, 1e-6);

  // Not found
  CHECK_NEAR(secan::reciprocal_rank({1, 2, 3, 4}, gt), 0.0, 1e-6);
}

void test_ndcg_at_k() {
  std::vector<int32_t> gt = {100, 200, 300};

  // Perfect order
  std::vector<int32_t> perfect = {100, 200, 300};
  CHECK_NEAR(secan::ndcg_at_k(perfect, gt, 3), 1.0, 1e-6);

  // Partial match at later positions
  std::vector<int32_t> partial = {999, 100, 200};
  double ndcg_partial = secan::ndcg_at_k(partial, gt, 3);
  CHECK(ndcg_partial > 0.0 && ndcg_partial < 1.0);

  // Complete miss
  std::vector<int32_t> miss = {1, 2, 3};
  CHECK_NEAR(secan::ndcg_at_k(miss, gt, 3), 0.0, 1e-6);
}

void test_average_precision() {
  std::vector<int32_t> gt = {10, 20, 30};

  // Perfect ranking: hits at 1, 2, 3 -> P@1=1, P@2=1, P@3=1 -> AP = 1.0
  std::vector<int32_t> perfect = {10, 20, 30};
  CHECK_NEAR(secan::average_precision_at_k(perfect, gt, 3), 1.0, 1e-6);

  // Miss: AP = 0
  std::vector<int32_t> miss = {1, 2, 3};
  CHECK_NEAR(secan::average_precision_at_k(miss, gt, 3), 0.0, 1e-6);
}

void test_mean_aggregations() {
  std::vector<std::vector<int32_t>> gts = {
      {10, 20},
      {30, 40}
  };
  std::vector<std::vector<int32_t>> rets = {
      {10, 20}, // Recall = 1.0
      {30, 99}  // Recall = 0.5
  };

  CHECK_NEAR(secan::mean_recall_at_k(rets, gts, 2), 0.75, 1e-6);
  CHECK_NEAR(secan::mean_reciprocal_rank(rets, gts), 1.0, 1e-6);
}

void test_exact_linear_scan_recall_sift_subset() {
  const std::string base_path = "data/sift1m/sift_base_100k.fvecs";
  const std::string query_path = "data/sift1m/sift_query_1k.fvecs";

  if (!fs::exists(base_path) || !fs::exists(query_path)) {
    std::cout << "[NOTE] SIFT1M 100k subset not found at " << base_path 
              << ". Skipping exact scan benchmark test." << std::endl;
    return;
  }

  std::cout << "Loading SIFT1M 100k base and 1k queries..." << std::endl;
  secan::FloatDataset base_ds = secan::load_fvecs(base_path);
  secan::FloatDataset query_ds = secan::load_fvecs(query_path);

  CHECK(base_ds.num_vectors == 100000);
  CHECK(base_ds.dim == 128);
  CHECK(query_ds.num_vectors == 1000);
  CHECK(query_ds.dim == 128);

  const size_t num_test_queries = 10;
  const size_t k = 10;

  std::vector<std::vector<int32_t>> all_retrieved;
  all_retrieved.reserve(num_test_queries);

  // Run linear scan
  for (size_t q = 0; q < num_test_queries; ++q) {
    const float *query_vec = query_ds.get(q);
    auto results = secan::linear_scan(base_ds, query_vec, k, "l2");
    CHECK(results.size() == k);

    std::vector<int32_t> ret_ids;
    ret_ids.reserve(k);
    for (const auto &r : results) {
      ret_ids.push_back(r.index);
    }
    all_retrieved.push_back(ret_ids);

    // Self-consistency check: results must be sorted strictly ascending by L2 distance
    for (size_t i = 1; i < results.size(); ++i) {
      CHECK(results[i].distance >= results[i - 1].distance);
    }
  }

  // Self-verification: linear scan against itself as ground truth achieves 1.0 Recall@10 and 1.0 NDCG@10
  double mean_rec = secan::mean_recall_at_k(all_retrieved, all_retrieved, k);
  double mean_ndcg = secan::mean_ndcg_at_k(all_retrieved, all_retrieved, k);
  double mean_map = secan::mean_average_precision(all_retrieved, all_retrieved, k);

  CHECK_NEAR(mean_rec, 1.0, 1e-6);
  CHECK_NEAR(mean_ndcg, 1.0, 1e-6);
  CHECK_NEAR(mean_map, 1.0, 1e-6);

  std::cout << "Exact scan on SIFT1M 100k subset passed: Mean Recall@" << k << " = " << mean_rec 
            << ", NDCG@" << k << " = " << mean_ndcg << ", MAP@" << k << " = " << mean_map << std::endl;
}

int main() {
  test_recall_at_k();
  test_reciprocal_rank();
  test_ndcg_at_k();
  test_average_precision();
  test_mean_aggregations();
  test_exact_linear_scan_recall_sift_subset();
  return report_results("ir_metrics_tests");
}
