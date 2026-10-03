#include "secan/index/flat.h"
#include "secan/index/ivf_flat.h"
#include "secan/search/distance_avx2.h"
#include "secan/tree/kd_tree.h"
#include "secan/utils/io.h"
#include "secan/utils/metrics.h"
#include <benchmark/benchmark.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace {

struct SiftBenchmarkEnvironment {
  secan::FloatDataset base;
  secan::FloatDataset queries;
  secan::IntDataset ground_truth;
  std::unique_ptr<secan::Flat2DIndex> flat_index;
  std::unique_ptr<secan::RandomizedKdTree> kd_tree;
  std::unique_ptr<secan::IvfFlatIndex> ivf_index;
  bool is_loaded{false};

  void load() {
    if (is_loaded) return;

    std::string base_path = "data/sift1m/sift_base_100k.fvecs";
    std::string query_path = "data/sift1m/sift_query_1k.fvecs";
    std::string gt_path = "data/sift1m/sift_groundtruth.ivecs";

    // Fallback if running from a different subfolder
    if (!std::filesystem::exists(base_path)) {
      base_path = "../data/sift1m/sift_base_100k.fvecs";
      query_path = "../data/sift1m/sift_query_1k.fvecs";
      gt_path = "../data/sift1m/sift_groundtruth.ivecs";
    }

    if (!std::filesystem::exists(base_path)) {
      std::cerr << "Warning: SIFT dataset not found at " << base_path << std::endl;
      return;
    }

    base = secan::load_fvecs(base_path);
    queries = secan::load_fvecs(query_path);
    ground_truth = secan::load_ivecs(gt_path);

    // Build Flat2DIndex
    flat_index = std::make_unique<secan::Flat2DIndex>(base.dim, secan::MetricType::L2);
    flat_index->add(base);

    // Build RandomizedKdTree
    secan::KdTreeParams kd_params;
    kd_params.num_trees = 4;
    kd_params.leaf_max_size = 32;
    kd_params.top_variance_dims = 5;
    kd_params.seed = 42;
    kd_tree = std::make_unique<secan::RandomizedKdTree>(kd_params);
    kd_tree->build(base);

    // Build IvfFlatIndex (nlist = 256 for 100K vectors)
    const size_t nlist = 256;
    ivf_index = std::make_unique<secan::IvfFlatIndex>(base.dim, nlist, secan::MetricType::L2);
    ivf_index->train(base.num_vectors, base.data.data(), 15);

    std::vector<int32_t> ids(base.num_vectors);
    for (size_t i = 0; i < base.num_vectors; ++i) {
      ids[i] = static_cast<int32_t>(i);
    }
    ivf_index->add(base.num_vectors, ids.data(), base.data.data());

    is_loaded = true;
  }
};

SiftBenchmarkEnvironment g_sift_env;

} // anonymous namespace

// ----------------------------------------------------------------------------
// 1. Flat2DIndex Exact Search Benchmark (Single Query & Batch Tiled)
// ----------------------------------------------------------------------------

static void BM_SIFT100K_Flat2D_SingleQuery(benchmark::State &state) {
  g_sift_env.load();
  if (!g_sift_env.is_loaded) {
    state.SkipWithError("SIFT dataset missing");
    return;
  }
  secan::enable_ftz_daz();

  const size_t k = 10;
  const size_t num_queries = std::min<size_t>(100, g_sift_env.queries.num_vectors);

  size_t q_idx = 0;
  for (auto _ : state) {
    const float *query = g_sift_env.queries.get(q_idx % num_queries);
    auto results = g_sift_env.flat_index->search(query, k);
    benchmark::DoNotOptimize(results);
    q_idx++;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_SIFT100K_Flat2D_BatchTiled(benchmark::State &state) {
  g_sift_env.load();
  if (!g_sift_env.is_loaded) {
    state.SkipWithError("SIFT dataset missing");
    return;
  }
  secan::enable_ftz_daz();

  const size_t k = 10;
  const size_t num_queries = 64;
  const size_t tile_size = state.range(0);
  const float *query_batch = g_sift_env.queries.get(0);

  for (auto _ : state) {
    auto results = g_sift_env.flat_index->batch_search(num_queries, query_batch, k, tile_size);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * num_queries);
}

// ----------------------------------------------------------------------------
// 2. RandomizedKdTree Search Benchmark with Recall@10 Measurement
// ----------------------------------------------------------------------------

static void BM_SIFT100K_KdTree_Search(benchmark::State &state) {
  g_sift_env.load();
  if (!g_sift_env.is_loaded) {
    state.SkipWithError("SIFT dataset missing");
    return;
  }
  secan::enable_ftz_daz();

  const size_t k = 10;
  const size_t max_checks = state.range(0);
  const size_t num_queries = std::min<size_t>(100, g_sift_env.queries.num_vectors);

  secan::KdSearchParams search_params;
  search_params.max_checks = max_checks;
  search_params.metric = secan::MetricType::L2;

  size_t q_idx = 0;
  for (auto _ : state) {
    const float *query = g_sift_env.queries.get(q_idx % num_queries);
    auto results = g_sift_env.kd_tree->search(query, k, search_params);
    benchmark::DoNotOptimize(results);
    q_idx++;
  }

  state.SetItemsProcessed(state.iterations());

  // Compute and record Recall@10 as a benchmark user counter
  std::vector<std::vector<int32_t>> retrieved(num_queries);
  std::vector<std::vector<int32_t>> gt(num_queries);

  for (size_t q = 0; q < num_queries; ++q) {
    const float *query = g_sift_env.queries.get(q);
    auto res = g_sift_env.kd_tree->search(query, k, search_params);
    retrieved[q].reserve(res.size());
    for (const auto &r : res) {
      retrieved[q].push_back(r.id);
    }

    const int32_t *gt_ptr = g_sift_env.ground_truth.get(q);
    gt[q].assign(gt_ptr, gt_ptr + k);
  }

  double mean_recall = secan::mean_recall_at_k(retrieved, gt, k);
  state.counters["Recall@10"] = benchmark::Counter(mean_recall, benchmark::Counter::kAvgThreads);
}

// ----------------------------------------------------------------------------
// 3. IvfFlatIndex Multi-Probe Search Benchmark with Recall@10 Sweep
// ----------------------------------------------------------------------------

static void BM_SIFT100K_IvfFlat_Search(benchmark::State &state) {
  g_sift_env.load();
  if (!g_sift_env.is_loaded) {
    state.SkipWithError("SIFT dataset missing");
    return;
  }
  secan::enable_ftz_daz();

  const size_t k = 10;
  const size_t nprobe = state.range(0);
  const size_t num_queries = std::min<size_t>(100, g_sift_env.queries.num_vectors);

  size_t q_idx = 0;
  for (auto _ : state) {
    const float *query = g_sift_env.queries.get(q_idx % num_queries);
    auto results = g_sift_env.ivf_index->search(query, k, nprobe);
    benchmark::DoNotOptimize(results);
    q_idx++;
  }

  state.SetItemsProcessed(state.iterations());

  // Compute and record Recall@10 as a benchmark user counter
  std::vector<std::vector<int32_t>> retrieved(num_queries);
  std::vector<std::vector<int32_t>> gt(num_queries);

  for (size_t q = 0; q < num_queries; ++q) {
    const float *query = g_sift_env.queries.get(q);
    auto res = g_sift_env.ivf_index->search(query, k, nprobe);
    retrieved[q].reserve(res.size());
    for (const auto &r : res) {
      retrieved[q].push_back(r.id);
    }

    const int32_t *gt_ptr = g_sift_env.ground_truth.get(q);
    gt[q].assign(gt_ptr, gt_ptr + k);
  }

  double mean_recall = secan::mean_recall_at_k(retrieved, gt, k);
  state.counters["Recall@10"] = benchmark::Counter(mean_recall, benchmark::Counter::kAvgThreads);
}

// ----------------------------------------------------------------------------
// Benchmark Registrations
// ----------------------------------------------------------------------------

BENCHMARK(BM_SIFT100K_Flat2D_SingleQuery)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SIFT100K_Flat2D_BatchTiled)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SIFT100K_KdTree_Search)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SIFT100K_IvfFlat_Search)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();

