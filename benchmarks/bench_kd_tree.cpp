#include "secan/tree/kd_tree.h"
#include "secan/search/search.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static void BM_KdTree_Search_DimensionSweep(benchmark::State &state) {
  const size_t dim = state.range(0);
  const size_t num_vectors = 50000;
  const size_t k = 10;

  secan::FloatDataset ds;
  ds.num_vectors = num_vectors;
  ds.dim = dim;
  ds.data.resize(num_vectors * dim);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < num_vectors * dim; ++i) {
    ds.data[i] = dist(rng);
  }

  secan::KdTreeParams params;
  params.num_trees = 4;
  params.leaf_max_size = 32;
  params.top_variance_dims = 5;

  secan::RandomizedKdTree index(params);
  index.build(ds);

  std::vector<float> query(dim);
  for (size_t d = 0; d < dim; ++d) {
    query[d] = dist(rng);
  }

  secan::KdSearchParams search_params;
  search_params.max_checks = 512;
  search_params.metric = secan::MetricType::L2;

  for (auto _ : state) {
    auto res = index.search(query.data(), k, search_params);
    benchmark::DoNotOptimize(res);
  }
}

static void BM_BruteForce_DimensionSweep(benchmark::State &state) {
  const size_t dim = state.range(0);
  const size_t num_vectors = 50000;
  const size_t k = 10;

  secan::FloatDataset ds;
  ds.num_vectors = num_vectors;
  ds.dim = dim;
  ds.data.resize(num_vectors * dim);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < num_vectors * dim; ++i) {
    ds.data[i] = dist(rng);
  }

  std::vector<float> query(dim);
  for (size_t d = 0; d < dim; ++d) {
    query[d] = dist(rng);
  }

  for (auto _ : state) {
    auto res = secan::linear_scan(ds, query.data(), k, secan::MetricType::L2);
    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK(BM_KdTree_Search_DimensionSweep)->Arg(2)->Arg(8)->Arg(16)->Arg(64)->Arg(128);
BENCHMARK(BM_BruteForce_DimensionSweep)->Arg(2)->Arg(8)->Arg(16)->Arg(64)->Arg(128);

BENCHMARK_MAIN();
