#include "secan/search/distance.h"
#include "secan/search/distance_avx2.h"
#include "secan/search/distance_avx512.h"
#include "secan/search/search.h"
#include "secan/utils/utils.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <limits>
#include <vector>

// ----------------------------------------------------------------------------
// 1. Exact Scan Working Set Sweeps (Scalar, AVX2 Unrolled-4, AVX-512)
// Sweeps from L1/L2 (100 vectors = 51.2 KB) to DRAM (1,000,000 vectors = 512 MB)
// ----------------------------------------------------------------------------

static void BM_ExactScan_Scalar(benchmark::State &state) {
  const size_t n = state.range(0);
  const size_t dim = 128;
  std::vector<float> data(n * dim, 0.5f);
  std::vector<float> query(dim, 1.0f);

  for (auto _ : state) {
    float min_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < n; ++i) {
      float d = secan::l2_squared_scalar(query.data(), data.data() + i * dim, dim);
      if (d < min_dist) min_dist = d;
    }
    benchmark::DoNotOptimize(min_dist);
  }
  state.SetBytesProcessed(state.iterations() * n * dim * sizeof(float));
}

static void BM_ExactScan_AVX2(benchmark::State &state) {
  secan::enable_ftz_daz();
  const size_t n = state.range(0);
  const size_t dim = 128;
  std::vector<float> data(n * dim, 0.5f);
  std::vector<float> query(dim, 1.0f);

  for (auto _ : state) {
    float min_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < n; ++i) {
      float d = secan::l2_squared_avx2_unroll4(query.data(), data.data() + i * dim, dim);
      if (d < min_dist) min_dist = d;
    }
    benchmark::DoNotOptimize(min_dist);
  }
  state.SetBytesProcessed(state.iterations() * n * dim * sizeof(float));
}

static void BM_ExactScan_AVX512(benchmark::State &state) {
  secan::enable_ftz_daz();
  const size_t n = state.range(0);
  const size_t dim = 128;
  std::vector<float> data(n * dim, 0.5f);
  std::vector<float> query(dim, 1.0f);

  for (auto _ : state) {
    float min_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < n; ++i) {
      float d = secan::l2_squared_avx512(query.data(), data.data() + i * dim, dim);
      if (d < min_dist) min_dist = d;
    }
    benchmark::DoNotOptimize(min_dist);
  }
  state.SetBytesProcessed(state.iterations() * n * dim * sizeof(float));
}

// ----------------------------------------------------------------------------
// 2. Memory Mountain Microbenchmark
// Sweeps working set size (KB) and stride (elements) to characterize cache hierarchy:
// L1 (<32KB), L2 (<1MB), L3 (<16MB), and DRAM (>16MB).
// ----------------------------------------------------------------------------

static void BM_MemoryMountain(benchmark::State &state) {
  const size_t size_kb = state.range(0);
  const size_t stride = state.range(1);
  const size_t num_elems = (size_kb * 1024) / sizeof(float);

  std::vector<float> buffer(num_elems, 1.0001f);
  const float *data = buffer.data();

  for (auto _ : state) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    size_t i = 0;
    const size_t step = 4 * stride;
    for (; i + step <= num_elems; i += step) {
      sum0 += data[i];
      sum1 += data[i + stride];
      sum2 += data[i + 2 * stride];
      sum3 += data[i + 3 * stride];
    }
    for (; i < num_elems; i += stride) {
      sum0 += data[i];
    }
    float total = sum0 + sum1 + sum2 + sum3;
    benchmark::DoNotOptimize(total);
    benchmark::ClobberMemory();
  }

  // Bytes actually accessed in each iteration of the state loop
  const size_t accessed_elements = (num_elems + stride - 1) / stride;
  state.SetBytesProcessed(state.iterations() * accessed_elements * sizeof(float));
}

// ----------------------------------------------------------------------------
// 3. Full Linear Scan (Top-K = 10) Working Set Sweep
// ----------------------------------------------------------------------------

static void BM_LinearScan_Top10(benchmark::State &state) {
  const size_t n = state.range(0);
  const size_t dim = 128;
  std::vector<float> data(n * dim, 0.5f);
  std::vector<float> query(dim, 1.0f);

  for (auto _ : state) {
    auto res = secan::linear_scan(data, query, 10, secan::MetricType::L2);
    benchmark::DoNotOptimize(res);
  }
  state.SetBytesProcessed(state.iterations() * n * dim * sizeof(float));
}

// ----------------------------------------------------------------------------
// 4. Batched Tiled Linear Scan (Q=64 queries, Top-K=10) with L2 Cache Tiling
// ----------------------------------------------------------------------------

static void BM_BatchLinearScan_Untiled(benchmark::State &state) {
  secan::enable_ftz_daz();
  const size_t n = state.range(0);
  const size_t dim = 128;
  const size_t num_queries = 64;
  std::vector<float> data(n * dim, 0.5f);
  std::vector<float> queries(num_queries * dim, 1.0f);

  for (auto _ : state) {
    std::vector<std::vector<SearchResult>> all_results(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
      all_results[q] = secan::linear_scan(data, std::vector<float>(queries.begin() + q * dim, queries.begin() + (q + 1) * dim), 10, secan::MetricType::L2);
    }
    benchmark::DoNotOptimize(all_results);
  }
  state.SetBytesProcessed(state.iterations() * num_queries * n * dim * sizeof(float));
}

static void BM_BatchLinearScan_Tiled(benchmark::State &state) {
  secan::enable_ftz_daz();
  const size_t n = state.range(0);
  const size_t dim = 128;
  const size_t num_queries = 64;
  std::vector<float> data(n * dim, 0.5f);
  std::vector<float> queries(num_queries * dim, 1.0f);

  for (auto _ : state) {
    auto res = secan::batch_linear_scan_tiled(queries.data(), num_queries, data.data(), n, dim, 10, /*tile_size=*/2048);
    benchmark::DoNotOptimize(res);
  }
  state.SetBytesProcessed(state.iterations() * num_queries * n * dim * sizeof(float));
}

// Register Exact Scan sweeps across working sets
BENCHMARK(BM_ExactScan_Scalar)->Arg(100)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);
BENCHMARK(BM_ExactScan_AVX2)->Arg(100)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);
BENCHMARK(BM_ExactScan_AVX512)->Arg(100)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);
BENCHMARK(BM_LinearScan_Top10)->Arg(100)->Arg(1000)->Arg(10000)->Arg(100000);
BENCHMARK(BM_BatchLinearScan_Untiled)->Arg(10000)->Arg(100000);
BENCHMARK(BM_BatchLinearScan_Tiled)->Arg(10000)->Arg(100000);

// Register Memory Mountain sweep args: sizes in KB x strides in elements
static void CustomMemoryMountainArgs(benchmark::internal::Benchmark *b) {
  const std::vector<int64_t> sizes_kb = {
      4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536};
  const std::vector<int64_t> strides = {1, 2, 4, 8, 16, 32, 64};

  for (int64_t sz : sizes_kb) {
    for (int64_t st : strides) {
      b->Args({sz, st});
    }
  }
}

BENCHMARK(BM_MemoryMountain)->Apply(CustomMemoryMountainArgs);

BENCHMARK_MAIN();
