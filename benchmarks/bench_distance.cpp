#include "secan/search/distance.h"
#include "secan/search/distance_avx2.h"
#include "secan/utils/utils.h"
#include <benchmark/benchmark.h>
#include <vector>

static void BM_L2_Squared_Scalar(benchmark::State &state) {
  size_t dim = state.range(0);

  // Generate random test vectors
  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float dist = secan::l2_squared_scalar(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(dist);
    benchmark::ClobberMemory();
  }

  // Reports memory bandwidth throughput in GB/s
  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_L2_Squared_AVX2(benchmark::State &state) {
  size_t dim = state.range(0);

  // Generate random test vectors
  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float dist = secan::l2_squared_avx2_single(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(dist);
    benchmark::ClobberMemory();
  }

  // Reports memory bandwidth throughput in GB/s
  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_IP_Scalar(benchmark::State &state) {
  size_t dim = state.range(0);

  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float dist = secan::inner_product_scalar(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(dist);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_Cosine_Scalar(benchmark::State &state) {
  size_t dim = state.range(0);

  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float dist = secan::cosine_distance_scalar(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(dist);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_Cosine_Fast_Scalar(benchmark::State &state) {
  size_t dim = state.range(0);

  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float dist = secan::cosine_distance_fast_scalar(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(dist);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

// Benchmark across standard AI embedding dimensions
BENCHMARK(BM_L2_Squared_Scalar)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_IP_Scalar)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_Cosine_Scalar)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_Cosine_Fast_Scalar)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK(BM_L2_Squared_AVX2)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

BENCHMARK_MAIN();
