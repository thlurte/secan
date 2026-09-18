#include "secan/search/distance.h"
#include "secan/search/distance_avx2.h"
#include "secan/search/distance_avx512.h"
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

static void BM_L2_Squared_AVX2_Single(benchmark::State &state) {
  size_t dim = state.range(0);
  secan::enable_ftz_daz();

  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float dist = secan::l2_squared_avx2_single(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(dist);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

BENCHMARK(BM_L2_Squared_AVX2_Single)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

static void BM_L2_Squared_AVX2_Unroll4(benchmark::State &state) {
  size_t dim = state.range(0);
  secan::enable_ftz_daz();

  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float dist = secan::l2_squared_avx2_unroll4(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(dist);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

BENCHMARK(BM_L2_Squared_AVX2_Unroll4)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

static void BM_L2_Squared_AVX512(benchmark::State &state) {
  size_t dim = state.range(0);
  secan::enable_ftz_daz();

  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float dist = secan::l2_squared_avx512(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(dist);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

BENCHMARK(BM_L2_Squared_AVX512)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

static void BM_Cosine_AVX2(benchmark::State &state) {
  size_t dim = state.range(0);
  secan::enable_ftz_daz();

  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float dist = secan::cosine_distance_avx2(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(dist);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

BENCHMARK(BM_Cosine_AVX2)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

static void BM_IP_AVX2_Unroll4(benchmark::State &state) {
  size_t dim = state.range(0);
  secan::enable_ftz_daz();

  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float ip = secan::ip_avx2_unroll4(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(ip);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

BENCHMARK(BM_IP_AVX2_Unroll4)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

static void BM_IP_AVX512(benchmark::State &state) {
  size_t dim = state.range(0);
  secan::enable_ftz_daz();

  auto [a, shape_a] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});
  auto [b, shape_b] = generate(-1.0f, 1.0f, {static_cast<int>(dim)});

  for (auto _ : state) {
    float ip = secan::ip_avx512(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(ip);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

BENCHMARK(BM_IP_AVX512)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1024)
    ->Arg(1536);

// Cache-Line Split Microbenchmark:
// Measures throughput difference when memory is 64-byte cache-line aligned
// vs when vectors cross 64-byte cache line boundaries (4-byte / 1-float misaligned).
static void BM_CacheLine_Split_Aligned(benchmark::State &state) {
  size_t dim = state.range(0);
  secan::enable_ftz_daz();

  // Allocate 64-byte aligned buffer with padding
  size_t alloc_size = (dim + 16) * sizeof(float);
  void *mem_a = nullptr;
  void *mem_b = nullptr;
  if (posix_memalign(&mem_a, 64, alloc_size) != 0 ||
      posix_memalign(&mem_b, 64, alloc_size) != 0) {
    state.SkipWithError("posix_memalign failed");
    return;
  }

  float *a = static_cast<float *>(mem_a);
  float *b = static_cast<float *>(mem_b);
  for (size_t i = 0; i < dim; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  for (auto _ : state) {
    float ip = secan::ip_avx2_unroll4(a, b, dim);
    benchmark::DoNotOptimize(ip);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
  free(mem_a);
  free(mem_b);
}

static void BM_CacheLine_Split_Unaligned(benchmark::State &state) {
  size_t dim = state.range(0);
  secan::enable_ftz_daz();

  size_t alloc_size = (dim + 16) * sizeof(float);
  void *mem_a = nullptr;
  void *mem_b = nullptr;
  if (posix_memalign(&mem_a, 64, alloc_size) != 0 ||
      posix_memalign(&mem_b, 64, alloc_size) != 0) {
    state.SkipWithError("posix_memalign failed");
    return;
  }

  // Shift pointer by 1 float (4 bytes).
  // 4 bytes offset guarantees that 32-byte AVX loads will straddle 64-byte cache-line boundaries!
  float *a = static_cast<float *>(mem_a) + 1;
  float *b = static_cast<float *>(mem_b) + 1;
  for (size_t i = 0; i < dim; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  for (auto _ : state) {
    float ip = secan::ip_avx2_unroll4(a, b, dim);
    benchmark::DoNotOptimize(ip);
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
  free(mem_a);
  free(mem_b);
}

BENCHMARK(BM_CacheLine_Split_Aligned)->Arg(768)->Arg(1536);
BENCHMARK(BM_CacheLine_Split_Unaligned)->Arg(768)->Arg(1536);

BENCHMARK_MAIN();



