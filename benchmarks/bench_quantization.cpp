#include "secan/quantization/distance_sq8.h"
#include "secan/quantization/scalar_quantizer.h"
#include "secan/search/distance_avx2.h"
#include "secan/utils/io.h"
#include "secan/utils/normalize.h"
#include <benchmark/benchmark.h>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct QuantizationBenchmarkEnvironment {
  secan::FloatDataset base;
  std::vector<uint8_t> base_sq8;
  secan::ScalarQuantizer8 quantizer;
  bool is_loaded{false};

  void load() {
    if (is_loaded) return;

    std::string base_path = "data/sift1m/sift_base_100k.fvecs";
    if (!std::filesystem::exists(base_path)) {
      base_path = "../data/sift1m/sift_base_100k.fvecs";
    }

    if (std::filesystem::exists(base_path)) {
      base = secan::load_fvecs(base_path);
    } else {
      // Synthetic 100k vectors, dim 128
      base.dim = 128;
      base.num_vectors = 100000;
      base.data.resize(base.num_vectors * base.dim);
      std::mt19937 rng(42);
      std::normal_distribution<float> norm_dist(0.0f, 1.0f);
      for (float &val : base.data) {
        val = norm_dist(rng);
      }
    }

    // Train SQ8 on base
    quantizer = secan::ScalarQuantizer8(base.dim);
    quantizer.train(base.num_vectors, base.data.data());

    base_sq8.resize(base.num_vectors * base.dim);
    quantizer.encode(base.data.data(), base_sq8.data(), base.num_vectors);

    is_loaded = true;
  }
};

QuantizationBenchmarkEnvironment g_quant_env;

} // anonymous namespace

// ----------------------------------------------------------------------------
// 1. ScalarQuantizer8 Training Benchmark
// ----------------------------------------------------------------------------
static void BM_SQ8_Train(benchmark::State &state) {
  g_quant_env.load();
  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;

  for (auto _ : state) {
    secan::ScalarQuantizer8 sq(dim);
    sq.train(num_vectors, g_quant_env.base.data.data());
    benchmark::DoNotOptimize(sq);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetBytesProcessed(state.iterations() * num_vectors * dim * sizeof(float));
}

// ----------------------------------------------------------------------------
// 2. ScalarQuantizer8 Encode Benchmark
// ----------------------------------------------------------------------------
static void BM_SQ8_Encode(benchmark::State &state) {
  g_quant_env.load();
  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;

  secan::ScalarQuantizer8 sq(dim);
  sq.train(num_vectors, g_quant_env.base.data.data());

  std::vector<uint8_t> encoded(num_vectors * dim);

  for (auto _ : state) {
    sq.encode(g_quant_env.base.data.data(), encoded.data(), num_vectors);
    benchmark::DoNotOptimize(encoded.data());
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetBytesProcessed(state.iterations() * num_vectors * dim * sizeof(float));
}

// ----------------------------------------------------------------------------
// 3. ScalarQuantizer8 Decode Benchmark
// ----------------------------------------------------------------------------
static void BM_SQ8_Decode(benchmark::State &state) {
  g_quant_env.load();
  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;

  secan::ScalarQuantizer8 sq(dim);
  sq.train(num_vectors, g_quant_env.base.data.data());

  std::vector<uint8_t> encoded(num_vectors * dim);
  sq.encode(g_quant_env.base.data.data(), encoded.data(), num_vectors);

  std::vector<float> decoded(num_vectors * dim);

  for (auto _ : state) {
    sq.decode(encoded.data(), decoded.data(), num_vectors);
    benchmark::DoNotOptimize(decoded.data());
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetBytesProcessed(state.iterations() * num_vectors * dim * sizeof(float));

  // Compute reconstruction MSE and report counter
  double mse = secan::ScalarQuantizer8::compute_mse(g_quant_env.base.data.data(), decoded.data(), num_vectors * dim);
  state.counters["Recon_MSE"] = benchmark::Counter(mse, benchmark::Counter::kAvgThreads);
}

// ----------------------------------------------------------------------------
// 4. Distance Kernel Linear Scan Benchmark: FP32 AVX2 vs SQ8 AVX2
// ----------------------------------------------------------------------------
static void BM_Scan_FP32_AVX2(benchmark::State &state) {
  g_quant_env.load();
  secan::enable_ftz_daz();

  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;
  const float *data = g_quant_env.base.data.data();
  const float *query = data; // query with 0th vector

  for (auto _ : state) {
    float min_dist = 1e30f;
    size_t min_idx = 0;
    for (size_t i = 0; i < num_vectors; ++i) {
      float d = secan::l2_squared_avx2_unroll4(query, data + i * dim, dim);
      if (d < min_dist) {
        min_dist = d;
        min_idx = i;
      }
    }
    benchmark::DoNotOptimize(min_dist);
    benchmark::DoNotOptimize(min_idx);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetBytesProcessed(state.iterations() * num_vectors * dim * sizeof(float));
}

static void BM_Scan_SQ8_AVX2(benchmark::State &state) {
  g_quant_env.load();

  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;
  const uint8_t *data = g_quant_env.base_sq8.data();
  const uint8_t *query = data;

  for (auto _ : state) {
    uint32_t min_dist = 0xFFFFFFFF;
    size_t min_idx = 0;
    for (size_t i = 0; i < num_vectors; ++i) {
      uint32_t d = secan::l2_squared_sq8_avx2(query, data + i * dim, dim);
      if (d < min_dist) {
        min_dist = d;
        min_idx = i;
      }
    }
    benchmark::DoNotOptimize(min_dist);
    benchmark::DoNotOptimize(min_idx);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetBytesProcessed(state.iterations() * num_vectors * dim * sizeof(uint8_t));
}

// Register benchmarks across vector batch sizes
BENCHMARK(BM_SQ8_Train)->Arg(1000)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SQ8_Encode)->Arg(1000)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SQ8_Decode)->Arg(1000)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Scan_FP32_AVX2)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Scan_SQ8_AVX2)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
