#include "secan/quantization/binary_quantizer.h"
#include "secan/quantization/distance_sq8.h"
#include "secan/quantization/product_quantizer.h"
#include "secan/quantization/scalar_quantizer.h"
#include "secan/search/distance_avx2.h"
#include "secan/utils/io.h"
#include "secan/utils/normalize.h"
#include <benchmark/benchmark.h>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct QuantizationBenchmarkEnvironment {
  secan::FloatDataset base;
  std::vector<uint8_t> base_sq8;
  std::vector<uint8_t> base_sq4;
  std::vector<uint8_t> base_pq;
  std::vector<uint64_t> base_bq;
  secan::ScalarQuantizer8 quantizer;
  secan::ScalarQuantizer4 quantizer4;
  secan::ProductQuantizer pq;
  secan::BinaryQuantizer bq;
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

    // Train SQ4 on base (packs into 64 bytes per 128-D vector)
    quantizer4 = secan::ScalarQuantizer4(base.dim);
    quantizer4.train(base.num_vectors, base.data.data());

    base_sq4.resize(base.num_vectors * quantizer4.bytes_per_vector());
    quantizer4.encode(base.data.data(), base_sq4.data(), base.num_vectors);

    // Train PQ (M=8 subspaces, d_sub=16) on base sample (10k vectors)
    pq = secan::ProductQuantizer(base.dim, 8);
    pq.train(base.data.data(), std::min(base.num_vectors, size_t(10000)), 15);

    base_pq.resize(base.num_vectors * 8);
    pq.encode(base.data.data(), base_pq.data(), base.num_vectors);

    // Initialize BinaryQuantizer (1-bit BQ)
    bq = secan::BinaryQuantizer(base.dim);
    base_bq.resize(base.num_vectors * bq.num_u64());
    bq.encode(base.data.data(), base_bq.data(), base.num_vectors);

    is_loaded = true;
  }
};

QuantizationBenchmarkEnvironment g_quant_env;

} // anonymous namespace

#include "secan/search/reranker.h"

// ----------------------------------------------------------------------------
// 1. ScalarQuantizer8 & ScalarQuantizer4 Training Benchmark
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

static void BM_SQ4_Train(benchmark::State &state) {
  g_quant_env.load();
  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;

  for (auto _ : state) {
    secan::ScalarQuantizer4 sq(dim);
    sq.train(num_vectors, g_quant_env.base.data.data());
    benchmark::DoNotOptimize(sq);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetBytesProcessed(state.iterations() * num_vectors * dim * sizeof(float));
}

// ----------------------------------------------------------------------------
// 2. ScalarQuantizer8 & ScalarQuantizer4 Encode Benchmark
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

static void BM_SQ4_Encode(benchmark::State &state) {
  g_quant_env.load();
  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;

  secan::ScalarQuantizer4 sq(dim);
  sq.train(num_vectors, g_quant_env.base.data.data());

  std::vector<uint8_t> encoded(num_vectors * sq.bytes_per_vector());

  for (auto _ : state) {
    sq.encode(g_quant_env.base.data.data(), encoded.data(), num_vectors);
    benchmark::DoNotOptimize(encoded.data());
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetBytesProcessed(state.iterations() * num_vectors * dim * sizeof(float));
}

// ----------------------------------------------------------------------------
// 3. ScalarQuantizer8 & ScalarQuantizer4 Decode Benchmark
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

  double mse = secan::ScalarQuantizer8::compute_mse(g_quant_env.base.data.data(), decoded.data(), num_vectors * dim);
  state.counters["Recon_MSE"] = benchmark::Counter(mse, benchmark::Counter::kAvgThreads);
}

static void BM_SQ4_Decode(benchmark::State &state) {
  g_quant_env.load();
  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;

  secan::ScalarQuantizer4 sq(dim);
  sq.train(num_vectors, g_quant_env.base.data.data());

  std::vector<uint8_t> encoded(num_vectors * sq.bytes_per_vector());
  sq.encode(g_quant_env.base.data.data(), encoded.data(), num_vectors);

  std::vector<float> decoded(num_vectors * dim);

  for (auto _ : state) {
    sq.decode(encoded.data(), decoded.data(), num_vectors);
    benchmark::DoNotOptimize(decoded.data());
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetBytesProcessed(state.iterations() * num_vectors * dim * sizeof(float));

  double mse = secan::ScalarQuantizer4::compute_mse(g_quant_env.base.data.data(), decoded.data(), num_vectors * dim);
  state.counters["Recon_MSE"] = benchmark::Counter(mse, benchmark::Counter::kAvgThreads);
}

// ----------------------------------------------------------------------------
// 4. Distance Kernel Linear Scan Benchmark: FP32 AVX2 vs SQ8 AVX2 vs TwoStage SQ4
// ----------------------------------------------------------------------------
static void BM_Scan_FP32_AVX2(benchmark::State &state) {
  g_quant_env.load();
  secan::enable_ftz_daz();

  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;
  const float *data = g_quant_env.base.data.data();
  const float *query = data;

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

static void BM_TwoStage_SQ4_Rerank(benchmark::State &state) {
  g_quant_env.load();
  secan::enable_ftz_daz();

  const size_t num_vectors = state.range(0);
  const size_t dim = g_quant_env.base.dim;
  const float *query = g_quant_env.base.data.data();

  secan::TwoStageReranker reranker(
      g_quant_env.quantizer4,
      g_quant_env.base_sq4.data(),
      g_quant_env.base.data.data(),
      num_vectors,
      dim
  );

  const size_t k = 10;
  const size_t coarse_m = 100;

  for (auto _ : state) {
    auto results = reranker.search(query, k, coarse_m);
    benchmark::DoNotOptimize(results.data());
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
}

static void BM_Subspace_KMeans(benchmark::State &state) {
  g_quant_env.load();
  secan::enable_ftz_daz();

  const size_t num_subvectors = state.range(0);
  const size_t d_sub = 16;
  const size_t k_centroids = 256;
  const size_t iters = 15;

  // Extract contiguous subvectors
  std::vector<float> sub_data(num_subvectors * d_sub);
  for (size_t i = 0; i < num_subvectors; ++i) {
    std::memcpy(sub_data.data() + i * d_sub,
                g_quant_env.base.data.data() + i * g_quant_env.base.dim,
                d_sub * sizeof(float));
  }

  for (auto _ : state) {
    auto centroids = secan::train_subspace_kmeans(
        sub_data.data(), num_subvectors, d_sub, k_centroids, iters);
    benchmark::DoNotOptimize(centroids.data());
  }

  state.SetItemsProcessed(state.iterations() * num_subvectors);
}

static void BM_Scan_PQ_ADC(benchmark::State &state) {
  g_quant_env.load();
  secan::enable_ftz_daz();

  const size_t num_vectors = state.range(0);
  const float *query = g_quant_env.base.data.data();
  const uint8_t *codes = g_quant_env.base_pq.data();
  const size_t M = 8;

  for (auto _ : state) {
    float min_dist = 1e30f;
    size_t min_idx = 0;
    for (size_t i = 0; i < num_vectors; ++i) {
      float d = g_quant_env.pq.compute_adc_distance(query, codes + i * M);
      if (d < min_dist) {
        min_dist = d;
        min_idx = i;
      }
    }
    benchmark::DoNotOptimize(min_dist);
    benchmark::DoNotOptimize(min_idx);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
}

static void BM_Scan_Asymmetric_BQ(benchmark::State &state) {
  g_quant_env.load();
  secan::enable_ftz_daz();

  const size_t num_vectors = state.range(0);
  const float *query = g_quant_env.base.data.data();
  const uint64_t *codes = g_quant_env.base_bq.data();
  const size_t num_u64 = g_quant_env.bq.num_u64();

  float query_sum = 0.0f;
  for (size_t d = 0; d < g_quant_env.base.dim; ++d) {
    query_sum += query[d];
  }

  for (auto _ : state) {
    float max_ip = -1e30f;
    size_t max_idx = 0;
    for (size_t i = 0; i < num_vectors; ++i) {
      float ip = secan::asymmetric_ip_bq(query, codes + i * num_u64, num_u64, query_sum);
      if (ip > max_ip) {
        max_ip = ip;
        max_idx = i;
      }
    }
    benchmark::DoNotOptimize(max_ip);
    benchmark::DoNotOptimize(max_idx);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
}

// Register benchmarks across vector batch sizes
BENCHMARK(BM_SQ8_Train)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SQ4_Train)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SQ8_Encode)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SQ4_Encode)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SQ8_Decode)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SQ4_Decode)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Scan_FP32_AVX2)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Scan_SQ8_AVX2)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Scan_PQ_ADC)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Scan_Asymmetric_BQ)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TwoStage_SQ4_Rerank)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Subspace_KMeans)->Arg(1000)->Arg(10000)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

