#include "secan/quantization/scalar_quantizer.h"
#include "secan/utils/io.h"
#include "secan/utils/normalize.h"
#include "test_utils.h"
#include <cmath>
#include <filesystem>
#include <random>
#include <vector>

void test_quantizer_untrained_exceptions() {
  secan::ScalarQuantizer8 sq(4);
  CHECK(!sq.is_trained());
  CHECK(sq.dim() == 4);

  float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  uint8_t encoded[4] = {0};
  float decoded[4] = {0};

  bool threw_encode = false;
  try {
    sq.encode(src, encoded, 1);
  } catch (const std::runtime_error &) {
    threw_encode = true;
  }
  CHECK(threw_encode);

  bool threw_decode = false;
  try {
    sq.decode(encoded, decoded, 1);
  } catch (const std::runtime_error &) {
    threw_decode = true;
  }
  CHECK(threw_decode);
}

void test_quantizer_invalid_args() {
  secan::ScalarQuantizer8 sq(4);
  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  // Zero n
  bool threw_zero_n = false;
  try {
    sq.train(0, data);
  } catch (const std::invalid_argument &) {
    threw_zero_n = true;
  }
  CHECK(threw_zero_n);

  // Null data
  bool threw_null = false;
  try {
    sq.train(1, nullptr);
  } catch (const std::invalid_argument &) {
    threw_null = true;
  }
  CHECK(threw_null);

  // Invalid percentiles
  bool threw_bad_pct = false;
  try {
    sq.train(1, data, 0.9f, 0.1f);
  } catch (const std::invalid_argument &) {
    threw_bad_pct = true;
  }
  CHECK(threw_bad_pct);
}

void test_quantizer_constant_data() {
  secan::ScalarQuantizer8 sq(4);
  std::vector<float> const_data(40, 5.0f); // 10 vectors of constant 5.0f

  sq.train(10, const_data.data());
  CHECK(sq.is_trained());
  CHECK_NEAR(sq.min_val(), 5.0f, 1e-5f);
  CHECK(sq.max_val() > sq.min_val()); // Protected from zero-division

  std::vector<uint8_t> encoded(40);
  std::vector<float> decoded(40);
  sq.encode(const_data.data(), encoded.data(), 10);
  sq.decode(encoded.data(), decoded.data(), 10);

  for (size_t i = 0; i < 40; ++i) {
    CHECK_NEAR(decoded[i], 5.0f, 1e-3f);
  }
}

void test_quantizer_percentile_clipping() {
  const size_t dim = 10;
  const size_t num_vectors = 1000;
  std::vector<float> data(num_vectors * dim);

  // Generate standard uniform in [-1.0, 1.0]
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = dist(rng);
  }

  // Inject a few extreme outliers
  data[0] = -1000.0f;
  data[1] = 1000.0f;

  secan::ScalarQuantizer8 sq(dim);
  // Default lower_pct = 0.0005 (0.05th percentile), upper_pct = 0.9995 (99.95th percentile)
  sq.train(num_vectors, data.data());

  // Check that the quantizer min/max are around -1.0 and 1.0, NOT -1000 and 1000
  CHECK(sq.min_val() >= -1.05f && sq.min_val() <= -0.95f);
  CHECK(sq.max_val() <= 1.05f && sq.max_val() >= 0.95f);
}

void test_quantizer_roundtrip_reconstruction() {
  const size_t dim = 128;
  const size_t num_vectors = 500;
  std::vector<float> original(num_vectors * dim);

  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-2.0f, 3.0f);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = dist(rng);
  }

  secan::ScalarQuantizer8 sq(dim);
  sq.train(num_vectors, original.data());

  std::vector<uint8_t> encoded(num_vectors * dim);
  std::vector<float> decoded(num_vectors * dim);

  sq.encode(original.data(), encoded.data(), num_vectors);
  sq.decode(encoded.data(), decoded.data(), num_vectors);

  float max_step = sq.step();
  for (size_t i = 0; i < original.size(); ++i) {
    float diff = std::abs(original[i] - decoded[i]);
    CHECK(diff <= max_step + 1e-4f);
  }

  double mse = secan::ScalarQuantizer8::compute_mse(original.data(), decoded.data(), original.size());
  // Theoretical uniform quantization MSE: step^2 / 12
  double expected_mse = (max_step * max_step) / 12.0;
  CHECK_NEAR(static_cast<float>(mse), static_cast<float>(expected_mse), 0.001f);
}

void test_quantizer_sift_reconstruction() {
  std::string sift_path = "data/sift1m/sift_base_100k.fvecs";
  if (!std::filesystem::exists(sift_path)) {
    sift_path = "../data/sift1m/sift_base_100k.fvecs";
  }

  if (std::filesystem::exists(sift_path)) {
    auto base = secan::load_fvecs(sift_path);
    size_t num_eval = std::min<size_t>(base.num_vectors, 10000);

    // Normalize vectors to unit sphere so coordinates are in [-1, 1]
    std::vector<float> normed_data(base.data.begin(), base.data.begin() + num_eval * base.dim);
    secan::normalize_dataset_avx2(normed_data.data(), num_eval, base.dim);

    secan::ScalarQuantizer8 sq(base.dim);
    sq.train(num_eval, normed_data.data());

    std::vector<uint8_t> encoded(num_eval * base.dim);
    std::vector<float> decoded(num_eval * base.dim);

    sq.encode(normed_data.data(), encoded.data(), num_eval);
    sq.decode(encoded.data(), decoded.data(), num_eval);

    double mse = secan::ScalarQuantizer8::compute_mse(normed_data.data(), decoded.data(), num_eval * base.dim);
    // On unit-normalized embeddings, SQ8 MSE is well below 10^-4
    CHECK(mse < 1e-4);
  } else {
    // Synthetic normalized SIFT-like embeddings (dim = 128)
    const size_t dim = 128;
    const size_t num_eval = 2000;
    std::vector<float> normed_data(num_eval * dim);
    std::mt19937 rng(42);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);

    for (size_t i = 0; i < num_eval; ++i) {
      for (size_t d = 0; d < dim; ++d) {
        normed_data[i * dim + d] = norm_dist(rng);
      }
    }
    secan::normalize_dataset_avx2(normed_data.data(), num_eval, dim);

    secan::ScalarQuantizer8 sq(dim);
    sq.train(num_eval, normed_data.data());

    std::vector<uint8_t> encoded(num_eval * dim);
    std::vector<float> decoded(num_eval * dim);

    sq.encode(normed_data.data(), encoded.data(), num_eval);
    sq.decode(encoded.data(), decoded.data(), num_eval);

    double mse = secan::ScalarQuantizer8::compute_mse(normed_data.data(), decoded.data(), num_eval * dim);
    CHECK(mse < 1e-4);
  }
}

#include "secan/search/reranker.h"
#include "secan/search/distance_avx2.h"

void test_quantizer_sq4_nibble_packing() {
  const size_t dim = 128;
  secan::ScalarQuantizer4 sq4(dim);
  CHECK(sq4.bytes_per_vector() == 64); // Exactly 1 cache line!

  std::vector<float> data(dim);
  for (size_t d = 0; d < dim; ++d) {
    data[d] = static_cast<float>(d) / 128.0f;
  }

  sq4.train(1, data.data());
  CHECK(sq4.is_trained());

  std::vector<uint8_t> packed(sq4.bytes_per_vector());
  std::vector<float> decoded(dim);

  sq4.encode(data.data(), packed.data(), 1);
  sq4.decode(packed.data(), decoded.data(), 1);

  CHECK(packed.size() == 64);

  // Verify reconstruction within step tolerance
  float max_step = sq4.step();
  for (size_t d = 0; d < dim; ++d) {
    float diff = std::abs(data[d] - decoded[d]);
    CHECK(diff <= max_step + 1e-4f);
  }
}

void test_twostage_reranker_recall() {
  const size_t dim = 128;
  const size_t num_vectors = 2000;
  const size_t k = 10;
  const size_t coarse_m = 100;

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> base_data(num_vectors * dim);
  for (size_t i = 0; i < base_data.size(); ++i) {
    base_data[i] = dist(rng);
  }
  secan::normalize_dataset_avx2(base_data.data(), num_vectors, dim);

  secan::ScalarQuantizer4 sq4(dim);
  sq4.train(num_vectors, base_data.data());

  std::vector<uint8_t> packed_base(num_vectors * sq4.bytes_per_vector());
  sq4.encode(base_data.data(), packed_base.data(), num_vectors);

  secan::TwoStageReranker reranker(sq4, packed_base.data(), base_data.data(), num_vectors, dim);

  // Generate 20 test queries
  size_t num_queries = 20;
  std::vector<float> queries(num_queries * dim);
  for (size_t i = 0; i < queries.size(); ++i) {
    queries[i] = dist(rng);
  }
  secan::normalize_dataset_avx2(queries.data(), num_queries, dim);

  size_t total_recalled = 0;
  for (size_t q = 0; q < num_queries; ++q) {
    const float* query_ptr = queries.data() + q * dim;

    // Ground truth exact scan
    std::vector<SearchResult> exact_results(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
      float dist_exact = secan::l2_squared_avx2_unroll4(query_ptr, base_data.data() + i * dim, dim);
      exact_results[i] = SearchResult{static_cast<int>(i), dist_exact};
    }
    std::partial_sort(exact_results.begin(), exact_results.begin() + k, exact_results.end(),
                      [](const SearchResult& a, const SearchResult& b) { return a.distance < b.distance; });

    // Two-stage re-ranking
    auto reranked = reranker.search(query_ptr, k, coarse_m);
    CHECK(reranked.size() == k);

    // Measure overlap (Recall@K)
    for (size_t i = 0; i < k; ++i) {
      for (size_t j = 0; j < k; ++j) {
        if (reranked[i].index == exact_results[j].index) {
          total_recalled++;
          break;
        }
      }
    }
  }

  double recall_at_10 = static_cast<double>(total_recalled) / static_cast<double>(num_queries * k);
  CHECK(recall_at_10 >= 0.95); // High recall guaranteed by two-stage filtering
}

#include "secan/quantization/quantized_dataset.h"

void test_quantized_dataset_span_views() {
  const size_t num_vectors = 10;
  const size_t dim = 128;

  // Test SQ8 Dataset View
  std::vector<uint8_t> sq8_raw(num_vectors * dim, 42);
  secan::QuantizedDatasetSQ8 dataset_sq8(num_vectors, dim, sq8_raw);

  CHECK(dataset_sq8.size() == num_vectors);
  CHECK(dataset_sq8.dim() == dim);
  CHECK(dataset_sq8.nbytes() == num_vectors * dim);

  std::span<const uint8_t> v0 = dataset_sq8.get_vector(0);
  CHECK(v0.size() == dim);
  CHECK(v0[0] == 42);

  std::span<const uint8_t> block = dataset_sq8.get_block(2, 3);
  CHECK(block.size() == 3 * dim);

  // Test SQ4 Dataset View
  const size_t bytes_per_vec = 64;
  std::vector<uint8_t> sq4_raw(num_vectors * bytes_per_vec, 0xAB);
  secan::QuantizedDatasetSQ4 dataset_sq4(num_vectors, dim, sq4_raw);

  CHECK(dataset_sq4.size() == num_vectors);
  CHECK(dataset_sq4.bytes_per_vector() == bytes_per_vec);
  CHECK(dataset_sq4.nbytes() == num_vectors * bytes_per_vec);

  std::span<const uint8_t> v0_sq4 = dataset_sq4.get_vector(0);
  CHECK(v0_sq4.size() == bytes_per_vec);
  CHECK(v0_sq4[0] == 0xAB);
}

int main() {
  test_quantizer_untrained_exceptions();
  test_quantizer_invalid_args();
  test_quantizer_constant_data();
  test_quantizer_percentile_clipping();
  test_quantizer_roundtrip_reconstruction();
  test_quantizer_sift_reconstruction();
  test_quantizer_sq4_nibble_packing();
  test_twostage_reranker_recall();
  test_quantized_dataset_span_views();
  return report_results("scalar_quantizer_tests");
}

