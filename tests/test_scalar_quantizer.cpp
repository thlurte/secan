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

int main() {
  test_quantizer_untrained_exceptions();
  test_quantizer_invalid_args();
  test_quantizer_constant_data();
  test_quantizer_percentile_clipping();
  test_quantizer_roundtrip_reconstruction();
  test_quantizer_sift_reconstruction();
  return report_results("scalar_quantizer_tests");
}
