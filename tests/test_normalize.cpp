#include "secan/utils/normalize.h"
#include "secan/search/distance.h"
#include "secan/search/distance_avx2.h"
#include "secan/search/search.h"
#include "test_utils.h"
#include <cmath>
#include <vector>

void test_single_vector_normalization() {
  std::vector<float> vec = {3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // norm = 5.0
  secan::normalize_vector_avx2(vec.data(), vec.size());

  CHECK(std::abs(vec[0] - 0.6f) < 1e-5f);
  CHECK(std::abs(vec[1] - 0.8f) < 1e-5f);

  // Compute L2 norm of normalized vector
  float norm_sq = 0.0f;
  for (float v : vec) norm_sq += v * v;
  CHECK(std::abs(std::sqrt(norm_sq) - 1.0f) < 1e-5f);
}

void test_dataset_normalization_parity() {
  const size_t num_vectors = 100;
  const size_t dim = 128;

  secan::FloatDataset ds1;
  ds1.num_vectors = num_vectors;
  ds1.dim = dim;
  ds1.data.resize(num_vectors * dim);

  for (size_t i = 0; i < num_vectors * dim; ++i) {
    ds1.data[i] = static_cast<float>((i % 37) + 1);
  }

  secan::FloatDataset ds2 = ds1;

  secan::normalize_dataset_avx2(ds1.data.data(), num_vectors, dim);
  secan::normalize_dataset_stream(ds2.data.data(), num_vectors, dim);

  for (size_t i = 0; i < num_vectors; ++i) {
    float norm_sq1 = 0.0f;
    float norm_sq2 = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      float v1 = ds1.data[i * dim + d];
      float v2 = ds2.data[i * dim + d];
      norm_sq1 += v1 * v1;
      norm_sq2 += v2 * v2;
      CHECK(std::abs(v1 - v2) < 1e-5f);
    }
    CHECK(std::abs(std::sqrt(norm_sq1) - 1.0f) < 1e-5f);
    CHECK(std::abs(std::sqrt(norm_sq2) - 1.0f) < 1e-5f);
  }
}

void test_cosine_inner_product_equivalence() {
  const size_t num_vectors = 50;
  const size_t dim = 64;

  secan::FloatDataset ds;
  ds.num_vectors = num_vectors;
  ds.dim = dim;
  ds.data.resize(num_vectors * dim);

  for (size_t i = 0; i < num_vectors * dim; ++i) {
    ds.data[i] = static_cast<float>((i * 17) % 53 + 1);
  }

  std::vector<float> query(dim);
  for (size_t d = 0; d < dim; ++d) {
    query[d] = static_cast<float>((d * 11) % 43 + 2);
  }

  // 1. Unnormalized Cosine Search
  auto cosine_results = secan::linear_scan(ds, query.data(), 10, secan::MetricType::Cosine);

  // 2. Normalize dataset and query to unit sphere
  secan::normalize_dataset(ds);
  secan::normalize_vector_avx2(query.data(), dim);

  // 3. Pre-normalized Inner Product Search
  auto ip_results = secan::linear_scan(ds, query.data(), 10, secan::MetricType::IP);

  CHECK(cosine_results.size() == 10);
  CHECK(ip_results.size() == 10);

  // Rankings must be identical between Cosine and Pre-Normalized IP
  for (size_t i = 0; i < 10; ++i) {
    CHECK(cosine_results[i].index == ip_results[i].index);

    // Mathematical equivalence check: d_cosine = 1.0 - dot_product = 1.0 - (-stored_ip_dist)
    float ip_value = -ip_results[i].distance;
    float expected_cosine_dist = 1.0f - ip_value;
    CHECK(std::abs(cosine_results[i].distance - expected_cosine_dist) < 1e-4f);
  }
}

int main() {
  test_single_vector_normalization();
  test_dataset_normalization_parity();
  test_cosine_inner_product_equivalence();
  return report_results("normalize_tests");
}
