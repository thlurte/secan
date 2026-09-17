#include "secan/search/distance.h"
#include "secan/search/distance_avx2.h"
#include "secan/search/search.h"
#include "test_utils.h"
#include <cmath>
#include <vector>

void test_l2_squared() {
  float a[] = {1.0f, 2.0f, 3.0f};
  float b[] = {4.0f, 5.0f, 6.0f};
  CHECK(secan::l2_squared_scalar(a, b, 3) == 27.0f);
  CHECK(l2_squared(a, b, 3) == 27.0f);

  // Symmetry Property
  CHECK(l2_squared(a, b, 3) == 27.0f && l2_squared(b, a, 3) == 27.0f);

  // Self-Distance
  float same[] = {1.0f, 2.0f, 3.0f};
  CHECK(l2_squared(same, same, 3) == 0.0f);
}

void test_inner_product() {
  float a[] = {1.0f, 2.0f, 3.0f};
  float b[] = {4.0f, 5.0f, 6.0f};
  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
  CHECK(secan::inner_product_scalar(a, b, 3) == 32.0f);
  CHECK(inner_product(a, b, 3) == 32.0f);

  // Symmetry / Commutativity
  CHECK(inner_product(a, b, 3) == inner_product(b, a, 3));

  // Orthogonal vectors
  float x[] = {1.0f, 0.0f};
  float y[] = {0.0f, 1.0f};
  CHECK(inner_product(x, y, 2) == 0.0f);

  // Negative dot product
  float neg[] = {-1.0f, -2.0f, -3.0f};
  CHECK(inner_product(a, neg, 3) == -14.0f);
}

void test_cosine_distance() {
  // Identical Vectors
  float a[] = {1.0f, 2.0f, 3.0f};
  CHECK_NEAR(cosine_distance(a, a, 3), 0.0f, 1e-5f);
  CHECK_NEAR(secan::cosine_distance_scalar(a, a, 3), 0.0f, 1e-5f);

  // Scale Invariance: cosine_distance(2*a, a) == 0
  float a_scaled[] = {2.0f, 4.0f, 6.0f};
  CHECK_NEAR(cosine_distance(a, a_scaled, 3), 0.0f, 1e-5f);

  // Opposite vectors: cosine distance should be 2.0
  float a_neg[] = {-1.0f, -2.0f, -3.0f};
  CHECK_NEAR(cosine_distance(a, a_neg, 3), 2.0f, 1e-5f);

  // Orthogonal vectors: cosine distance should be 1.0
  float x[] = {1.0f, 0.0f};
  float y[] = {0.0f, 1.0f};
  CHECK_NEAR(cosine_distance(x, y, 2), 1.0f, 1e-5f);
}

void test_fast_rsqrt_and_cosine() {
  // Check fast_rsqrt accuracy against 1.0f / sqrtf(x)
  float test_vals[] = {0.5f, 1.0f, 2.0f, 4.0f, 16.0f, 100.0f, 1024.0f};
  for (float v : test_vals) {
    float exact = 1.0f / std::sqrt(v);
    float approx = secan::fast_rsqrt(v);
    float rel_err = std::abs(approx - exact) / exact;
    // 1 Newton-Raphson iteration typically gives < 0.2% (0.002) relative error
    CHECK(rel_err < 0.002f);
  }

  // Check fast cosine distance vs standard cosine distance
  float u[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float v[] = {2.0f, 1.0f, 4.0f, 3.0f};
  float exact_cos = secan::cosine_distance_scalar(u, v, 4);
  float fast_cos = secan::cosine_distance_fast_scalar(u, v, 4);
  CHECK_NEAR(fast_cos, exact_cos, 0.005f);
}

void test_l2_squared_avx2_single() {
  secan::enable_ftz_daz();

  // Test 1: Small dimension with tail only (dim = 3)
  float a_small[] = {1.0f, 2.0f, 3.0f};
  float b_small[] = {4.0f, 5.0f, 6.0f};
  CHECK(secan::l2_squared_avx2_single(a_small, b_small, 3) == 27.0f);

  // Test 2: Multiple dimensions (including odd tail) compared to scalar
  // reference
  for (size_t dim : {8, 16, 19, 64, 128}) {
    std::vector<float> u(dim), v(dim);
    for (size_t i = 0; i < dim; ++i) {
      u[i] = static_cast<float>(i + 1) * 0.5f;
      v[i] = static_cast<float>(i + 2) * 0.25f;
    }
    float scalar_res = secan::l2_squared_scalar(u.data(), v.data(), dim);
    float avx2_res = secan::l2_squared_avx2_single(u.data(), v.data(), dim);
    CHECK_NEAR(avx2_res, scalar_res, 1e-4f);
  }
}

void test_linear_scan_metric_types() {
  // Dataset with 3 vectors in 2D
  // 0: (1, 0)
  // 1: (0, 1)
  // 2: (2, 0)
  std::vector<float> dataset = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f};
  std::vector<float> query = {1.0f, 0.0f};

  // For IP:
  // dot(0, q) = 1.0
  // dot(1, q) = 0.0
  // dot(2, q) = 2.0
  // Best match for IP (MIPS) should be vector 2!
  auto res_ip = secan::linear_scan(dataset, query, 3, secan::MetricType::IP);
  CHECK(res_ip.size() == 3);
  CHECK(res_ip[0].index == 2); // highest dot product = 2.0
  CHECK(res_ip[1].index == 0); // dot product = 1.0
  CHECK(res_ip[2].index == 1); // dot product = 0.0

  // For L2:
  // l2(0, q) = 0
  // l2(1, q) = 1 + 1 = 2
  // l2(2, q) = (2-1)^2 = 1
  // Ranking: 0 (dist 0), 2 (dist 1), 1 (dist 2)
  auto res_l2 = secan::linear_scan(dataset, query, 3, secan::MetricType::L2);
  CHECK(res_l2.size() == 3);
  CHECK(res_l2[0].index == 0);
  CHECK(res_l2[1].index == 2);
  CHECK(res_l2[2].index == 1);
}

void test_l2_squared_avx2_unroll4() {
  secan::enable_ftz_daz();

  // Test across multiple dimensions (small tail only, full 32-chunk, 32-chunk + vector tail + scalar tail)
  for (size_t dim : {3, 8, 16, 32, 45, 64, 128, 768, 1536}) {
    std::vector<float> u(dim), v(dim);
    for (size_t i = 0; i < dim; ++i) {
      u[i] = std::sin(static_cast<float>(i + 1));
      v[i] = std::cos(static_cast<float>(i + 2));
    }
    float scalar_res = secan::l2_squared_scalar(u.data(), v.data(), dim);
    float unroll4_res = secan::l2_squared_avx2_unroll4(u.data(), v.data(), dim);
    CHECK_NEAR(unroll4_res, scalar_res, 1e-2f);
  }
}

void test_cosine_distance_avx2() {
  secan::enable_ftz_daz();

  // 1. Basic geometric properties
  float a[] = {1.0f, 2.0f, 3.0f};
  CHECK_NEAR(secan::cosine_distance_avx2(a, a, 3), 0.0f, 1e-5f);

  float a_scaled[] = {2.0f, 4.0f, 6.0f};
  CHECK_NEAR(secan::cosine_distance_avx2(a, a_scaled, 3), 0.0f, 1e-5f);

  float a_neg[] = {-1.0f, -2.0f, -3.0f};
  CHECK_NEAR(secan::cosine_distance_avx2(a, a_neg, 3), 2.0f, 1e-5f);

  float x[] = {1.0f, 0.0f};
  float y[] = {0.0f, 1.0f};
  CHECK_NEAR(secan::cosine_distance_avx2(x, y, 2), 1.0f, 1e-5f);

  // Zero-vector handling (safe zero-denominator check without NaN)
  float zeros[] = {0.0f, 0.0f, 0.0f};
  CHECK(secan::cosine_distance_avx2(zeros, a, 3) == 0.0f);
  CHECK(secan::cosine_distance_avx2(zeros, zeros, 3) == 0.0f);

  // 2. Multi-dimensional precision parity against scalar reference
  for (size_t dim : {3, 7, 8, 16, 19, 32, 64, 128, 768, 1536}) {
    std::vector<float> u(dim), v(dim);
    for (size_t i = 0; i < dim; ++i) {
      u[i] = std::sin(static_cast<float>(i + 1) * 0.7f);
      v[i] = std::cos(static_cast<float>(i + 2) * 0.5f);
    }
    float scalar_res = secan::cosine_distance_scalar(u.data(), v.data(), dim);
    float avx2_res = secan::cosine_distance_avx2(u.data(), v.data(), dim);
    CHECK_NEAR(avx2_res, scalar_res, 1e-4f);
  }
}

void test_ip_avx2_unroll4() {
  secan::enable_ftz_daz();

  for (size_t dim : {1, 3, 7, 8, 15, 16, 31, 32, 45, 64, 128, 768, 1536}) {
    std::vector<float> u(dim), v(dim);
    for (size_t i = 0; i < dim; ++i) {
      u[i] = std::sin(static_cast<float>(i + 1) * 0.3f);
      v[i] = std::cos(static_cast<float>(i + 2) * 0.7f);
    }
    float scalar_res = secan::inner_product_scalar(u.data(), v.data(), dim);
    float avx2_res = secan::ip_avx2_unroll4(u.data(), v.data(), dim);
    CHECK_NEAR(avx2_res, scalar_res, 1e-3f);
  }
}

int main() {
  test_l2_squared();
  test_inner_product();
  test_cosine_distance();
  test_fast_rsqrt_and_cosine();
  test_linear_scan_metric_types();
  test_l2_squared_avx2_single();
  test_l2_squared_avx2_unroll4();
  test_cosine_distance_avx2();
  test_ip_avx2_unroll4();
  return report_results("distance_tests");
}


