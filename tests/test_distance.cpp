#include "secan/search/distance.h"
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

void test_linear_scan_metric_types() {
  // Dataset with 3 vectors in 2D
  // 0: (1, 0)
  // 1: (0, 1)
  // 2: (2, 0)
  std::vector<float> dataset = {
    1.0f, 0.0f,
    0.0f, 1.0f,
    2.0f, 0.0f
  };
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

int main() {
  test_l2_squared();
  test_inner_product();
  test_cosine_distance();
  test_fast_rsqrt_and_cosine();
  test_linear_scan_metric_types();
  return report_results("distance_tests");
}
