#include "secan/quantization/distance_sq8.h"
#include "test_utils.h"
#include <cstdint>
#include <random>
#include <vector>

void test_l2_squared_sq8_basic() {
  uint8_t a[] = {10, 20, 30};
  uint8_t b[] = {15, 25, 40};
  // (10-15)^2 + (20-25)^2 + (30-40)^2 = 25 + 25 + 100 = 150
  CHECK(secan::l2_squared_sq8_scalar(a, b, 3) == 150);
  CHECK(secan::l2_squared_sq8_avx2(a, b, 3) == 150);
  CHECK(secan::l2_squared_sq8(a, b, 3) == 150);

  // Symmetry
  CHECK(secan::l2_squared_sq8_avx2(b, a, 3) == 150);

  // Self-distance
  CHECK(secan::l2_squared_sq8_avx2(a, a, 3) == 0);
}

void test_l2_squared_sq8_parity() {
  for (size_t dim : {1, 3, 7, 8, 15, 16, 19, 31, 32, 45, 64, 128, 768, 1536, 4096}) {
    std::vector<uint8_t> u(dim), v(dim);
    std::mt19937 rng(static_cast<unsigned int>(dim));
    std::uniform_int_distribution<int> dist(0, 255);

    for (size_t i = 0; i < dim; ++i) {
      u[i] = static_cast<uint8_t>(dist(rng));
      v[i] = static_cast<uint8_t>(dist(rng));
    }

    uint32_t scalar_res = secan::l2_squared_sq8_scalar(u.data(), v.data(), dim);
    uint32_t avx2_res = secan::l2_squared_sq8_avx2(u.data(), v.data(), dim);
    CHECK(avx2_res == scalar_res);
  }
}

void test_l2_squared_sq8_extremes() {
  // Extreme values: 0 vs 255 across 128 dimensions
  const size_t dim = 128;
  std::vector<uint8_t> zeros(dim, 0);
  std::vector<uint8_t> maxs(dim, 255);

  uint32_t expected = static_cast<uint32_t>(255 * 255 * dim);
  uint32_t avx2_res = secan::l2_squared_sq8_avx2(zeros.data(), maxs.data(), dim);
  CHECK(avx2_res == expected);
}

void test_inner_product_sq8_parity() {
  for (size_t dim : {1, 3, 7, 8, 15, 16, 19, 31, 32, 45, 64, 128, 768, 1536, 4096}) {
    std::vector<uint8_t> u(dim), v(dim);
    std::mt19937 rng(static_cast<unsigned int>(dim + 100));
    std::uniform_int_distribution<int> dist(0, 255);

    for (size_t i = 0; i < dim; ++i) {
      u[i] = static_cast<uint8_t>(dist(rng));
      v[i] = static_cast<uint8_t>(dist(rng));
    }

    uint32_t scalar_res = secan::inner_product_sq8_scalar(u.data(), v.data(), dim);
    uint32_t avx2_res = secan::inner_product_sq8_avx2(u.data(), v.data(), dim);
    CHECK(avx2_res == scalar_res);
  }
}

int main() {
  test_l2_squared_sq8_basic();
  test_l2_squared_sq8_parity();
  test_l2_squared_sq8_extremes();
  test_inner_product_sq8_parity();
  return report_results("distance_sq8_tests");
}
