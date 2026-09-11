#include "secan/search/distance.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace secan {

float l2_squared_scalar(const float *a, const float *b, size_t dim) noexcept {
  float diff = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float r_diff = a[i] - b[i];
    diff += r_diff * r_diff;
  }
  return diff;
}

float inner_product_scalar(const float *a, const float *b, size_t dim) noexcept {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float cosine_distance_scalar(const float *a, const float *b, size_t dim) noexcept {
  float dot_prod = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    dot_prod += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  float denom = std::sqrt(norm_a * norm_b);
  if (denom <= 1e-12f) {
    return 0.0f;
  }
  float similarity = std::clamp(dot_prod / denom, -1.0f, 1.0f);
  return 1.0f - similarity;
}

float fast_rsqrt(float x) noexcept {
  // Quake III / IEEE-754 fast inverse square root with 1 Newton-Raphson iteration
  if (x <= 0.0f) {
    return 0.0f;
  }
  float xhalf = 0.5f * x;
  uint32_t i;
  std::memcpy(&i, &x, sizeof(i));
  i = 0x5f3759df - (i >> 1);
  float y;
  std::memcpy(&y, &i, sizeof(y));
  y = y * (1.5f - xhalf * y * y); // 1st Newton-Raphson iteration
  return y;
}

float cosine_distance_fast_scalar(const float *a, const float *b, size_t dim) noexcept {
  float dot_prod = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    dot_prod += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  float prod = norm_a * norm_b;
  if (prod <= 1e-12f) {
    return 0.0f;
  }
  float inv_norm = fast_rsqrt(prod);
  float similarity = std::clamp(dot_prod * inv_norm, -1.0f, 1.0f);
  return 1.0f - similarity;
}

} // namespace secan
