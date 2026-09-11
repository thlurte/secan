#ifndef SECAN_SEARCH_DISTANCE_H
#define SECAN_SEARCH_DISTANCE_H

#include <cstddef>
#include <cstdint>

namespace secan {

enum class MetricType {
  L2,           // Squared Euclidean Distance: ||u - v||^2
  IP,           // Inner Product (Dot Product): <u, v> (MIPS)
  Cosine        // Cosine Distance: 1.0 - <u, v> / (||u|| * ||v||)
};

float l2_squared_scalar(const float *a, const float *b, size_t dim) noexcept;
float inner_product_scalar(const float *a, const float *b, size_t dim) noexcept;
float cosine_distance_scalar(const float *a, const float *b, size_t dim) noexcept;

/// Fast reciprocal square root approximation: 1.0f / sqrtf(x)
float fast_rsqrt(float x) noexcept;

/// Fast cosine distance using fast reciprocal square root
float cosine_distance_fast_scalar(const float *a, const float *b, size_t dim) noexcept;

} // namespace secan

// Backward compatibility & convenience wrappers
inline float l2_squared(const float *a, const float *b, size_t dim) {
  return secan::l2_squared_scalar(a, b, dim);
}

inline float inner_product(const float *a, const float *b, size_t dim) {
  return secan::inner_product_scalar(a, b, dim);
}

inline float cosine_distance(const float *a, const float *b, size_t dim) {
  return secan::cosine_distance_scalar(a, b, dim);
}

#endif // SECAN_SEARCH_DISTANCE_H
