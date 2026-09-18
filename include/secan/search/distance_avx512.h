#ifndef SECAN_SEARCH_DISTANCE_AVX512_H
#define SECAN_SEARCH_DISTANCE_AVX512_H

#include <cstddef>

namespace secan {

float l2_squared_avx512(const float *a, const float *b, size_t dim) noexcept;

float ip_avx512(const float *a, const float *b, size_t dim) noexcept;

} // namespace secan

#endif // SECAN_SEARCH_DISTANCE_AVX512_H
