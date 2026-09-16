#ifndef SECAN_SEARCH_DISTANCE_AVX2_H
#define SECAN_SEARCH_DISTANCE_AVX2_H

#include <cstddef>

namespace secan {

void enable_ftz_daz() noexcept;

float l2_squared_avx2_single(const float *a, const float *b,
                             size_t dim) noexcept;

float l2_squared_avx2_unroll4(const float *a, const float *b,
                              size_t dim) noexcept;

float cosine_distance_avx2(const float *a, const float *b,
                           size_t dim) noexcept;

} // namespace secan

#endif // SECAN_SEARCH_DISTANCE_AVX2_H

