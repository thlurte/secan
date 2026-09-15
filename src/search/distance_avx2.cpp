#include "secan/search/distance_avx2.h"
#include <immintrin.h>

namespace secan {

void enable_ftz_daz() noexcept {
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

float l2_squared_avx2_single(const float *a, const float *b,
                             size_t dim) noexcept {

  __m256 sum = _mm256_setzero_ps();

  // 2. Loop in chunks of 8 floats
  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }

  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float total = _mm_cvtss_f32(sum128);

  i = (dim / 8) * 8;
  for (; i < dim; ++i) {
    float d = a[i] - b[i];
    total += d * d;
  }
  return total;
}

} // namespace secan