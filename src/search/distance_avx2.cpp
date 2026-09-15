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

float l2_squared_avx2_unroll4(const float *a, const float *b,
                              size_t dim) noexcept {

  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  size_t i = 0;

  for (; i + 32 <= dim; i += 32) {

    __m256 va0 = _mm256_loadu_ps(a + i);
    __m256 vb0 = _mm256_loadu_ps(b + i);
    __m256 diff0 = _mm256_sub_ps(va0, vb0);
    acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

    __m256 va1 = _mm256_loadu_ps(a + i + 8);
    __m256 vb1 = _mm256_loadu_ps(b + i + 8);
    __m256 diff1 = _mm256_sub_ps(va1, vb1);
    acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

    __m256 va2 = _mm256_loadu_ps(a + i + 16);
    __m256 vb2 = _mm256_loadu_ps(b + i + 16);
    __m256 diff2 = _mm256_sub_ps(va2, vb2);
    acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

    __m256 va3 = _mm256_loadu_ps(a + i + 24);
    __m256 vb3 = _mm256_loadu_ps(b + i + 24);
    __m256 diff3 = _mm256_sub_ps(va3, vb3);
    acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
  }

  // Combine the 4 independent accumulators into a single __m256 sum
  __m256 sum01 = _mm256_add_ps(acc0, acc1);
  __m256 sum23 = _mm256_add_ps(acc2, acc3);
  __m256 sum = _mm256_add_ps(sum01, sum23);

  // Vector tail loop for remaining full 8-float chunks
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    sum = _mm256_fmadd_ps(diff, diff, sum);
  }

  // Horizontal reduction
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float total = _mm_cvtss_f32(sum128);

  // Scalar tail cleanup
  for (; i < dim; ++i) {
    float d = a[i] - b[i];
    total += d * d;
  }

  return total;
}

} // namespace secan