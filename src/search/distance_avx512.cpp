#include "secan/search/distance_avx512.h"
#include "secan/search/distance_avx2.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vl,fma")
#endif

#include <immintrin.h>

namespace secan {

float l2_squared_avx512(const float *a, const float *b, size_t dim) noexcept {
#ifdef __AVX512F__
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();

    size_t i = 0;
    // Process 32 elements per iteration using 2 ZMM accumulators
    for (; i + 32 <= dim; i += 32) {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        __m512 d0  = _mm512_sub_ps(va0, vb0);
        sum0 = _mm512_fmadd_ps(d0, d0, sum0);

        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        __m512 d1  = _mm512_sub_ps(va1, vb1);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
    }

    __m512 sum = _mm512_add_ps(sum0, sum1);

    // Vector tail loop (16 floats in one ZMM register)
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 d  = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(d, d, sum);
    }

    float total = _mm512_reduce_add_ps(sum);

    // Scalar tail cleanup
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
#else
    return l2_squared_avx2_unroll4(a, b, dim);
#endif
}

float ip_avx512(const float *a, const float *b, size_t dim) noexcept {
#ifdef __AVX512F__
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        sum0 = _mm512_fmadd_ps(va0, vb0, sum0);

        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        sum1 = _mm512_fmadd_ps(va1, vb1, sum1);
    }

    __m512 sum = _mm512_add_ps(sum0, sum1);

    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    float total = _mm512_reduce_add_ps(sum);

    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }
    return total;
#else
    return ip_avx2_unroll4(a, b, dim);
#endif
}

} // namespace secan
