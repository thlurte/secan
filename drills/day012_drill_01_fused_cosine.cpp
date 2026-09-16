#include <iostream>
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>

inline float horizontal_sum_avx2(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

float cosine_distance_avx2(const float* a, const float* b, size_t dim) noexcept {
    __m256 sum_dot = _mm256_setzero_ps();
    __m256 sum_na  = _mm256_setzero_ps();
    __m256 sum_nb  = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        sum_dot = _mm256_fmadd_ps(va, vb, sum_dot);
        sum_na  = _mm256_fmadd_ps(va, va, sum_na);
        sum_nb  = _mm256_fmadd_ps(vb, vb, sum_nb);
    }

    float dot = horizontal_sum_avx2(sum_dot);
    float na  = horizontal_sum_avx2(sum_na);
    float nb  = horizontal_sum_avx2(sum_nb);

    // Scalar tail
    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }

    float denom = std::sqrt(na * nb);
    if (denom <= 1e-12f) {
        return 0.0f;
    }
    float similarity = std::clamp(dot / denom, -1.0f, 1.0f);
    return 1.0f - similarity;
}

int main() {
    std::cout << "--- Drill 12.1: Fused 1-Pass AVX2 Cosine Distance ---\n";

    // Test 1: Identical vectors (cosine distance = 0.0)
    std::vector<float> u = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float dist_self = cosine_distance_avx2(u.data(), u.data(), u.size());
    std::cout << "1. Self distance (dim=10): " << dist_self << " (expected 0.0)\n";
    assert(std::abs(dist_self) < 1e-6f);

    // Test 2: Orthogonal vectors in 2D (padded to 9D with tail)
    std::vector<float> a(9, 0.0f), b(9, 0.0f);
    a[0] = 1.0f; // a = (1, 0, 0, ...)
    b[1] = 1.0f; // b = (0, 1, 0, ...)
    float dist_ortho = cosine_distance_avx2(a.data(), b.data(), 9);
    std::cout << "2. Orthogonal distance: " << dist_ortho << " (expected 1.0)\n";
    assert(std::abs(dist_ortho - 1.0f) < 1e-6f);

    // Test 3: Zero vector guard
    std::vector<float> zero(16, 0.0f);
    float dist_zero = cosine_distance_avx2(zero.data(), u.data(), 10);
    std::cout << "3. Zero-vector guard distance: " << dist_zero << " (expected 0.0, no NaN!)\n";
    assert(!std::isnan(dist_zero));

    std::cout << "\nAll test cases passed cleanly!\n";
    return 0;
}
