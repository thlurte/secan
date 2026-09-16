#include <iostream>
#include <immintrin.h>
#include <cmath>
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

void normalize_l2_inplace(float* vec, size_t dim) noexcept {
    // Pass 1: Compute squared norm ||vec||^2
    __m256 sum_sq = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
    }
    float norm_sq = horizontal_sum_avx2(sum_sq);

    // Scalar tail for norm calculation
    for (; i < dim; ++i) {
        norm_sq += vec[i] * vec[i];
    }

    // Epsilon guard: Prevent division by zero
    if (norm_sq <= 1e-12f) {
        return;
    }

    // Pass 2: In-place broadcast scaling
    float inv_norm = 1.0f / std::sqrt(norm_sq);
    __m256 v_inv = _mm256_set1_ps(inv_norm);

    i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        v = _mm256_mul_ps(v, v_inv);
        _mm256_storeu_ps(vec + i, v);
    }

    // Scalar tail for scaling
    for (; i < dim; ++i) {
        vec[i] *= inv_norm;
    }
}

int main() {
    std::cout << "--- Drill 11.4: In-Place L2 Normalization ---\n";

    // Test 1: 3D vector (3, 4, 0) -> norm = 5 -> normalized = (0.6, 0.8, 0.0)
    std::vector<float> v3 = {3.0f, 4.0f, 0.0f};
    normalize_l2_inplace(v3.data(), v3.size());
    std::cout << "1. Normalized 3D vector: (" << v3[0] << ", " << v3[1] << ", " << v3[2] << ")\n";
    assert(std::abs(v3[0] - 0.6f) < 1e-5f);
    assert(std::abs(v3[1] - 0.8f) < 1e-5f);

    // Test 2: 10D vector (8-chunk + tail) -> Check that ||vec||_2 == 1.0
    std::vector<float> v10 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    normalize_l2_inplace(v10.data(), v10.size());
    
    float new_norm_sq = 0.0f;
    for (float x : v10) new_norm_sq += x * x;
    std::cout << "2. New norm of 10D vector: " << std::sqrt(new_norm_sq) << " (expected 1.0)\n";
    assert(std::abs(std::sqrt(new_norm_sq) - 1.0f) < 1e-5f);

    // Test 3: Zero vector guard
    std::vector<float> zero(16, 0.0f);
    normalize_l2_inplace(zero.data(), zero.size());
    std::cout << "3. Zero vector after normalization: (" << zero[0] << ", " << zero[1] << ") (No NaN!)\n";
    assert(!std::isnan(zero[0]));

    std::cout << "\nAll test cases passed cleanly!\n";
    return 0;
}
