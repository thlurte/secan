#include <iostream>
#include <immintrin.h>
#include <cassert>

void fma_8_elements(const float* a, const float* b, float* out) {
    __m256 va = _mm256_loadu_ps(a);
    __m256 vb = _mm256_loadu_ps(b);
    __m256 acc = _mm256_setzero_ps();
    acc = _mm256_fmadd_ps(va, vb, acc);
    _mm256_storeu_ps(out, acc);
}

int main() {
    std::cout << "--- Drill 10.2: SIMD Load & FMA Operations ---\n";

    float a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[8] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float out[8] = {0};

    fma_8_elements(a, b, out);

    std::cout << "Results out[0..7]:\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "out[" << i << "] = " << out[i] << " (expected " << a[i] * b[i] << ")\n";
        assert(out[i] == a[i] * b[i]);
    }
    std::cout << "All 8 parallel lanes verified!\n";
    return 0;
}
