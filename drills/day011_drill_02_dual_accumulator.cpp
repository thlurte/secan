// Drill 2: Breaking Dependency Chains (Dual Accumulators)
// Completed: 2026-09-15
// Concept: Breaking 4-cycle FMA latency dependencies via dual independent accumulator registers.

#include <cstddef>
#include <cassert>
#include <cmath>
#include <iostream>

float dot_product_2acc(const float* a, const float* b, size_t dim) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    
    size_t i = 0;

    // Step 1: Main unrolled loop (2 elements per iteration)
    for (; i + 1 < dim; i += 2) {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
    }

    // Step 2: Tail cleanup (for odd dim)
    for (; i < dim; ++i) {
        sum0 += a[i] * b[i];
    }

    // Step 3: Combine accumulators
    return sum0 + sum1;
}

int main() {
    // Test 1: Even dimension (dim = 4)
    // a = [1, 2, 3, 4], b = [2, 3, 4, 5]
    // dot = 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    float a1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b1[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float res1 = dot_product_2acc(a1, b1, 4);
    assert(std::abs(res1 - 40.0f) < 1e-6f);

    // Test 2: Odd dimension (dim = 5, exercises tail loop)
    // a = [1, 2, 3, 4, 5], b = [2, 1, 2, 1, 2]
    // dot = 2 + 2 + 6 + 4 + 10 = 24
    float a2[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b2[5] = {2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
    float res2 = dot_product_2acc(a2, b2, 5);
    assert(std::abs(res2 - 24.0f) < 1e-6f);

    std::cout << "Drill 2 tests passed successfully!" << std::endl;
    return 0;
}
