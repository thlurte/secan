// Drill 3: SIMD Chunk & Tail-Loop Pattern
// Completed: 2026-09-15
// Concept: Processing in strides of 8 (SIMD YMM register size) with scalar remainder cleanup.

#include <cstddef>
#include <cassert>
#include <cmath>
#include <iostream>

float sum_chunks_of_8(const float* arr, size_t dim) {
    float total_sum = 0.0f;
    size_t i = 0;

    // Step 1: Process full chunks of 8
    for (; i + 8 <= dim; i += 8) {
        total_sum += arr[i] + arr[i+1] + arr[i+2] + arr[i+3] + 
                     arr[i+4] + arr[i+5] + arr[i+6] + arr[i+7];
    }

    // Step 2: Cleanup loop for remaining elements
    for (; i < dim; ++i) {
        total_sum += arr[i];
    }

    return total_sum;
}

int main() {
    // Test 1: Exactly 8 elements (no remainder)
    float a1[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    assert(std::abs(sum_chunks_of_8(a1, 8) - 8.0f) < 1e-6f);

    // Test 2: Less than 8 elements (dim = 3, only remainder loop runs)
    float a2[3] = {2.0f, 3.0f, 4.0f};
    assert(std::abs(sum_chunks_of_8(a2, 3) - 9.0f) < 1e-6f);

    // Test 3: 19 elements (2 full chunks of 8 + 3 remainder elements)
    float a3[19];
    float expected_sum = 0.0f;
    for (size_t k = 0; k < 19; ++k) {
        a3[k] = static_cast<float>(k + 1);
        expected_sum += a3[k];
    }
    assert(std::abs(sum_chunks_of_8(a3, 19) - expected_sum) < 1e-6f);

    std::cout << "Drill 3 tests passed successfully!" << std::endl;
    return 0;
}
