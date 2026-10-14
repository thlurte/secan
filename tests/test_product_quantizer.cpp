#include "secan/quantization/product_quantizer.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_product_quantizer_basic() {
    constexpr size_t N = 500;
    constexpr size_t D = 32;
    constexpr size_t M = 4; // d_sub = 8

    std::vector<float> data(N * D);
    for (size_t i = 0; i < N * D; ++i) {
        data[i] = static_cast<float>(i % 100) * 0.1f;
    }

    secan::ProductQuantizer pq(D, M);
    assert(pq.d_sub == 8);
    assert(pq.M == 4);

    pq.train(data.data(), N, 5);

    std::vector<uint8_t> codes(N * M);
    pq.encode(data.data(), codes.data(), N);

    std::vector<float> reconstructed(N * D);
    pq.decode(codes.data(), reconstructed.data(), N);

    // Compute reconstruction error
    float total_diff = 0.0f;
    for (size_t i = 0; i < N * D; ++i) {
        float diff = data[i] - reconstructed[i];
        total_diff += diff * diff;
    }
    float mse = total_diff / static_cast<float>(N * D);
    std::cout << "PQ Reconstruction MSE: " << mse << "\n";
    assert(mse < 5.0f && "Reconstruction MSE must be bounded!");

    // Test ADC distance
    std::vector<float> query(D, 1.0f);
    float adc_dist = pq.compute_adc_distance(query.data(), codes.data());
    assert(adc_dist >= 0.0f);

    std::cout << "✓ test_product_quantizer_basic passed!\n";
}

void test_product_quantizer_query_lut() {
    constexpr size_t N = 100;
    constexpr size_t D = 32;
    constexpr size_t M = 4; // d_sub = 8

    std::vector<float> data(N * D);
    for (size_t i = 0; i < N * D; ++i) {
        data[i] = static_cast<float>(i % 50) * 0.2f - 5.0f;
    }

    secan::ProductQuantizer pq(D, M);
    pq.train(data.data(), N, 5);

    std::vector<uint8_t> codes(N * M);
    pq.encode(data.data(), codes.data(), N);

    std::vector<float> query(D);
    for (size_t d = 0; d < D; ++d) {
        query[d] = static_cast<float>(d % 10) * 0.5f;
    }

    // 1. Precompute LUT
    std::vector<float> lut = pq.compute_query_lut(query.data());
    assert(lut.size() == M * pq.k_sub);

    // 2. Compare individual compute_adc_distance_lut with on-the-fly compute_adc_distance
    for (size_t i = 0; i < N; ++i) {
        float direct_dist = pq.compute_adc_distance(query.data(), codes.data() + i * M);
        float lut_dist = pq.compute_adc_distance_lut(lut.data(), codes.data() + i * M);
        assert(std::abs(direct_dist - lut_dist) < 1e-4f);
    }

    // 3. Compare batch scan_adc_lut with individual LUT calls
    std::vector<float> batch_distances(N);
    pq.scan_adc_lut(lut.data(), codes.data(), batch_distances.data(), N);

    for (size_t i = 0; i < N; ++i) {
        float expected_dist = pq.compute_adc_distance_lut(lut.data(), codes.data() + i * M);
        assert(std::abs(batch_distances[i] - expected_dist) < 1e-5f);
    }

    std::cout << "✓ test_product_quantizer_query_lut passed!\n";
}

int main() {
    std::cout << "--- Running Product Quantizer Tests ---\n";
    test_product_quantizer_basic();
    test_product_quantizer_query_lut();
    std::cout << "All Product Quantizer tests passed successfully!\n";
    return 0;
}
