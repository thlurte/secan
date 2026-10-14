#include "secan/quantization/binary_quantizer.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void test_basic_encode_decode() {
    constexpr size_t D = 128; // 2 x 64-bit words
    secan::BinaryQuantizer bq(D);

    assert(bq.dim() == 128);
    assert(bq.num_u64() == 2);
    assert(bq.bytes_per_vector() == 16);

    std::vector<float> vec(D);
    for (size_t i = 0; i < D; ++i) {
        vec[i] = (i % 2 == 0) ? 1.5f : -1.5f;
    }

    std::vector<uint64_t> packed(bq.num_u64());
    bq.encode(vec.data(), packed.data(), 1);

    // Verify bit patterns: bit i should be 1 if i is even, 0 if i is odd (0x5555555555555555)
    assert(packed[0] == 0x5555555555555555ULL);
    assert(packed[1] == 0x5555555555555555ULL);

    std::vector<float> decoded(D);
    bq.decode(packed.data(), decoded.data(), 1);

    for (size_t i = 0; i < D; ++i) {
        float expected = (i % 2 == 0) ? 1.0f : -1.0f;
        assert(decoded[i] == expected);
    }

    std::cout << "✓ test_basic_encode_decode passed!\n";
}

void test_hamming_distance() {
    constexpr size_t D = 128;
    secan::BinaryQuantizer bq(D);

    std::vector<float> a(D, 1.0f);  // All positive -> all bits 1
    std::vector<float> b(D, -1.0f); // All negative -> all bits 0

    std::vector<uint64_t> packed_a(bq.num_u64());
    std::vector<uint64_t> packed_b(bq.num_u64());

    bq.encode(a.data(), packed_a.data(), 1);
    bq.encode(b.data(), packed_b.data(), 1);

    uint32_t dist = secan::BinaryQuantizer::hamming_distance(packed_a.data(), packed_b.data(), bq.num_u64());
    assert(dist == 128); // All 128 bits differ

    std::cout << "✓ test_hamming_distance passed!\n";
}

void test_asymmetric_ip() {
    constexpr size_t D = 128;
    secan::BinaryQuantizer bq(D);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> query(D);
    std::vector<float> base_vec(D);
    for (size_t i = 0; i < D; ++i) {
        query[i] = dist(rng);
        base_vec[i] = dist(rng);
    }

    std::vector<uint64_t> packed(bq.num_u64());
    bq.encode(base_vec.data(), packed.data(), 1);

    float query_sum = 0.0f;
    for (float q : query) query_sum += q;

    // Asymmetric IP via our bit-scan implementation
    float computed_ip = bq.compute_distance_ip(query.data(), packed.data(), query_sum);

    // Ground truth signed inner product: sum(q_d * (2*b_d - 1))
    float expected_ip = 0.0f;
    for (size_t d = 0; d < D; ++d) {
        float signed_b = (base_vec[d] >= 0.0f) ? 1.0f : -1.0f;
        expected_ip += query[d] * signed_b;
    }

    assert(std::abs(computed_ip - expected_ip) < 1e-4f);

    // Test batch version
    float batch_res[1];
    bq.compute_distance_ip_batch(query.data(), packed.data(), batch_res, 1);
    assert(std::abs(batch_res[0] - expected_ip) < 1e-4f);

    std::cout << "✓ test_asymmetric_ip passed!\n";
}

int main() {
    std::cout << "--- Running BinaryQuantizer Tests ---\n";
    test_basic_encode_decode();
    test_hamming_distance();
    test_asymmetric_ip();
    std::cout << "All BinaryQuantizer tests passed successfully!\n";
    return 0;
}
