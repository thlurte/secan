#include "secan/quantization/fastscan.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

void test_construction() {
    std::cout << "Running test_construction... ";
    secan::FastScan4Bit fs(128, 16);
    assert(fs.dim() == 128);
    assert(fs.M() == 16);
    assert(fs.d_sub() == 8);

    bool caught = false;
    try {
        secan::FastScan4Bit invalid(128, 7);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
    std::cout << "PASSED\n";
}

void test_packing_and_scan_parity() {
    std::cout << "Running test_packing_and_scan_parity... ";
    const size_t dim = 64;
    const size_t M = 8;
    const size_t d_sub = dim / M; // 8
    const size_t K = secan::FastScan4Bit::K_SUB; // 16
    const size_t num_vectors = 67; // Non-multiple of 32 to test boundary handling

    secan::FastScan4Bit fs(dim, M);

    // Generate random codebooks: M * 16 * d_sub
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_f(0.0f, 1.0f);
    std::vector<float> codebooks(M * K * d_sub);
    for (auto& val : codebooks) {
        val = dist_f(rng);
    }

    // Generate random 4-bit codes for num_vectors
    std::uniform_int_distribution<uint8_t> dist_code(0, 15);
    std::vector<uint8_t> src_codes(num_vectors * M);
    for (auto& c : src_codes) {
        c = dist_code(rng);
    }

    // Pack codes
    size_t num_blocks = (num_vectors + 31) / 32;
    std::vector<uint8_t> packed_codes(num_blocks * M * 16, 0);
    fs.pack_codes(src_codes.data(), packed_codes.data(), num_vectors);

    // Random query vector
    std::vector<float> query(dim);
    for (auto& q : query) {
        q = dist_f(rng);
    }

    // Compute uint8 query LUT
    std::vector<uint8_t> lut(M * K);
    fs.compute_query_lut_uint8(query.data(), codebooks.data(), lut.data());

    // Execute FastScan
    std::vector<uint16_t> scan_results(num_vectors, 0);
    fs.scan(packed_codes.data(), lut.data(), scan_results.data(), num_vectors);

    // Verify parity against ground-truth scalar accumulation of LUT entries
    for (size_t i = 0; i < num_vectors; ++i) {
        uint16_t expected_sum = 0;
        for (size_t m = 0; m < M; ++m) {
            uint8_t code = src_codes[i * M + m];
            assert(code < 16);
            expected_sum += lut[m * 16 + code];
        }
        if (scan_results[i] != expected_sum) {
            std::cerr << "Mismatch at vector " << i << ": fastscan=" << scan_results[i]
                      << ", expected=" << expected_sum << "\n";
            assert(false);
        }
    }
    std::cout << "PASSED\n";
}

void test_lookup_32bytes_avx2() {
    std::cout << "Running test_lookup_32bytes_avx2... ";
    // Test the 32-byte in-register pshufb helper
    alignas(32) uint8_t lut_16[16];
    for (uint8_t i = 0; i < 16; ++i) {
        lut_16[i] = static_cast<uint8_t>(i * 10);
    }
    __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_16));
    __m256i lut256 = _mm256_set_m128i(lut128, lut128);

    alignas(32) uint8_t packed_codes[32];
    for (size_t i = 0; i < 32; ++i) {
        uint8_t low = static_cast<uint8_t>(i % 16);
        uint8_t high = static_cast<uint8_t>((15 - i) % 16);
        packed_codes[i] = static_cast<uint8_t>(low | (high << 4));
    }

    alignas(32) uint8_t out_distances[64];
    secan::FastScan4Bit::lookup_32bytes_avx2(packed_codes, lut256, out_distances);

    // out_distances[0..31] are low nibble lookups
    for (size_t i = 0; i < 32; ++i) {
        uint8_t expected_low = lut_16[i % 16];
        assert(out_distances[i] == expected_low);
    }
    // out_distances[32..63] are high nibble lookups
    for (size_t i = 0; i < 32; ++i) {
        uint8_t expected_high = lut_16[(15 - i) % 16];
        assert(out_distances[32 + i] == expected_high);
    }

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== Running FastScan 4-bit Unit Tests ===\n";
    test_construction();
    test_lookup_32bytes_avx2();
    test_packing_and_scan_parity();
    std::cout << "All FastScan unit tests passed successfully!\n";
    return 0;
}
