#include <iostream>
#include <iomanip>

int main() {
    std::cout << "--- Drill 12.3: 1-Pass vs 2-Pass Memory Bus Traffic Analysis ---\n\n";

    size_t dim = 10'000'000; // 10 Million dimensional vector
    double vector_bytes = dim * sizeof(float); // 40 MB per vector
    double vector_mb = vector_bytes / (1024.0 * 1024.0);

    std::cout << "Vector Dimension      : " << dim << " floats\n";
    std::cout << "Single Vector Size    : " << std::fixed << std::setprecision(1) << vector_mb << " MB\n\n";

    // 2-Pass Naive:
    // Pass 1: Reads a (40MB) + b (40MB) = 80MB
    // Pass 2: Reads a (40MB) + b (40MB) = 80MB
    double naive_reads_mb = 4 * vector_mb;

    // 1-Pass Fused:
    // Single Pass: Reads a (40MB) + b (40MB) = 80MB (Fused FMA reuses loaded registers!)
    double fused_reads_mb = 2 * vector_mb;

    double reduction_pct = ((naive_reads_mb - fused_reads_mb) / naive_reads_mb) * 100.0;

    std::cout << "2-Pass Naive Memory Read   : " << naive_reads_mb << " MB\n";
    std::cout << "1-Pass Fused Memory Read   : " << fused_reads_mb << " MB\n";
    std::cout << "Memory Bus Traffic Saved   : " << std::setprecision(1) << reduction_pct << "%\n";
    std::cout << "Effective Bandwidth Boost  : " << std::setprecision(2) << (naive_reads_mb / fused_reads_mb) << "x\n";

    return 0;
}
