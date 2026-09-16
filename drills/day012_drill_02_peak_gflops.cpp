#include <iostream>
#include <iomanip>

int main() {
    std::cout << "--- Drill 12.2: Theoretical Peak Single-Core GFLOPS ---\n\n";

    double clock_ghz = 4.30;
    int fma_pipes = 2;              // AMD Zen 4 has 2 FMA pipes (Pipe 0 & Pipe 1)
    int floats_per_ymm = 8;         // 256-bit AVX2 / 32-bit float = 8 floats
    int ops_per_fma = 2;            // 1 Multiply + 1 Add = 2 FLOPs per lane

    double peak_flops_per_cycle = fma_pipes * floats_per_ymm * ops_per_fma;
    double peak_gflops = clock_ghz * peak_flops_per_cycle;

    std::cout << "CPU Clock Frequency       : " << clock_ghz << " GHz\n";
    std::cout << "Parallel FMA Pipes / Core : " << fma_pipes << " units\n";
    std::cout << "Floats per 256-bit Vector : " << floats_per_ymm << " floats\n";
    std::cout << "FLOPs per FMA Lane        : " << ops_per_fma << " ops (1 mul + 1 add)\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Peak FLOPs per Cycle      : " << peak_flops_per_cycle << " FLOPs/cycle\n";
    std::cout << "Peak Compute Throughput   : " << std::fixed << std::setprecision(1) 
              << peak_gflops << " GFLOPS (Billion FLOPs/sec per core)\n";

    return 0;
}
