#include <iostream>
#include <immintrin.h>
#include <limits>
#include <iomanip>

inline void enable_ftz_daz() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

int main() {
    std::cout << "--- Drill 10.1: MXCSR FTZ/DAZ Modes ---\n";

    // Smallest positive normal float is approx 1.175494e-38
    float min_normal = std::numeric_limits<float>::min();
    std::cout << "Smallest normal float: " << min_normal << "\n\n";

    // Create a denormal by dividing by 2 (exponent becomes 0, bits drop to subnormal)
    volatile float denormal = min_normal / 2.0f;

    std::cout << "1. BEFORE enable_ftz_daz():\n";
    std::cout << "   denormal value = " << denormal << "\n";
    std::cout << "   Is it zero?    = " << (denormal == 0.0f ? "YES" : "NO (It's a slow denormal!)") << "\n\n";

    // Now enable FTZ and DAZ
    enable_ftz_daz();

    // Perform an operation that underflows below normal range
    volatile float underflow_result = min_normal * 0.5f;

    std::cout << "2. AFTER enable_ftz_daz():\n";
    std::cout << "   underflow result = " << underflow_result << "\n";
    std::cout << "   Is it zero?      = " << (underflow_result == 0.0f ? "YES (Hardware flushed directly to 0.0f!)" : "NO") << "\n";

    return 0;
}
