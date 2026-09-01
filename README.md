# secan

This is a C++ vector search library currently implementing exact brute-force $k$-nearest neighbor ($k$-NN) linear scan across vector datasets using L2 and cosine distance metrics. The project serves as an experimental testbed for CPU and GPU performance optimizations specially on SIMD vectorization, cache-aware data layouts, multi-threading, and eventual support for approximate nearest neighbor (ANN) algorithms such as HNSW.

## Build

Building `secan` requires CMake 3.15+ and a C++20-compliant compiler.

```bash
# Configure build
cmake -B build

# Build library, executable, and tests
cmake --build build

# Run test suite
ctest --test-dir build --output-on-failure
```

## Usage

`secan` operates on flat contiguous row-major vector datasets and query vectors.

```cpp
#include <iostream>
#include <vector>
#include "secan/search/search.h"

int main() {
    // 3 vectors of dimension 2 (flat row-major layout: N * dim)
    std::vector<float> dataset = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    std::vector<float> query = {3.0f, 4.0f};
    int top_k = 2;

    // Search using "l2" (squared L2) or "cosine" distance
    std::vector<SearchResult> results = linear_scan(dataset, query, top_k, "l2");

    for (const auto &result : results) {
        std::cout << "Index: " << result.index
                  << ", Distance: " << result.distance << "\n";
    }

    return 0;
}
```

Link against `secan_lib` in your `CMakeLists.txt`:

```cmake
target_link_libraries(your_target PRIVATE secan_lib)
```

## Benchmarks

Microbenchmarks are implemented using **Google Benchmark v1.9.0** with memory clobber barriers (`benchmark::DoNotOptimize`) to prevent compiler dead-code elimination.

```bash
# Build and run microbenchmarks (Release mode required)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/benchmarks/bench_distance
```

### Baseline: Scalar $L_2$ Squared Distance (Unvectorized)
*Benchmarked on 12-Core @ 4.30 GHz (L1D 32 KiB, L2 1 MiB, L3 16 MiB)*

| Vector Dimension ($D$) | Workload / Embedding Model | Latency (ns) | Memory Bandwidth (GiB/s) | 1M Vector Scan Estimate |
|:---|:---|:---:|:---:|:---:|
| **$D = 64$** | Micro Embeddings / Image Hashes | **26.8 ns** | 17.84 GiB/s | ~26.8 ms |
| **$D = 128$** | SIFT / Audio Features | **58.1 ns** | 16.49 GiB/s | ~58.1 ms |
| **$D = 768$** | BERT / Standard Vector Embeddings | **513.0 ns** | 11.21 GiB/s | **~513.0 ms** |
| **$D = 1536$** | OpenAI `text-embedding-3-small` | **1055.0 ns** | 10.89 GiB/s | **~1.05 seconds** |

## Roadmap

- SIMD-vectorized distance computation
- Cache-aware memory layout
- Multithreaded search
- HNSW as an approximate-search mode

## License

MIT License. Copyright (c) 2026 Adheeb Ahmed.