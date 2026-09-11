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

### Baseline: Scalar Vector Distance Kernels (Unvectorized)
*Benchmarked on AMD Zen 4 Hawk Point 12-Core @ 4.30 GHz (L1D 32 KiB, L2 1 MiB, L3 16 MiB). Compiler: Release `-O3 -march=native -DNDEBUG`.*

#### Hardware PMU Performance Counters (`perf stat`, $D = 128$)

| Kernel | Dimension ($D$) | Latency (ns) | IPC | L1D Miss Rate | Branch Miss Rate | Throughput (GiB/s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scalar `l2_squared`** | 128 | **57.7 ns** | **2.18** | **0.007%** | **0.003%** | **16.53 GiB/s** |
| **Scalar `inner_product`** | 128 | **54.4 ns** | **2.17** | **0.010%** | **0.003%** | **17.54 GiB/s** |
| **Scalar `cosine_distance`** | 128 | **88.8 ns** | **2.14** | **0.011%** | **0.003%** | **10.74 GiB/s** |
| **Scalar `cosine_fast` (rsqrt)** | 128 | **89.0 ns** | **2.14** | **0.011%** | **0.003%** | **10.71 GiB/s** |

#### Dimensional Scaling Sweep ($D \in [64, 1536]$)

| Vector Dimension ($D$) | Workload / Embedding Model | $L_2^2$ Latency | IP Latency | Cosine Latency | Effective Bandwidth |
|:---|:---|:---:|:---:|:---:|:---:|
| **$D = 64$** | Micro Embeddings / Image Hashes | **38.9 ns** | **26.0 ns** | 44.4 ns | 18.34 GiB/s |
| **$D = 128$** | SIFT1M / Audio Features | **58.1 ns** | **55.6 ns** | 88.8 ns | 17.15 GiB/s |
| **$D = 256$** | Compact Dense Representations | **151.7 ns** | **147.7 ns** | 179.7 ns | 12.91 GiB/s |
| **$D = 512$** | Small Language Embeddings | **328.6 ns** | **323.6 ns** | 359.5 ns | 11.79 GiB/s |
| **$D = 768$** | BERT / `all-mpnet-base-v2` | **509.6 ns** | **504.3 ns** | 541.0 ns | 11.35 GiB/s |
| **$D = 1024$** | BGE-Large / Large Text Embeddings | **693.7 ns** | **687.1 ns** | 723.2 ns | 11.10 GiB/s |
| **$D = 1536$** | OpenAI `text-embedding-3-small` | **1050.0 ns** | **1046.1 ns** | 1089.3 ns | 10.94 GiB/s |

## Roadmap

- SIMD-vectorized distance computation
- Cache-aware memory layout
- Multithreaded search
- HNSW as an approximate-search mode

## License

MIT License. Copyright (c) 2026 Adheeb Ahmed.