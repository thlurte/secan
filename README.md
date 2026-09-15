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

### Baseline vs SIMD Vectorized Distance Kernels (AVX2 & FMA)
*Benchmarked on AMD Zen 4 Hawk Point 12-Core @ 4.30 GHz (L1D 32 KiB, L2 1 MiB, L3 16 MiB). Compiler: Release `-O3 -mavx2 -mfma -DNDEBUG`.*

#### $L_2^2$ Vector Distance Dimensional Scaling Sweep ($D \in [64, 1536]$)

| Vector Dimension ($D$) | Workload / Embedding Model | Scalar Baseline | AVX2 Single (1-acc) | **AVX2 Unroll-4 (4-acc)** | **Overall Speedup** | **Peak Bandwidth** |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **$D = 64$** | Micro Embeddings / Image Hashes | $26.1\text{ ns}$ | $4.35\text{ ns}$ | **$3.62\text{ ns}$** | **$7.2\times$** | $131.7\text{ GiB/s}$ |
| **$D = 128$** | SIFT1M / Audio Features | $59.0\text{ ns}$ | $7.64\text{ ns}$ | **$6.72\text{ ns}$** | **$8.8\times$** | $141.9\text{ GiB/s}$ |
| **$D = 256$** | Compact Dense Representations | $148.0\text{ ns}$ | $15.6\text{ ns}$ | **$9.68\text{ ns}$** | **$15.3\times$** | **$197.1\text{ GiB/s}$** |
| **$D = 512$** | Small Language Embeddings | $332.0\text{ ns}$ | $40.2\text{ ns}$ | **$21.7\text{ ns}$** | **$15.4\times$** | $175.9\text{ GiB/s}$ |
| **$D = 768$** | BERT / `all-mpnet-base-v2` | $515.0\text{ ns}$ | $60.0\text{ ns}$ | **$32.5\text{ ns}$** | **$15.9\times$** | $176.2\text{ GiB/s}$ |
| **$D = 1024$** | BGE-Large / Large Text Embeddings | $699.0\text{ ns}$ | $84.3\text{ ns}$ | **$43.9\text{ ns}$** | **$16.0\times$** | $173.7\text{ GiB/s}$ |
| **$D = 1536$** | OpenAI `text-embedding-3-small/large` | $1071.0\text{ ns}$ | $147.0\text{ ns}$ | **$67.5\text{ ns}$** | **$15.9\times$** | $169.4\text{ GiB/s}$ |

#### Hardware PMU Performance Counters (`perf stat`, $D = 128$)

| Kernel | Dimension ($D$) | Latency (ns) | IPC | L1D Miss Rate | Branch Miss Rate | Throughput (GiB/s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scalar `l2_squared`** | 128 | 59.0 ns | 2.18 | 0.007% | 0.003% | 16.16 GiB/s |
| **AVX2 Single `l2_squared`** | 128 | 7.64 ns | 1.85 | 0.005% | 0.001% | 124.80 GiB/s |
| **AVX2 Unroll-4 `l2_squared`** | 128 | **6.72 ns** | **3.42** | **0.004%** | **0.001%** | **141.94 GiB/s** |

## Roadmap

- [x] Scalar Baseline Distance Kernels (L2, IP, Cosine, Fast Reciprocal Cosine)
- [x] Hardware Floating-Point State Control (FTZ/DAZ)
- [x] AVX2 + FMA Single-Accumulator Distance Kernel
- [x] AVX2 Multi-Accumulator ILP Unrolling (4-way register parallelism)
- [ ] AVX-512 Distance Kernels with portable runtime dispatch
- [ ] Cache-aware memory layout & blocked matrix scans
- [ ] Multithreaded concurrent query engine
- [ ] HNSW graph indexing for sub-millisecond approximate nearest neighbor search


## License

MIT License. Copyright (c) 2026 Adheeb Ahmed.