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

### Baseline vs SIMD Vectorized Distance Kernels (AVX2 & AVX-512)
*Benchmarked on AMD Zen 4 Hawk Point 12-Core @ 4.30 GHz (L1D 32 KiB, L2 1 MiB, L3 16 MiB). Compiler: Release `-O3 -mavx2 -mfma -mavx512f -mavx512dq -mavx512bw -mavx512vl -DNDEBUG`.*

#### $L_2^2$ Vector Distance Dimensional Scaling Sweep ($D \in [64, 1536]$)

| Vector Dimension ($D$) | Workload / Embedding Model | Scalar Baseline | AVX2 Single (1-acc) | AVX2 Unroll-4 (4-acc) | **AVX-512 Dual (2-acc)** | **Max Speedup** | **Peak Bandwidth** |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$D = 64$** | Micro Embeddings / Image Hashes | $68.6\text{ ns}$ | $4.04\text{ ns}$ | $3.37\text{ ns}$ | **$3.17\text{ ns}$** | **$21.6\times$** | $151.0\text{ GiB/s}$ |
| **$D = 128$** | SIFT1M / Audio Features | $156.0\text{ ns}$ | $6.99\text{ ns}$ | $5.12\text{ ns}$ | **$4.64\text{ ns}$** | **$33.6\times$** | **$206.2\text{ GiB/s}$** |
| **$D = 256$** | Compact Dense Representations | $424.0\text{ ns}$ | $15.8\text{ ns}$ | **$10.7\text{ ns}$** | $11.3\text{ ns}$ | **$39.6\times$** | $178.8\text{ GiB/s}$ |
| **$D = 512$** | Small Language Embeddings | $947.0\text{ ns}$ | $40.0\text{ ns}$ | **$21.4\text{ ns}$** | $22.5\text{ ns}$ | **$44.3\times$** | $180.3\text{ GiB/s}$ |
| **$D = 768$** | BERT / `all-mpnet-base-v2` | $1600.0\text{ ns}$ | $58.8\text{ ns}$ | **$31.8\text{ ns}$** | $33.3\text{ ns}$ | **$50.3\times$** | $181.0\text{ GiB/s}$ |
| **$D = 1024$** | BGE-Large / Large Text Embeddings | $1826.0\text{ ns}$ | $82.8\text{ ns}$ | **$42.8\text{ ns}$** | $43.1\text{ ns}$ | **$42.7\times$** | $179.6\text{ GiB/s}$ |
| **$D = 1536$** | OpenAI `text-embedding-3-small/large` | $2749.0\text{ ns}$ | $143.0\text{ ns}$ | $65.5\text{ ns}$ | **$64.8\text{ ns}$** | **$42.4\times$** | $177.5\text{ GiB/s}$ |

#### Inner Product (Dot Product) AVX-512 vs AVX2 Sweep ($D \in [64, 1536]$)

| Vector Dimension ($D$) | Scalar Baseline | AVX2 Unroll-4 | **AVX-512 Dual (2-acc)** | **Max Speedup** | **Peak Bandwidth** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **$D = 64$** | $43.6\text{ ns}$ | $3.58\text{ ns}$ | **$2.98\text{ ns}$** | **$14.6\times$** | $160.7\text{ GiB/s}$ |
| **$D = 128$** | $70.1\text{ ns}$ | $4.75\text{ ns}$ | **$4.48\text{ ns}$** | **$15.6\times$** | **$213.5\text{ GiB/s}$** |
| **$D = 256$** | $142.0\text{ ns}$ | **$10.7\text{ ns}$** | $11.0\text{ ns}$ | **$13.3\times$** | $179.5\text{ GiB/s}$ |
| **$D = 512$** | $326.0\text{ ns}$ | **$21.2\text{ ns}$** | $22.3\text{ ns}$ | **$15.4\times$** | $180.7\text{ GiB/s}$ |
| **$D = 768$** | $506.0\text{ ns}$ | **$31.8\text{ ns}$** | $32.8\text{ ns}$ | **$15.9\times$** | $180.3\text{ GiB/s}$ |
| **$D = 1024$** | $686.0\text{ ns}$ | **$43.0\text{ ns}$** | $43.3\text{ ns}$ | **$16.0\times$** | $178.0\text{ GiB/s}$ |
| **$D = 1536$** | $1055.0\text{ ns}$ | $65.5\text{ ns}$ | **$64.5\text{ ns}$** | **$16.4\times$** | $178.1\text{ GiB/s}$ |

#### Hardware PMU Performance Counters (`perf stat`, $D = 128$)

| Kernel | Dimension ($D$) | Latency (ns) | IPC | L1D Miss Rate | Branch Miss Rate | Throughput (GiB/s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scalar `l2_squared`** | 128 | 59.0 ns | 2.18 | 0.007% | 0.003% | 16.16 GiB/s |
| **AVX2 Single `l2_squared`** | 128 | 7.64 ns | 1.85 | 0.005% | 0.001% | 124.80 GiB/s |
| **AVX2 Unroll-4 `l2_squared`** | 128 | 5.12 ns | 3.42 | 0.004% | 0.001% | 187.97 GiB/s |
| **AVX-512 Dual `l2_squared`** | 128 | **4.64 ns** | **3.65** | **0.003%** | **0.001%** | **206.22 GiB/s** |

#### Fused 1-Pass AVX2 Cosine Distance Dimensional Sweep ($D \in [64, 1536]$)

*Concurrently computes $\sum a_i b_i$, $\sum a_i^2$, and $\sum b_i^2$ in a single SIMD pass, eliminating redundant memory round-trips and reducing cache line traffic by $66\%$.*

| Vector Dimension ($D$) | Workload / Embedding Model | Scalar Cosine | Fast Reciprocal Cosine | **Fused AVX2 Cosine** | **Speedup** | **Peak Bandwidth** |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **$D = 64$** | Micro Embeddings / Image Hashes | $131.0\text{ ns}$ | $99.4\text{ ns}$ | **$8.64\text{ ns}$** | **$15.2\times$** | $55.2\text{ GiB/s}$ |
| **$D = 128$** | SIFT1M / Audio Features | $277.0\text{ ns}$ | $128.0\text{ ns}$ | **$13.40\text{ ns}$** | **$20.7\times$** | **$71.3\text{ GiB/s}$** |
| **$D = 256$** | Compact Dense Representations | $574.0\text{ ns}$ | $182.0\text{ ns}$ | **$27.20\text{ ns}$** | **$21.1\times$** | $70.3\text{ GiB/s}$ |
| **$D = 512$** | Small Language Embeddings | $1151.0\text{ ns}$ | $359.0\text{ ns}$ | **$54.50\text{ ns}$** | **$21.1\times$** | $70.1\text{ GiB/s}$ |
| **$D = 768$** | BERT / `all-mpnet-base-v2` | $1730.0\text{ ns}$ | $540.0\text{ ns}$ | **$84.50\text{ ns}$** | **$20.5\times$** | $67.7\text{ GiB/s}$ |
| **$D = 1024$** | BGE-Large / Large Text Embeddings | $2310.0\text{ ns}$ | $723.0\text{ ns}$ | **$114.0\text{ ns}$** | **$20.3\times$** | $66.7\text{ GiB/s}$ |
| **$D = 1536$** | OpenAI `text-embedding-3-small/large` | $3295.0\text{ ns}$ | $1082.0\text{ ns}$ | **$175.0\text{ ns}$** | **$18.8\times$** | $65.4\text{ GiB/s}$ |

## Roadmap

- [x] Scalar Baseline Distance Kernels (L2, IP, Cosine, Fast Reciprocal Cosine)
- [x] Hardware Floating-Point State Control (FTZ/DAZ)
- [x] AVX2 + FMA Single-Accumulator Distance Kernel
- [x] AVX2 Multi-Accumulator ILP Unrolling (4-way register parallelism)
- [x] Fused 1-Pass AVX2 Cosine Distance Kernel (66% cache bus traffic reduction)
- [x] AVX-512 Distance Kernels (512-bit ZMM dual-accumulator unrolling)
- [ ] Cache-aware memory layout & blocked matrix scans
- [ ] Multithreaded concurrent query engine
- [ ] HNSW graph indexing for sub-millisecond approximate nearest neighbor search


## License

MIT License. Copyright (c) 2026 Adheeb Ahmed.