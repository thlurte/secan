#include <benchmark/benchmark.h>
#include "secan/search/distance.h"
#include "secan/utils/utils.h"
#include <vector>

static void BM_L2_Squared_Scalar(benchmark::State& state) {
    int dim = state.range(0);

    // Generate random test vectors
    auto [a, shape_a] = generate(-1.0f, 1.0f, {dim});
    auto [b, shape_b] = generate(-1.0f, 1.0f, {dim});

    for (auto _ : state) {
        float dist = l2_squared(a.data(), b.data(), dim);
        benchmark::DoNotOptimize(dist);
        benchmark::ClobberMemory();
    }

    // Reports memory bandwidth throughput in GB/s
    state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

// Benchmark across standard AI embedding dimensions
BENCHMARK(BM_L2_Squared_Scalar)->Arg(64)->Arg(128)->Arg(768)->Arg(1536);

BENCHMARK_MAIN();
