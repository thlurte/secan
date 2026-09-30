# secan command runner

default:
    @just --list

# Configure CMake build directory
config build_type="Release":
    cmake -B build -DCMAKE_BUILD_TYPE={{build_type}}

# Build the library, CLI, tests, and benchmarks
build j="$(nproc)":
    cmake --build build -j{{j}}

# Run all unit tests via ctest
test:
    ctest --test-dir build --output-on-failure

# Run all benchmarks
bench:
    @echo "Running distance microbenchmarks..."
    ./build/benchmarks/bench_distance
    @echo "Running exact scan cache sweeps..."
    ./build/benchmarks/bench_exact_scan
    @echo "Running SIFT index benchmarks..."
    ./build/benchmarks/bench_index_sift

# Run a specific benchmark (e.g. `just bench-sift`)
bench-sift:
    ./build/benchmarks/bench_index_sift

bench-exact:
    ./build/benchmarks/bench_exact_scan

bench-distance:
    ./build/benchmarks/bench_distance

bench-kdtree:
    ./build/benchmarks/bench_kd_tree

# Build and run the entire test suite in one command
check: build test

# Clean build artifacts
clean:
    rm -rf build
