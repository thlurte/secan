# 🥋 Daily C++ Systems Drills

Targeted, micro-drills designed to build instinctual C++ reflexes for memory layout, pointer arithmetic, cache line locality, and SIMD execution.

---

## Day 010 (Monday, 2026-09-14) — AVX2 Fundamentals & Hardware State

| Drill | Title | Key Concepts | Status | File |
|:---:|:---|:---|:---:|:---|
| **01** | MXCSR FTZ/DAZ Modes | Denormals, microcode traps, `_MM_SET_FLUSH_ZERO_MODE` | ✅ Completed | [`day010_drill_01_mxcsr_ftz_daz.cpp`](day010_drill_01_mxcsr_ftz_daz.cpp) |
| **02** | SIMD Load & FMA Operations | `_mm256_loadu_ps`, `_mm256_fmadd_ps`, 8-lane parallel math | ✅ Completed | [`day010_drill_02_simd_fma_load.cpp`](day010_drill_02_simd_fma_load.cpp) |
| **03** | AVX2 Horizontal Reduction | High/Low 128-bit extraction, horizontal add, extracting scalar | ✅ Completed | [`day010_drill_03_horizontal_reduction.cpp`](day010_drill_03_horizontal_reduction.cpp) |
| **04** | Complete Baseline AVX2 Kernel | Single accumulator L2 loop, remainder tail, unit tests | ✅ Completed | [`day010_drill_04_baseline_avx2_l2.cpp`](day010_drill_04_baseline_avx2_l2.cpp) |

---

## Day 011 (Tuesday, 2026-09-15) — Memory Strides, Accumulator Unrolling & Normalization

| Drill | Title | Key Concepts | Status | File |
|:---:|:---|:---|:---:|:---|
| **01** | Pointer Offset & Flat Indexing | Row-major layout, `(i * D + j)`, pointer arithmetic vs dereference | ✅ Completed | [`day011_drill_01_pointer_offset.cpp`](day011_drill_01_pointer_offset.cpp) |
| **02** | Dual Accumulator Unrolling | Breaking dependency chains, ILP, FMA port saturation | ✅ Completed | [`day011_drill_02_dual_accumulator.cpp`](day011_drill_02_dual_accumulator.cpp) |
| **03** | SIMD Chunk & Tail Loop | Processing in chunks of 8, remainder loops | ✅ Completed | [`day011_drill_03_chunk_tail_loop.cpp`](day011_drill_03_chunk_tail_loop.cpp) |
| **04** | In-Place L2 Normalization | Unit vector scaling, numerical epsilon guard | ✅ Completed | [`day011_drill_04_normalize_inplace.cpp`](day011_drill_04_normalize_inplace.cpp) |

---

## Day 012 (Wednesday, 2026-09-16) — Fused Memory Passes & Vector Scaling

| Drill | Title | Key Concepts | Status | File |
|:---:|:---|:---|:---:|:---|
| **01** | Fused 1-Pass Cosine Distance | Computing $\langle a,b\rangle, \|a\|^2, \|b\|^2$ simultaneously in 1 pass | ✅ Completed | [`day012_drill_01_fused_cosine.cpp`](day012_drill_01_fused_cosine.cpp) |
| **02** | Vector Broadcast Scaling | `_mm256_set1_ps`, `_mm256_mul_ps`, scaling 8 floats in 1 cycle | ⏳ Ready | [`day012_drill_02_broadcast_scaling.cpp`](day012_drill_02_broadcast_scaling.cpp) |
| **03** | Peak Single-Core GFLOPS | Calculating execution port saturation on Zen 4 Pipes 0 & 1 | ⏳ Ready | [`day012_drill_03_peak_gflops.cpp`](day012_drill_03_peak_gflops.cpp) |
| **04** | Fused 1-Pass vs 2-Pass Memory | Comparing memory bus load count ($2\times$ vs $4\times$ reads) | ⏳ Ready | [`day012_drill_04_pass_comparison.cpp`](day012_drill_04_pass_comparison.cpp) |
