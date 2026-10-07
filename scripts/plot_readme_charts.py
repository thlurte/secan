import matplotlib.pyplot as plt
import numpy as np

# Configure global matplotlib parameters for high-contrast transparent theme
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11

def setup_transparent_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14, color='#4A5568')
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold', color='#4A5568')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='#4A5568')
    ax.tick_params(colors='#4A5568', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle='--', alpha=0.35, color='#94A3B8')
    ax.patch.set_alpha(0.0)

# =========================================================================
# CHART 1: Cache Hierarchy Scaling & DRAM Eviction Cliff
# =========================================================================
fig, ax1 = plt.subplots(figsize=(11, 5.2))

n_labels = ['100\n(51KB)', '1k\n(512KB)', '10k\n(5.1MB)', '100k\n(51.2MB)', '1M\n(512MB)']
x = np.arange(len(n_labels))
avx512_bw = [40.56, 58.65, 67.63, 14.82, 14.28]
avx2_bw   = [30.60, 25.28, 24.04, 13.75, 13.47]
scalar_bw = [3.10, 2.72, 2.70, 2.70, 3.07]

ax1.plot(x, avx512_bw, marker='o', lw=2.5, markersize=8, color='#00E676', label='AVX-512 Dual (512-bit)')
ax1.plot(x, avx2_bw, marker='s', lw=2.5, markersize=7, color='#00B0FF', label='AVX2 Unroll-4 (256-bit)')
ax1.plot(x, scalar_bw, marker='^', lw=2.0, markersize=7, color='#FF5252', label='Scalar Baseline', linestyle=':')

# Cache boundary annotations
ax1.axvspan(0, 2, alpha=0.10, color='#00E676', label='L1/L2/L3 Cache Resident (<16MB)')
ax1.axvspan(2, 4, alpha=0.10, color='#FF5252', label='DRAM Spilled (>16MB)')

ax1.annotate('4.7x DRAM Cliff\n(67.6 -> 14.8 GiB/s)', xy=(3, 14.82), xytext=(2.6, 38),
             fontsize=10, fontweight='bold', color='#FF1744',
             arrowprops=dict(arrowstyle='->', color='#FF1744', lw=2.0))

setup_transparent_axes(ax1, 'Exact Linear Scan Working Set Scaling & Cache Cliffs', 'Dataset Size N (Working Memory)', 'Throughput (GiB/s)')
ax1.set_xticks(x)
ax1.set_xticklabels(n_labels)
ax1.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', loc='upper right')

fig.patch.set_alpha(0.0)
plt.tight_layout()
plt.savefig('assets/cache_hierarchy_scaling.png', dpi=300, transparent=True)
plt.close()
print("Saved assets/cache_hierarchy_scaling.png")

# =========================================================================
# CHART 2: Kernel Latency & Scaling across Dimensions (D = 64..1536)
# =========================================================================
fig, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(14, 5.2))

dims = [64, 128, 256, 512, 768, 1024, 1536]
scalar_l2 = [68.6, 156.0, 424.0, 947.0, 1600.0, 1826.0, 2749.0]
avx2_l2   = [3.37, 5.12, 10.7, 21.4, 31.8, 42.8, 65.5]
avx512_l2 = [3.17, 4.64, 11.3, 22.5, 33.3, 43.1, 64.8]

# Latency plot
ax2_1.plot(dims, scalar_l2, marker='^', lw=2, markersize=6, color='#FF5252', label='Scalar Baseline', linestyle='--')
ax2_1.plot(dims, avx2_l2, marker='s', lw=2.5, markersize=6, color='#00B0FF', label='AVX2 Unroll-4')
ax2_1.plot(dims, avx512_l2, marker='o', lw=2.5, markersize=6, color='#00E676', label='AVX-512 Dual (2-acc)')
setup_transparent_axes(ax2_1, 'L2 Squared Distance Latency vs Dimension', 'Vector Dimension D', 'Latency (ns)')
ax2_1.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1')

# Speedup plot
speedup_avx2 = [s/a for s, a in zip(scalar_l2, avx2_l2)]
speedup_avx512 = [s/a for s, a in zip(scalar_l2, avx512_l2)]

ax2_2.plot(dims, speedup_avx512, marker='o', lw=2.5, markersize=7, color='#00E676', label='AVX-512 Speedup')
ax2_2.plot(dims, speedup_avx2, marker='s', lw=2.5, markersize=7, color='#00B0FF', label='AVX2 Unroll-4 Speedup')
setup_transparent_axes(ax2_2, 'SIMD Acceleration Speedup vs Scalar', 'Vector Dimension D', 'Speedup Multiplier (x)')
ax2_2.axhline(50, color='#94A3B8', linestyle=':', alpha=0.6)
ax2_2.annotate('Up to 50.3x Acceleration!', xy=(768, 50.3), xytext=(600, 32),
               fontsize=10, fontweight='bold', color='#00E676',
               arrowprops=dict(arrowstyle='->', color='#00E676', lw=2.0))
ax2_2.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1')

fig.patch.set_alpha(0.0)
plt.tight_layout()
plt.savefig('assets/kernel_dimension_scaling.png', dpi=300, transparent=True)
plt.close()
print("Saved assets/kernel_dimension_scaling.png")

# =========================================================================
# CHART 3: Index Pareto Frontier (Recall@10 vs Throughput QPS)
# =========================================================================
fig, ax3 = plt.subplots(figsize=(10, 5.5))

# SIFT-100K Index data
# IvfFlat
ivf_recall = [48.7, 79.4, 92.9, 97.9, 100.0]
ivf_qps    = [80475, 24506, 12900, 5884, 1740]

# Randomized KD-Tree
kd_recall = [7.3, 14.2, 16.7]
kd_qps    = [218268, 19048, 5710]

# Flat Oracle
flat_recall = [100.0, 100.0]
flat_qps    = [378, 1209]

ax3.plot(ivf_recall, ivf_qps, marker='o', lw=2.8, markersize=9, color='#00E676', label='IvfFlatIndex (AVX2 Multi-Probe)')
ax3.plot(kd_recall, kd_qps, marker='^', lw=2.0, markersize=8, color='#FF5252', label='RandomizedKdTree (FLANN Baseline)', linestyle='--')
ax3.scatter([100], [1209], color='#00B0FF', s=140, zorder=5, label='Flat2D (Batch-Tiled GEMM)')
ax3.scatter([100], [378], color='#FFB300', s=100, zorder=5, label='Flat2D (Single Query Exact)')

ax3.set_yscale('log')
setup_transparent_axes(ax3, 'SIFT-100K Recall@10 vs Query Throughput (QPS)', 'Recall@10 (%)', 'Throughput QPS (Log Scale)')

# Annotations for sweet spot
ax3.annotate('Optimal Operating Point\n(nprobe=8: 92.9% Recall @ 12.9k QPS)',
             xy=(92.9, 12900), xytext=(65, 3000),
             fontsize=10, fontweight='bold', color='#00E676',
             arrowprops=dict(arrowstyle='->', color='#00E676', lw=2.0))

ax3.annotate('Curse of Dimensionality:\nKD-Tree Recall Collapses (<17%)',
             xy=(16.7, 5710), xytext=(22, 1000),
             fontsize=9, fontweight='semibold', color='#FF5252',
             arrowprops=dict(arrowstyle='->', color='#FF5252', lw=1.5))

ax3.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', loc='upper right')

fig.patch.set_alpha(0.0)
plt.tight_layout()
plt.savefig('assets/index_pareto_frontier.png', dpi=300, transparent=True)
plt.close()
print("Saved assets/index_pareto_frontier.png")
