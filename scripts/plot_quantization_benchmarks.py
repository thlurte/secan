#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11

# Benchmark Data on 100,000 vectors (128-D)
categories = ['FP32 Baseline\n(Uncompressed)', 'SQ8 Quantized\n(4x Compression)', 'SQ4 Two-Stage\n(8x Compression + Re-rank)']
latencies = [2914.0, 470.0, 7819.0 / 100.0]  # Normalized per query (μs)
throughputs = [34.68, 213.67, 128.27]         # Million vectors / sec
memory_mb = [51.2, 12.8, 6.4]                 # Memory footprint (MB)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

c_fp32 = '#FF5370'  # Coral Pink
c_sq8 = '#00E676'   # Neon Emerald
c_sq4 = '#00B0FF'   # Electric Sky Blue
colors = [c_fp32, c_sq8, c_sq4]

# Subplot 1: Scan Throughput
bars1 = ax1.bar(range(len(categories)), throughputs, color=colors, edgecolor='#FFFFFF', linewidth=1.5, width=0.55, alpha=0.90)
ax1.set_title('Vector Scan Throughput (Higher is Better)', fontsize=13, fontweight='bold', pad=15, color='#1A202C')
ax1.set_ylabel('Throughput (Million Vecs / sec)', fontsize=11, fontweight='bold', color='#1A202C')
ax1.set_xticks(range(len(categories)))
ax1.set_xticklabels(categories, fontsize=10, fontweight='semibold')
ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')

ax1.annotate('6.16x Faster!\n(213.7M vecs/s)', xy=(1, throughputs[1]), xytext=(1, throughputs[1] + 15),
             ha='center', fontsize=10, fontweight='bold', color='#00C853')

# Subplot 2: Memory Footprint vs Cache Residency
bars2 = ax2.bar(range(len(categories)), memory_mb, color=colors, edgecolor='#FFFFFF', linewidth=1.5, width=0.55, alpha=0.90)
ax2.set_title('Index Memory Footprint (100k Vectors, Lower is Better)', fontsize=13, fontweight='bold', pad=15, color='#1A202C')
ax2.set_ylabel('Dataset Memory (MB)', fontsize=11, fontweight='bold', color='#1A202C')
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels(categories, fontsize=10, fontweight='semibold')
ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')

ax2.axhline(16.0, color='#D50000', linestyle=':', linewidth=1.8, label='16 MB L3 Cache Boundary')
ax2.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', loc='upper right')

for bar, mem in zip(bars2, memory_mb):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{mem:.1f} MB", ha='center', va='bottom', fontsize=10, fontweight='bold')

os.makedirs('assets', exist_ok=True)
plt.tight_layout()
plt.savefig('assets/quantization_benchmark.png', dpi=300)
print("Saved assets/quantization_benchmark.png successfully!")
