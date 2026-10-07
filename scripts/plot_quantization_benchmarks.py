import matplotlib.pyplot as plt
import numpy as np

# Set clean styling using pure matplotlib (supports transparent light & dark)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11

# Benchmark Data from AMD Ryzen 5 8645HS (Zen 4)
categories = ['N = 10,000 Vectors\n(L3 Cache Resident)', 'N = 100,000 Vectors\n(DRAM vs L3 Bound)']
fp32_latency = [77.0, 2424.0]     # microseconds
sq8_latency = [48.9, 523.0]       # microseconds

throughput_fp32 = [130.48, 41.91] # Millions of vectors / sec
throughput_sq8 = [205.64, 192.95] # Millions of vectors / sec

x = np.arange(len(categories))
width = 0.32

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

# Modern Vibrant Palette: Coral/Red for baseline, Neon Cyan/Green for SQ8
c_fp32 = '#FF5370'   # Coral Pink
c_sq8 = '#00E676'    # Neon Emerald

# ----------------- SUBPLOT 1: Latency -----------------
rects1 = ax1.bar(x - width/2, fp32_latency, width, label='FP32 AVX2', color=c_fp32, edgecolor='#FFFFFF', linewidth=1.2, alpha=0.90)
rects2 = ax1.bar(x + width/2, sq8_latency, width, label='SQ8 AVX2 (4x Memory Reduction)', color=c_sq8, edgecolor='#FFFFFF', linewidth=1.2, alpha=0.90)

ax1.set_title('Scan Latency (Lower is Better)', fontsize=13, fontweight='bold', pad=15, color='#4A5568')
ax1.set_ylabel('Query Latency (μs)', fontsize=11, fontweight='bold', color='#4A5568')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10, fontweight='semibold')
ax1.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')

# Annotate speedup
ax1.annotate('1.57x Faster', xy=(x[0]+width/2, sq8_latency[0]), xytext=(x[0]+width/2, sq8_latency[0]+220),
             ha='center', fontsize=10, fontweight='bold', color=c_sq8,
             arrowprops=dict(arrowstyle='->', color=c_sq8, lw=1.8))

ax1.annotate('4.64x FASTER!\n(523 μs vs 2,424 μs)', xy=(x[1]+width/2, sq8_latency[1]), xytext=(x[1]+width/2, sq8_latency[1]+650),
             ha='center', fontsize=11, fontweight='bold', color=c_sq8,
             arrowprops=dict(arrowstyle='->', color=c_sq8, lw=2.0))

# ----------------- SUBPLOT 2: Throughput -----------------
rects3 = ax2.bar(x - width/2, throughput_fp32, width, label='FP32 AVX2', color=c_fp32, edgecolor='#FFFFFF', linewidth=1.2, alpha=0.90)
rects4 = ax2.bar(x + width/2, throughput_sq8, width, label='SQ8 AVX2 (4x Memory Reduction)', color=c_sq8, edgecolor='#FFFFFF', linewidth=1.2, alpha=0.90)

ax2.set_title('Search Throughput (Higher is Better)', fontsize=13, fontweight='bold', pad=15, color='#4A5568')
ax2.set_ylabel('Throughput (Million Vecs / sec)', fontsize=11, fontweight='bold', color='#4A5568')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=10, fontweight='semibold')
ax2.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')

ax2.annotate('193M vecs/s\n(4.6x Throughput)', xy=(x[1]+width/2, throughput_sq8[1]), xytext=(x[1]+width/2, throughput_sq8[1]-45),
             ha='center', fontsize=10, fontweight='bold', color='#1A202C')

# Style axes spines & tick colors for Dark & Light transparency
for ax in (ax1, ax2):
    ax.tick_params(colors='#4A5568', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
        spine.set_linewidth(1.2)
    ax.patch.set_alpha(0.0)

fig.patch.set_alpha(0.0)
plt.tight_layout()
plt.savefig('assets/quantization_benchmark.png', dpi=300, transparent=True)
print("Saved assets/quantization_benchmark.png successfully!")
