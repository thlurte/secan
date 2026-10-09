import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

def setup_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=14, color='#4A5568')
    ax.set_xlabel(xlabel, fontsize=10.5, fontweight='bold', color='#4A5568')
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight='bold', color='#4A5568')
    ax.tick_params(colors='#4A5568', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle='--', alpha=0.35, color='#94A3B8')
    ax.patch.set_alpha(0.0)

# =========================================================================
# SUBPLOT 1: In-Place Neural Activations (GELU vs SwiGLU vs ReLU)
# =========================================================================
x = np.linspace(-3.5, 3.5, 300)

# Exact mathematical formulations implemented today in cennan
relu = np.maximum(0, x)
gelu = 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * (x**3))))
# SwiGLU with gate = x (self-gated SiLU)
swish = x * (1.0 / (1.0 + np.exp(-x)))

ax1.plot(x, gelu, lw=3.0, color='#00E676', label='GELU (tanh approx - ViT/BERT)')
ax1.plot(x, swish, lw=2.5, color='#00B0FF', label='SwiGLU / SiLU (LLaMA/Mistral)')
ax1.plot(x, relu, lw=2.0, color='#FF5370', linestyle='--', label='Standard ReLU (Dying Neus)')

ax1.axhline(0, color='#94A3B8', lw=1.0, linestyle=':')
ax1.axvline(0, color='#94A3B8', lw=1.0, linestyle=':')

ax1.annotate('Smooth Non-Zero Negative Gradient\n(Prevents Dying Neurons)',
             xy=(-1.5, -0.15), xytext=(-3.3, 1.2),
             fontsize=9.5, fontweight='bold', color='#00E676',
             arrowprops=dict(arrowstyle='->', color='#00E676', lw=1.8))

setup_axes(ax1, 'Neural Activations Implemented in cennan', 'Input Logit (x)', 'Activated Output f(x)')
ax1.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', loc='upper left')

# =========================================================================
# SUBPLOT 2: Dual Accumulator ILP vs 4-Cycle FMA Pipeline Latency
# =========================================================================
accumulators = ['1 Accumulator\n(Loop-Carried Stall)', '2 Accumulators\n(Drill 06)', '4 Accumulators\n(Port 0/1 Saturation)']
cpe = [4.0, 2.0, 1.0] # Cycles Per Element / Instruction stall latency
throughput_ipc = [1.2, 2.4, 3.8] # Instructions Per Cycle

x_bars = np.arange(len(accumulators))
w = 0.35

c_stall = '#FF5370'
c_ipc = '#00E676'

rects1 = ax2.bar(x_bars - w/2, cpe, w, label='FMA Stall Latency (Cycles/Iter)', color=c_stall, edgecolor='#FFFFFF', linewidth=1.2, alpha=0.9)
rects2 = ax2.bar(x_bars + w/2, throughput_ipc, w, label='CPU IPC (Retiring Efficiency)', color=c_ipc, edgecolor='#FFFFFF', linewidth=1.2, alpha=0.9)

ax2.set_xticks(x_bars)
ax2.set_xticklabels(accumulators, fontsize=9.5, fontweight='bold')
setup_axes(ax2, 'Instruction-Level Parallelism (ILP) FMA Latency Hiding', 'Accumulator Strategy', 'Cycles / IPC Metric')

ax2.annotate('4-Cycle Pipeline Stall!', xy=(0 - w/2, 4.0), xytext=(0 - w/2, 4.3),
             ha='center', fontsize=9.5, fontweight='bold', color=c_stall)

ax2.annotate('Stalls Eliminated\n(3.8 IPC Peak)', xy=(2 + w/2, 3.8), xytext=(1.5, 4.2),
             ha='center', fontsize=9.5, fontweight='bold', color=c_ipc,
             arrowprops=dict(arrowstyle='->', color=c_ipc, lw=1.8))

ax2.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', loc='upper right')

fig.patch.set_alpha(0.0)
plt.tight_layout()
plt.savefig('assets/day33_activations_and_ilp.png', dpi=300, transparent=True)
print("Saved assets/day33_activations_and_ilp.png successfully!")
