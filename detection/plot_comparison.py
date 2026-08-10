"""
Plot baseline vs fusion detector metrics as a grouped bar chart.
Saves the figure into the figures/ folder.
"""
import matplotlib.pyplot as plt
import numpy as np

ORANGE = "#E8720C"
BLUE = "#3b78c2"

# metrics from eval_fusion.py (2000 samples)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
baseline = [0.741, 0.910, 0.535, 0.674]
fusion   = [0.771, 0.851, 0.656, 0.741]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - width/2, baseline, width,
            label='Baseline (contradiction only)', color=BLUE)
b2 = ax.bar(x + width/2, fusion, width,
            label='Fusion (contradiction + entailment)', color=ORANGE)

ax.bar_label(b1, fmt='%.3f', padding=3, fontsize=9)
ax.bar_label(b2, fmt='%.3f', padding=3, fontsize=9)

ax.set_ylabel('Score')
ax.set_title('Baseline vs Fusion Detector (2000 samples)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# highlight the key recall improvement
ax.annotate('+0.12', xy=(2 + width/2, 0.656), xytext=(2 + width/2, 0.80),
            ha='center', color=ORANGE, fontweight='bold', fontsize=11,
            arrowprops=dict(arrowstyle='->', color=ORANGE))

plt.tight_layout()
plt.savefig('figures/baseline_vs_fusion.png', dpi=200, bbox_inches='tight')
print("saved figures/baseline_vs_fusion.png")