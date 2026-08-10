"""
Compare four hallucination detection methods:
  1. NLI single-signal (contradiction only)
  2. NLI fusion (contradiction + entailment)
  3. LLM-as-judge, weak (Gemini Flash-Lite)
  4. LLM-as-judge, strong (Llama 3.3 70B)
Produces a comparison table (printed) and a grouped bar chart.
"""
import matplotlib.pyplot as plt
import numpy as np

# results: NLI on 2000 samples; LLM judges on ~450-500 subset (API free-tier limits)
methods = ['NLI single', 'NLI fusion', 'Flash-Lite\n(weak LLM)', 'Llama-70B\n(strong LLM)']
n_samples = [2000, 2000, 496, 446]
accuracy  = [0.741, 0.771, 0.508, 0.771]
precision = [0.910, 0.851, 0.505, 0.771]
recall    = [0.535, 0.656, 0.798, 0.771]
f1        = [0.674, 0.741, 0.619, 0.771]

# print comparison table
print("=" * 80)
print(f"{'Method':<22}{'N':<8}{'Acc':<8}{'Prec':<8}{'Rec':<8}{'F1':<8}")
print("-" * 80)
labels = ['NLI single', 'NLI fusion', 'Flash-Lite (weak)', 'Llama-70B (strong)']
for i in range(4):
    print(f"{labels[i]:<22}{n_samples[i]:<8}{accuracy[i]:<8.3f}"
          f"{precision[i]:<8.3f}{recall[i]:<8.3f}{f1[i]:<8.3f}")
print("=" * 80)

# grouped bar chart
x = np.arange(len(methods))
width = 0.2
fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#3b78c2", "#E8720C", "#4BA9A6", "#9c5fb5"]

for k, (vals, lab) in enumerate([(accuracy, 'Accuracy'), (precision, 'Precision'),
                                  (recall, 'Recall'), (f1, 'F1')]):
    b = ax.bar(x + (k - 1.5) * width, vals, width, label=lab, color=colors[k])
    ax.bar_label(b, fmt='%.2f', padding=2, fontsize=7)

ax.set_ylabel('Score')
ax.set_title('Hallucination Detection: Method Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right', ncol=4, fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.4, linewidth=1)

plt.tight_layout()
plt.savefig('figures/method_comparison.png', dpi=200, bbox_inches='tight')
print("saved figures/method_comparison.png")