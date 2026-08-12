"""
Compute ROC-AUC and PR-AUC for the hallucination detector, using the cached NLI
scores. The contradiction probability is the continuous score; label 1 =
hallucinated, label 0 = faithful.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cached_scores.json"
FIGURES = Path(__file__).resolve().parent / "figures"

if not CACHE.exists():
    raise SystemExit(f"Missing {CACHE}. Run detection/rescore.py first.")

cache = json.load(CACHE.open(encoding="utf-8"))
all_scores = cache["all_scores"]
y_true = cache["y_true"]

# continuous score = contradiction probability (higher = more likely hallucinated)
y_score = [s["contradiction"] for s in all_scores]

roc_auc = roc_auc_score(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)

print("=" * 50)
print("Detector evaluation with AUC metrics")
print("=" * 50)
print(f"Samples:  {len(y_true)}")
print(f"ROC-AUC:  {roc_auc:.4f}")
print(f"PR-AUC:   {pr_auc:.4f}")
print("=" * 50)
print("ROC-AUC = ability to rank hallucinations above faithful answers")
print("PR-AUC  = precision-recall trade-off quality")

# --- draw ROC and PR curves side by side ---
fpr, tpr, _ = roc_curve(y_true, y_score)
prec, rec, _ = precision_recall_curve(y_true, y_score)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(fpr, tpr, color="#E8720C", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend(loc="lower right")

ax2.plot(rec, prec, color="#3b78c2", lw=2, label=f"PR (AUC={pr_auc:.3f})")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve")
ax2.legend(loc="lower left")

plt.tight_layout()

FIGURES.mkdir(parents=True, exist_ok=True)
output = FIGURES / "auc_curves.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
print(f"\nSaved {output}")