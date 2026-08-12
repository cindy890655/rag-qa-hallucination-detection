"""
Baseline versus fusion, as a chart.

The numbers are computed from cached_scores.json rather than typed in, so the
figure cannot drift away from the data the way a hand-copied table does. Paths
are resolved from this file's location, so the script runs from any directory.

The point of the chart is the shape of the trade, not the accuracy: the fusion
rule fires on contradiction OR low entailment, so it is a strict superset of the
baseline and cannot lose a detection. It buys recall and pays in precision.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cached_scores.json"
FIGURES = Path(__file__).resolve().parent / "figures"

CONTRA_T = 0.5
ENTAIL_T = 0.3

ORANGE = "#E8720C"
BLUE = "#3b78c2"


def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"Missing {CACHE}. Run detection/rescore.py first.")

    cache = json.load(CACHE.open(encoding="utf-8"))
    y_true = cache["y_true"]
    scores = cache["all_scores"]

    baseline_pred = [1 if s["contradiction"] >= CONTRA_T else 0 for s in scores]
    fusion_pred = [
        1 if (s["contradiction"] >= CONTRA_T or s["entailment"] < ENTAIL_T)
        else 0
        for s in scores
    ]

    def metrics(prediction):
        return [
            accuracy_score(y_true, prediction),
            precision_score(y_true, prediction, zero_division=0),
            recall_score(y_true, prediction, zero_division=0),
            f1_score(y_true, prediction, zero_division=0),
        ]

    baseline = metrics(baseline_pred)
    fusion = metrics(fusion_pred)

    # what the fusion rule actually trades, counted rather than described
    recovered = sum(1 for b, f, t in zip(baseline_pred, fusion_pred, y_true)
                    if t == 1 and b == 0 and f == 1)
    extra_alarms = (
        sum(1 for f, t in zip(fusion_pred, y_true) if t == 0 and f == 1)
        - sum(1 for b, t in zip(baseline_pred, y_true) if t == 0 and b == 1)
    )

    names = ["Accuracy", "Precision", "Recall", "F1"]
    print(f"samples: {len(y_true)}\n")
    print(f"{'metric':<12}{'baseline':<12}{'fusion':<12}{'change'}")
    print("-" * 48)
    for name, before, after in zip(names, baseline, fusion):
        print(f"{name:<12}{before:<12.3f}{after:<12.3f}{after - before:+.3f}")
    print("-" * 48)
    print(f"fusion recovers {recovered} hallucinations the baseline missed, "
          f"at the cost of {extra_alarms} additional false alarms")

    x = np.arange(len(names))
    width = 0.35

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars_baseline = axis.bar(x - width / 2, baseline, width,
                            label="Baseline (contradiction only)", color=BLUE)
    bars_fusion = axis.bar(x + width / 2, fusion, width,
                           label="Fusion (contradiction + entailment)",
                           color=ORANGE)

    axis.bar_label(bars_baseline, fmt="%.3f", padding=3, fontsize=9)
    axis.bar_label(bars_fusion, fmt="%.3f", padding=3, fontsize=9)

    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.set_title(f"Baseline vs fusion on {len(y_true)} HaluEval samples\n"
                   f"{recovered} hallucinations recovered for "
                   f"{extra_alarms} extra false alarms",
                   fontsize=13, fontweight="bold")
    axis.set_xticks(x)
    axis.set_xticklabels(names)
    axis.legend(loc="upper right", fontsize=10)
    axis.grid(axis="y", alpha=0.3)

    # mark the metric the change was made for
    recall_gain = fusion[2] - baseline[2]
    axis.annotate(f"{recall_gain:+.3f}",
                  xy=(2 + width / 2, fusion[2]),
                  xytext=(2 + width / 2, fusion[2] + 0.15),
                  ha="center", color=ORANGE, fontweight="bold", fontsize=11,
                  arrowprops=dict(arrowstyle="->", color=ORANGE))

    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "baseline_vs_fusion.png"
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()