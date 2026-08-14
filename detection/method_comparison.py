"""
Four-method comparison, computed from the cached scores rather than hardcoded.

Two tables are produced because they answer different questions.

Table 1 puts all four methods on the same samples. The LLM judges were run over a
prefix of the same sample list as the NLI detector, and each stopped when its free
tier ran out, at 496 and 446 items. Comparing accuracies that were computed on
different subsets is not a comparison, so the common prefix is used for the
head-to-head.

Table 2 reports the NLI methods on all 2000 samples. This matters for the
single-versus-fusion comparison specifically: the 446-item prefix happens to
flatter the contradiction baseline, which scores about two points higher there
than on the full set, so using the subset would understate what the fusion rule
buys.

Paths are resolved from this file's location, so the script runs from anywhere.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cached_scores.json"
GEMINI = ROOT / "llm_judge" / "results" / "llm_judge_scores.json"
GROQ = ROOT / "llm_judge" / "results" / "llm_judge_groq_scores.json"
FIGURES = ROOT / "detection" / "figures"

CONTRA_T = 0.5
ENTAIL_T = 0.3

BLUE, ORANGE, TEAL, PURPLE = "#3b78c2", "#E8720C", "#4BA9A6", "#9c5fb5"


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.96
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    low, high = wilson(round(accuracy * len(y_true)), len(y_true))
    return {
        "n": len(y_true),
        "accuracy": accuracy,
        "ci": (low, high),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def mcnemar(y_true: list[int], a: list[int], b: list[int]) -> tuple[int, int, float]:
    """
    Exact McNemar on which method is correct. The methods judge the same samples,
    so a paired test is both valid and far more sensitive than comparing two
    independent proportions.
    """
    a_only = sum(1 for x, y, t in zip(a, b, y_true) if x == t and y != t)
    b_only = sum(1 for x, y, t in zip(a, b, y_true) if x != t and y == t)

    if a_only + b_only == 0:
        return (a_only, b_only, 1.0)

    return (a_only, b_only, binomtest(a_only, a_only + b_only, 0.5).pvalue)


def nli_predictions(scores: list[dict]) -> dict[str, list[int]]:
    return {
        "NLI single": [1 if s["contradiction"] >= CONTRA_T else 0 for s in scores],
        "NLI fusion": [
            1 if (s["contradiction"] >= CONTRA_T or s["entailment"] < ENTAIL_T)
            else 0
            for s in scores
        ],
    }


def show_table(title: str, results: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'Method':<16}{'N':<7}{'Accuracy':<22}"
          f"{'Prec':<9}{'Rec':<9}{'F1'}")
    print("-" * 78)
    for name, m in results.items():
        print(f"{name:<16}{m['n']:<7}"
              f"{m['accuracy']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]  "
              f"{m['precision']:<9.3f}{m['recall']:<9.3f}{m['f1']:.3f}")


def chart(results: dict[str, dict], common_n: int) -> None:
    names = list(results)
    x = np.arange(len(names))
    width = 0.2

    figure, axis = plt.subplots(figsize=(11, 6))

    series = [("Accuracy", "accuracy", BLUE), ("Precision", "precision", ORANGE),
              ("Recall", "recall", TEAL), ("F1", "f1", PURPLE)]

    for index, (label, field, colour) in enumerate(series):
        values = [results[name][field] for name in names]
        bars = axis.bar(x + (index - 1.5) * width, values, width,
                        label=label, color=colour)
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    # error bars on accuracy only, where the interval is the informative one
    accuracies = [results[name]["accuracy"] for name in names]
    lower = [results[name]["accuracy"] - results[name]["ci"][0] for name in names]
    upper = [results[name]["ci"][1] - results[name]["accuracy"] for name in names]
    axis.errorbar(x - 1.5 * width, accuracies, yerr=[lower, upper],
                  fmt="none", ecolor="black", capsize=3, linewidth=1)

    axis.axhline(y=0.5, color="red", linestyle=":", alpha=0.4, linewidth=1)
    axis.text(len(names) - 0.65, 0.51, "chance", fontsize=7, color="red",
              ha="right")

    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.0)
    axis.set_title(f"Hallucination detection: four methods on the same "
                   f"{common_n} HaluEval samples\n"
                   f"95% Wilson interval shown on accuracy",
                   fontsize=13, fontweight="bold")
    axis.set_xticks(x)
    axis.set_xticklabels(names, fontsize=10)
    axis.legend(loc="upper right", ncol=4, fontsize=9)
    axis.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "method_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {path}")


def main() -> None:
    for path in (CACHE, GEMINI, GROQ):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    cache = json.load(CACHE.open(encoding="utf-8"))
    gemini = json.load(GEMINI.open(encoding="utf-8"))
    groq = json.load(GROQ.open(encoding="utf-8"))

    y_full = cache["y_true"]
    scores_full = cache["all_scores"]

    common_n = min(len(gemini["y_true"]), len(groq["y_true"]))
    y_common = y_full[:common_n]

    # the head-to-head is only valid if the judges really scored the same items
    if not (y_common == gemini["y_true"][:common_n] == groq["y_true"][:common_n]):
        raise SystemExit(
            "The LLM judge files are not a prefix of the NLI sample list; "
            "the paired comparison below would be invalid."
        )

    nli_common = nli_predictions(scores_full[:common_n])

    paired = {
        "NLI single": metrics(y_common, nli_common["NLI single"]),
        "NLI fusion": metrics(y_common, nli_common["NLI fusion"]),
        "Flash-Lite": metrics(y_common, gemini["llm_pred"][:common_n]),
        "Llama-70B": metrics(y_common, groq["llm_pred"][:common_n]),
    }

    show_table(f"TABLE 1  all four methods on the same {common_n} samples "
               f"(fully paired)", paired)
    print(f"\nFlash-Lite ran to {len(gemini['y_true'])} and Llama-70B to "
          f"{len(groq['y_true'])} before their free tiers ran out; the "
          f"common prefix is {common_n}.")

    nli_full = nli_predictions(scores_full)
    full = {name: metrics(y_full, pred) for name, pred in nli_full.items()}
    show_table(f"TABLE 2  NLI methods on all {len(y_full)} samples", full)

    print("\n" + "=" * 78)
    print("IS THE SUBSET REPRESENTATIVE?")
    print("=" * 78)
    for name in nli_full:
        subset = paired[name]["accuracy"]
        overall = full[name]["accuracy"]
        flag = "" if abs(subset - overall) < 0.01 else "   <- differs"
        print(f"  {name:<12}on {common_n}: {subset:.4f}   "
              f"on {len(y_full)}: {overall:.4f}   "
              f"difference {abs(subset - overall):.4f}{flag}")
    print("\nThe single-versus-fusion comparison is therefore read off Table 2.")

    print("\n" + "=" * 78)
    print("PAIRED COMPARISONS (exact McNemar, on which method is correct)")
    print("=" * 78)
    print(f"{'comparison':<34}{'A only':<9}{'B only':<9}{'p':<12}{'verdict'}")
    print("-" * 78)

    comparisons = [
        ("NLI single", "NLI fusion", y_full, nli_full["NLI single"],
         nli_full["NLI fusion"], len(y_full)),
        ("NLI fusion", "Llama-70B", y_common, nli_common["NLI fusion"],
         groq["llm_pred"][:common_n], common_n),
        ("NLI fusion", "Flash-Lite", y_common, nli_common["NLI fusion"],
         gemini["llm_pred"][:common_n], common_n),
    ]

    for left, right, y, a, b, n in comparisons:
        a_only, b_only, p = mcnemar(y, a, b)
        verdict = "significant" if p < 0.05 else "not distinguishable"
        label = f"{left} vs {right} (n={n})"
        print(f"{label:<34}{a_only:<9}{b_only:<9}{p:<12.4g}{verdict}")

    # what the fusion rule actually trades, on the full set
    single = nli_full["NLI single"]
    fusion = nli_full["NLI fusion"]
    recovered = sum(1 for s, f, t in zip(single, fusion, y_full)
                    if t == 1 and s == 0 and f == 1)
    lost = sum(1 for s, f, t in zip(single, fusion, y_full)
               if t == 1 and s == 1 and f == 0)
    extra_alarms = (sum(1 for f, t in zip(fusion, y_full) if t == 0 and f == 1)
                    - sum(1 for s, t in zip(single, y_full) if t == 0 and s == 1))

    print("\n" + "=" * 78)
    print(f"WHAT THE FUSION RULE TRADES (all {len(y_full)} samples)")
    print("=" * 78)
    print(f"  hallucinations recovered from the baseline : {recovered}")
    print(f"  hallucinations lost                        : {lost}")
    print(f"  additional false alarms                    : {extra_alarms}")
    print(f"  recall  {full['NLI single']['recall']:.3f} -> "
          f"{full['NLI fusion']['recall']:.3f}")
    print(f"  precision {full['NLI single']['precision']:.3f} -> "
          f"{full['NLI fusion']['precision']:.3f}")
    print("\nThe fusion rule is a strict superset of the baseline (it fires on")
    print("contradiction OR low entailment), so it cannot lose a detection; the")
    print("entire cost appears as false alarms.")

    chart(paired, common_n)


if __name__ == "__main__":
    main()