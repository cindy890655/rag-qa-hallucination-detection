"""
Figures for the 4b-2 results. Reads only JSON, loads no models.

    fig 1  threshold_calibration.png   why 0.3 had to be recalibrated
    fig 2  config_comparison.png       the per-configuration result
    fig 3  validity_checks.png          evidence the detector is not blind
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
SCORED_DIR = DETECTION_ROOT / "data" / "rag_eval" / "scored"
ANALYSIS_DIR = DETECTION_ROOT / "data" / "rag_eval" / "analysis"
FIGURE_DIR = DETECTION_ROOT / "detection" / "figures"

BLUE, ORANGE, TEAL, PURPLE = "#3b78c2", "#E8720C", "#4BA9A6", "#9c5fb5"
GREY = "#8a8a8a"

SHORT_NAME = {
    "phi3_k1_tok64": "Phi-3\nk=1",
    "phi3_k3_tok64": "Phi-3\nk=3",
    "phi3_k5_tok64": "Phi-3\nk=5",
    "flan-base_k3_tok64": "FLAN-base\nk=3",
    "flan-large_k3_tok64": "FLAN-large\nk=3",
    "phi3_k3_tok128": "Phi-3\nk=3, 128tok",
    "phi3_k3_tok64_bm25": "BM25\nk=3",
}


def load():
    summary = json.load((ANALYSIS_DIR / "summary.json").open(encoding="utf-8"))

    pooled = []
    for path in sorted(SCORED_DIR.glob("scored_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (not row["abstained"]
                        and row["best_entailment"] is not None
                        and row["control_best_entailment"] is not None):
                    pooled.append(row)

    return summary, pooled


def figure_calibration(summary, pooled):
    thresholds = np.arange(0.02, 1.0, 0.02)
    real = [np.mean([r["best_entailment"] < t for r in pooled])
            for t in thresholds]
    control = [np.mean([r["control_best_entailment"] < t for r in pooled])
               for t in thresholds]

    chosen = summary["threshold"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(thresholds, np.array(control) * 100, color=ORANGE, lw=2.5,
             label="mismatched evidence (unsupported by construction)")
    ax1.plot(thresholds, np.array(real) * 100, color=BLUE, lw=2.5,
             label="real retrieved evidence")
    ax1.axvline(0.3, color=GREY, ls="--", lw=1.5)
    ax1.axvline(chosen, color="red", ls=":", lw=2)
    ax1.annotate("threshold tuned on\nHaluEval (0.30):\ncatches only 5.5%",
                 xy=(0.3, 5.5), xytext=(0.05, 62), fontsize=9, color=GREY,
                 arrowprops=dict(arrowstyle="->", color=GREY))
    ax1.annotate(f"calibrated ({chosen:.2f}):\ncatches 90%",
                 xy=(chosen, 90), xytext=(0.55, 55), fontsize=9, color="red",
                 arrowprops=dict(arrowstyle="->", color="red"))
    ax1.set_xlabel("max-entailment threshold")
    ax1.set_ylabel("flagged as hallucinated (%)")
    ax1.set_title("Threshold calibration on the negative control",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.hist([r["best_entailment"] for r in pooled], bins=40, alpha=0.75,
             color=BLUE, label="real retrieved evidence")
    ax2.hist([r["control_best_entailment"] for r in pooled], bins=40,
             alpha=0.75, color=ORANGE, label="mismatched evidence")
    ax2.axvline(0.3, color=GREY, ls="--", lw=1.5, label="HaluEval threshold")
    ax2.axvline(chosen, color="red", ls=":", lw=2, label="calibrated")
    ax2.set_xlabel("max entailment over retrieved chunks")
    ax2.set_ylabel("answers")
    ax2.set_title("Why 0.30 fails: the mismatched pairs sit above it",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = FIGURE_DIR / "threshold_calibration.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def figure_configs(summary):
    metrics = summary["metrics"]
    names = [n for n in SHORT_NAME if n in metrics]

    grounded = [metrics[n]["grounded_answer_rate"] * 100 for n in names]
    grounded_err = [
        [(metrics[n]["grounded_answer_rate"] - metrics[n]["grounded_ci_low"]) * 100
         for n in names],
        [(metrics[n]["grounded_ci_high"] - metrics[n]["grounded_answer_rate"]) * 100
         for n in names],
    ]
    halluc = [metrics[n]["hallucination_rate"] * 100 for n in names]
    halluc_err = [
        [(metrics[n]["hallucination_rate"] - metrics[n]["hallucination_ci_low"]) * 100
         for n in names],
        [(metrics[n]["hallucination_ci_high"] - metrics[n]["hallucination_rate"]) * 100
         for n in names],
    ]
    abstain = [metrics[n]["abstention_rate"] * 100 for n in names]
    chunk_hit = [metrics[n]["chunk_hit_rate"] * 100 for n in names]

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    width = 0.38
    b1 = ax1.bar(x - width / 2, grounded, width, yerr=grounded_err, capsize=4,
                 color=TEAL, label="grounded answer rate (higher is better)")
    b2 = ax1.bar(x + width / 2, halluc, width, yerr=halluc_err, capsize=4,
                 color=ORANGE, label="hallucination rate (lower is better)")
    ax1.bar_label(b1, fmt="%.0f", padding=8, fontsize=8)
    ax1.bar_label(b2, fmt="%.0f", padding=8, fontsize=8)
    ax1.set_ylabel("percent")
    ax1.set_ylim(0, 100)
    ax1.set_title("Automatic evaluation of 6 RAG configurations "
                  "(200 questions each, 95% Wilson CI)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(x, chunk_hit, "o-", color=BLUE, lw=2, ms=7,
             label="retrieval: gold chunk found")
    ax2.plot(x, abstain, "s-", color=PURPLE, lw=2, ms=7,
             label="abstained (\"I don't know\")")
    for xi, (c, a) in enumerate(zip(chunk_hit, abstain)):
        ax2.annotate(f"{c:.0f}", (xi, c), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color=BLUE)
        ax2.annotate(f"{a:.0f}", (xi, a), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=PURPLE)
    ax2.set_ylabel("percent")
    ax2.set_ylim(0, 100)
    ax2.set_title("The mechanism: k=5 retrieves best yet abstains most, "
                  "because the 512-token prompt budget truncates each chunk",
                  fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([SHORT_NAME[n] for n in names], fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGURE_DIR / "config_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def figure_validity(summary):
    metrics = summary["metrics"]
    names = [n for n in SHORT_NAME if n in metrics]
    x = np.arange(len(names))
    width = 0.2

    series = [
        ("real evidence", "real_flag_rate", BLUE),
        ("mismatched evidence", "control_flag_rate", ORANGE),
        ("retrieval hit", "flag_when_retrieval_hit", TEAL),
        ("retrieval miss", "flag_when_retrieval_miss", PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for index, (label, key, colour) in enumerate(series):
        values = [(metrics[n][key] or 0) * 100 for n in names]
        bars = ax.bar(x + (index - 1.5) * width, values, width,
                      label=label, color=colour)
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7)

    ax.set_ylabel("flagged as hallucinated (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Validity of the detector, without any human labels:\n"
                 "mismatched evidence and retrieval misses must be flagged "
                 "far more often than real evidence and retrieval hits",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAME[n] for n in names], fontsize=9)
    ax.legend(fontsize=9, ncol=4, loc="upper center")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGURE_DIR / "validity_checks.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved {path.name}")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary, pooled = load()

    print(f"Plotting from {len(pooled)} scored answers...")
    figure_calibration(summary, pooled)
    figure_configs(summary)
    figure_validity(summary)
    print(f"\nFigures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()