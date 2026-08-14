"""
Turn the scored grid into the 4b-2 result tables.

The detection threshold is not hardcoded. The threshold that was tuned on
HaluEval (entailment < 0.3) does not transfer to this setting: RAG evidence is
several chunks, the support signal is the maximum entailment across them, and
mismatched-evidence pairs cluster around 0.42, with only 5.5% of them below 0.3.
Using 0.3 therefore flags almost nothing and reports a pooled hallucination rate
of 1.7%, under 1% on the k=3 configurations, which is an artefact.

Instead the threshold is calibrated on the negative control. Every answer was
also scored against the evidence retrieved for a different question; those pairs
are unsupported by construction, so they act as known positives. The threshold
is the lowest one at which the detector catches TARGET_CONTROL_RECALL of them.

Only entailment is used. The contradiction half of the fusion rule adds exactly
zero detections here: across the 1,680 answers the minimum contradiction crosses
0.5 only 8 times - six at k=1, where the minimum degenerates to a single chunk,
and two on flan-base at k=3 - and the entailment rule already flags all eight.
On the phi3 multi-chunk configurations it never exceeds 0.43. On HaluEval
contradiction was the dominant signal; on multi-chunk RAG evidence it is diluted
away and entailment carries the decision.

Outputs
-------
    data/rag_eval/analysis/config_metrics.csv
    data/rag_eval/analysis/summary.json
"""
import argparse
import csv
import json
import math
from pathlib import Path

from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
SCORED_DIR = DETECTION_ROOT / "data" / "rag_eval" / "scored"
ANALYSIS_DIR = DETECTION_ROOT / "data" / "rag_eval" / "analysis"

TARGET_CONTROL_RECALL = 0.90
THRESHOLD_SWEEP = [round(0.02 * i, 2) for i in range(1, 50)]

# Presentation order: the top_k axis first, then the generator axis.
CONFIG_ORDER = [
    "phi3_k1_tok64",
    "phi3_k3_tok64",
    "phi3_k5_tok64",
    "flan-base_k3_tok64",
    "flan-large_k3_tok64",
    "phi3_k3_tok128",
    "phi3_k3_tok64_bm25",
]


def wilson(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson score interval, which behaves sensibly near 0 and 1."""
    if total == 0:
        return (0.0, 0.0)

    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator

    return (max(0.0, centre - spread), min(1.0, centre + spread))


def load_scored() -> dict[str, list[dict]]:
    configs = {}

    for path in sorted(SCORED_DIR.glob("scored_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        configs[rows[0]["config_id"]] = rows

    if not configs:
        raise SystemExit(f"No scored files in {SCORED_DIR}. Run score_grid.py.")

    return configs


def calibrate(configs: dict[str, list[dict]]) -> float:
    """
    Pick the lowest threshold that catches TARGET_CONTROL_RECALL of the
    known-unsupported control pairs, pooled over every configuration.
    """
    pooled = [
        row
        for rows in configs.values()
        for row in rows
        if not row["abstained"]
        and row["best_entailment"] is not None
        and row["control_best_entailment"] is not None
    ]

    print("=" * 78)
    print("THRESHOLD CALIBRATION on the negative control")
    print("=" * 78)
    print(f"{'threshold':<14}{'control caught':<20}{'real flagged'}")
    print("-" * 78)

    chosen = None

    for threshold in THRESHOLD_SWEEP:
        control = sum(
            row["control_best_entailment"] < threshold for row in pooled
        ) / len(pooled)
        real = sum(
            row["best_entailment"] < threshold for row in pooled
        ) / len(pooled)

        if threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            marker = "  <- HaluEval value" if threshold == 0.3 else ""
            print(f"{threshold:<14.2f}{control:<20.1%}{real:.1%}{marker}")

        if chosen is None and control >= TARGET_CONTROL_RECALL:
            chosen = threshold

    if chosen is None:
        raise SystemExit("No threshold reaches the target control recall")

    print("-" * 78)
    print(f"chosen threshold: max-entailment < {chosen:.2f} "
          f"(catches >= {TARGET_CONTROL_RECALL:.0%} of control pairs)")
    print(f"pooled answered responses: {len(pooled)}")

    return chosen


def config_metrics(rows: list[dict], threshold: float) -> dict:
    flagged = lambda row: row["best_entailment"] < threshold

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    answered = [
        row for row in answerable
        if not row["abstained"] and row["best_entailment"] is not None
    ]
    hallucinated = sum(flagged(row) for row in answered)
    grounded = len(answered) - hallucinated

    hit = [row for row in answered if row["chunk_hit"]]
    miss = [row for row in answered if not row["chunk_hit"]]

    controlled = [
        row for row in rows
        if not row["abstained"]
        and row["best_entailment"] is not None
        and row["control_best_entailment"] is not None
    ]

    una_answered = [
        row for row in unanswerable
        if not row["abstained"] and row["best_entailment"] is not None
    ]

    halluc_lo, halluc_hi = wilson(hallucinated, len(answered))
    grounded_lo, grounded_hi = wilson(grounded, len(answerable))

    return {
        "n_answerable": len(answerable),
        "abstention_rate": sum(r["abstained"] for r in answerable) / len(answerable),
        "n_answered": len(answered),
        "hallucination_rate": hallucinated / len(answered),
        "hallucination_ci_low": halluc_lo,
        "hallucination_ci_high": halluc_hi,
        "grounded_answer_rate": grounded / len(answerable),
        "grounded_ci_low": grounded_lo,
        "grounded_ci_high": grounded_hi,
        "chunk_hit_rate": sum(bool(r["chunk_hit"]) for r in answerable) / len(answerable),
        "flag_when_retrieval_hit": (sum(flagged(r) for r in hit) / len(hit)
                                    if hit else None),
        "flag_when_retrieval_miss": (sum(flagged(r) for r in miss) / len(miss)
                                     if miss else None),
        "control_flag_rate": (sum(r["control_best_entailment"] < threshold
                                  for r in controlled) / len(controlled)),
        "real_flag_rate": sum(flagged(r) for r in controlled) / len(controlled),
        "unanswerable_abstention_rate": (
            sum(r["abstained"] for r in unanswerable) / len(unanswerable)),
        "unanswerable_flag_rate": (sum(flagged(r) for r in una_answered)
                                   / len(una_answered) if una_answered else None),
    }


def grounded_vector(rows: list[dict], threshold: float) -> dict[str, bool]:
    """Per-question outcome, so configurations can be compared pairwise."""
    return {
        row["id"]: (not row["abstained"]
                    and row["best_entailment"] is not None
                    and row["best_entailment"] >= threshold)
        for row in rows if row["answerable"]
    }


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """
    Exact McNemar test on paired outcomes. The configurations answer the same
    questions, so a paired test is both valid and far more sensitive than
    comparing two independent proportions.
    """
    shared = set(a) & set(b)
    only_a = sum(1 for key in shared if a[key] and not b[key])
    only_b = sum(1 for key in shared if b[key] and not a[key])

    if only_a + only_b == 0:
        return (only_a, only_b, 1.0)

    return (only_a, only_b,
            binomtest(only_a, only_a + only_b, 0.5).pvalue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="use this threshold instead of calibrating. Pass the value "
             "already in use when a configuration is added later: re-deriving "
             "it would move the operating point for every configuration at "
             "once, leaving the comparison confounded with a change of "
             "measurement, and would invalidate the audit sample, which was "
             "drawn according to the verdicts at the old value.")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    configs = load_scored()

    if args.threshold is None:
        threshold = calibrate(configs)
    else:
        threshold = args.threshold
        print("=" * 78)
        print(f"THRESHOLD FIXED AT {threshold:.2f} (calibration skipped)")
        print("=" * 78)

    present = [c for c in CONFIG_ORDER if c in configs]
    present += [c for c in configs if c not in CONFIG_ORDER]

    metrics = {name: config_metrics(configs[name], threshold)
               for name in present}

    print("\n" + "=" * 104)
    print(f"MAIN RESULT   rule: max-entailment < {threshold:.2f}")
    print("=" * 104)
    print(f"{'config':<22}{'chunk hit':<12}{'abstain':<10}"
          f"{'hallucination rate':<26}{'grounded answer rate'}")
    print("-" * 104)

    for name in present:
        m = metrics[name]
        print(f"{name:<22}{m['chunk_hit_rate']:<12.1%}"
              f"{m['abstention_rate']:<10.1%}"
              f"{m['hallucination_rate']:.1%} "
              f"[{m['hallucination_ci_low']:.0%},"
              f"{m['hallucination_ci_high']:.0%}]{'':<9}"
              f"{m['grounded_answer_rate']:.1%} "
              f"[{m['grounded_ci_low']:.0%},{m['grounded_ci_high']:.0%}]")

    print("\n" + "=" * 104)
    print("VALIDITY CHECKS (no human labels needed)")
    print("=" * 104)
    print(f"{'config':<22}{'real evidence':<16}{'mismatched':<14}"
          f"{'retrieval hit':<16}{'retrieval miss':<16}{'unanswerable'}")
    print("-" * 104)

    for name in present:
        m = metrics[name]
        fmt = lambda v: "n/a" if v is None else f"{v:.1%}"
        print(f"{name:<22}{m['real_flag_rate']:<16.1%}"
              f"{m['control_flag_rate']:<14.1%}"
              f"{fmt(m['flag_when_retrieval_hit']):<16}"
              f"{fmt(m['flag_when_retrieval_miss']):<16}"
              f"{fmt(m['unanswerable_flag_rate'])}")

    print("\nReading these: the detector should flag mismatched evidence far more")
    print("often than real evidence, and retrieval misses more than hits.")

    print("\n" + "=" * 104)
    print("PAIRED COMPARISONS of grounded answer rate (exact McNemar)")
    print("=" * 104)

    vectors = {name: grounded_vector(configs[name], threshold)
               for name in present}

    pairs = [
        ("phi3_k1_tok64", "phi3_k3_tok64"),
        ("phi3_k3_tok64", "phi3_k5_tok64"),
        ("phi3_k1_tok64", "phi3_k5_tok64"),
        ("flan-base_k3_tok64", "phi3_k3_tok64"),
        ("flan-base_k3_tok64", "flan-large_k3_tok64"),
        ("phi3_k3_tok64", "phi3_k3_tok128"),
        ("phi3_k3_tok64", "phi3_k3_tok64_bm25"),
    ]

    print(f"{'comparison':<46}{'A only':<9}{'B only':<9}{'p':<12}{'verdict'}")
    print("-" * 104)

    comparisons = []

    for left, right in pairs:
        if left not in vectors or right not in vectors:
            continue

        only_left, only_right, p = mcnemar(vectors[left], vectors[right])
        verdict = ("significant" if p < 0.05 else
                   "not distinguishable")
        label = f"{left}  vs  {right}"
        print(f"{label:<46}{only_left:<9}{only_right:<9}{p:<12.4g}{verdict}")
        comparisons.append({"a": left, "b": right, "a_only": only_left,
                            "b_only": only_right, "p_value": p})

    csv_path = ANALYSIS_DIR / "config_metrics.csv"
    fields = ["config"] + list(next(iter(metrics.values())).keys())

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in present:
            writer.writerow({"config": name, **metrics[name]})

    summary_path = ANALYSIS_DIR / "summary.json"
    json.dump(
        {
            "threshold": threshold,
            "target_control_recall": TARGET_CONTROL_RECALL,
            "rule": "hallucinated when max entailment over retrieved chunks "
                    f"< {threshold:.2f}",
            "metrics": metrics,
            "paired_comparisons": comparisons,
        },
        summary_path.open("w", encoding="utf-8"),
        indent=2,
    )

    print(f"\nSaved {csv_path.name} and {summary_path.name} in {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()