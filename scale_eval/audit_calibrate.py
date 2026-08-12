"""
Turn the 30 human labels into the detector's in-domain precision and recall, and
use them to bias-correct the hallucination rate of every configuration.

Why this step exists. Every number in the scaled-up evaluation is produced by the
detector, so all of them inherit its errors. On HaluEval the fusion detector had
recall 0.656, meaning it missed a third of the hallucinations there; without an
in-domain measurement there is no way to know whether the rates reported for RAG
output are close to the truth or badly biased.

Stratified sampling has to be undone. The audit sample was not drawn uniformly:
15 flagged answers, 8 unflagged-with-retrieval-hit and 7 unflagged-with-
retrieval-miss, out of populations of very different sizes. Precision can still
be read directly off the flagged stratum, because within that stratum the draw
was uniform. Recall cannot: its denominator includes hallucinations sitting in
the unflagged strata, which are sampled at a completely different rate. Every
sampled item therefore carries a weight equal to the size of its stratum in the
population divided by the number drawn from it.

Bias correction. For a screening test with true positive rate TPR and false
positive rate FPR, an observed flag rate r implies a true prevalence of
(r - FPR) / (TPR - FPR). This is the standard prevalence estimator; it is only
meaningful when TPR > FPR, which the negative control already established.

Outputs
-------
    data/rag_eval/analysis/calibration.json
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
RAG_EVAL = DETECTION_ROOT / "data" / "rag_eval"
AUDIT_DIR = RAG_EVAL / "audit"
ANALYSIS_DIR = RAG_EVAL / "analysis"

LABEL_CSV = AUDIT_DIR / "audit_filled.csv"
KEY_JSON = AUDIT_DIR / "audit_key.json"


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)

    z = 1.96
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d

    return (max(0.0, centre - spread), min(1.0, centre + spread))


def load_labels() -> dict[str, int]:
    if not LABEL_CSV.exists():
        raise SystemExit(f"Not found: {LABEL_CSV}")

    labels = {}
    with LABEL_CSV.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            value = row["unsupported"].strip()
            if value in ("0", "1"):
                labels[row["audit_id"].strip()] = int(value)

    return labels


def load_notes() -> dict[str, str]:
    with LABEL_CSV.open(encoding="utf-8-sig") as handle:
        return {row["audit_id"].strip(): row["notes"].strip()
                for row in csv.DictReader(handle)
                if row.get("notes", "").strip()}


def population_strata(threshold: float) -> Counter:
    """
    Rebuild the pool build_audit_sample.py drew from, and count each stratum.
    Must mirror that script's filtering and de-duplication exactly, or the
    weights will be wrong.
    """
    evidence_present = set()
    for path in (RAG_EVAL / "grid").glob("rag_*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["retrieved_documents"]:
                    evidence_present.add(
                        (record["config"]["config_id"], record["id"]))

    counts = Counter()
    seen = set()

    for path in sorted((RAG_EVAL / "scored").glob("scored_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                row = json.loads(line)

                if row["abstained"] or row["best_entailment"] is None:
                    continue

                fingerprint = (row["id"], row["answer"].strip().lower())
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)

                if (row["config_id"], row["id"]) not in evidence_present:
                    continue

                flagged = row["best_entailment"] < threshold
                counts["flagged" if flagged else
                       "clean_hit" if row["chunk_hit"] else "clean_miss"] += 1

    return counts


def main() -> None:
    summary_path = ANALYSIS_DIR / "summary.json"
    if not summary_path.exists():
        raise SystemExit("Run analyze_grid.py first")

    summary = json.load(summary_path.open(encoding="utf-8"))
    threshold = summary["threshold"]

    labels = load_labels()
    notes = load_notes()
    key = json.load(KEY_JSON.open(encoding="utf-8"))["key"]

    population = population_strata(threshold)
    sampled = Counter(key[audit_id]["stratum"] for audit_id in labels)

    weights = {stratum: population[stratum] / sampled[stratum]
               for stratum in sampled}

    print("=" * 74)
    print("STRATUM WEIGHTS")
    print("=" * 74)
    print(f"{'stratum':<16}{'population':<14}{'sampled':<10}{'weight'}")
    print("-" * 74)
    for stratum in ("flagged", "clean_hit", "clean_miss"):
        if stratum in sampled:
            print(f"{stratum:<16}{population[stratum]:<14}"
                  f"{sampled[stratum]:<10}{weights[stratum]:.2f}")

    # weighted confusion matrix: rows = human truth, columns = detector
    wtp = wfp = wfn = wtn = 0.0
    # unweighted counts inside the flagged stratum, for an exact precision CI
    flagged_total = flagged_true = 0

    for audit_id, human in labels.items():
        entry = key[audit_id]
        detector = 1 if entry["detector_flagged"] else 0
        weight = weights[entry["stratum"]]

        if human == 1 and detector == 1:
            wtp += weight
        elif human == 0 and detector == 1:
            wfp += weight
        elif human == 1 and detector == 0:
            wfn += weight
        else:
            wtn += weight

        if entry["stratum"] == "flagged":
            flagged_total += 1
            flagged_true += human

    precision = flagged_true / flagged_total if flagged_total else 0.0
    p_low, p_high = wilson(flagged_true, flagged_total)

    recall = wtp / (wtp + wfn) if (wtp + wfn) else 0.0
    fpr = wfp / (wfp + wtn) if (wfp + wtn) else 0.0
    specificity = 1 - fpr
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    prevalence = (wtp + wfn) / (wtp + wfp + wfn + wtn)

    print("\n" + "=" * 74)
    print("DETECTOR PERFORMANCE ON REAL RAG OUTPUT")
    print("=" * 74)
    print(f"human labels          : {len(labels)}  "
          f"({sum(labels.values())} unsupported, "
          f"{len(labels) - sum(labels.values())} supported)")
    print(f"agreement with detector: "
          f"{sum(1 for a, h in labels.items() if h == int(key[a]['detector_flagged']))}"
          f"/{len(labels)}")
    print()
    print(f"precision  : {precision:.3f}  [{p_low:.3f}, {p_high:.3f}]   "
          f"exact: read off the flagged stratum, n={flagged_total}")
    print(f"recall     : {recall:.3f}                    "
          f"weighted across strata, CI not exact")
    print(f"specificity: {specificity:.3f}")
    print(f"FPR        : {fpr:.3f}")
    print(f"F1         : {f1:.3f}")
    print()
    print(f"weighted confusion    : TP={wtp:.1f}  FP={wfp:.1f}  "
          f"FN={wfn:.1f}  TN={wtn:.1f}")
    print(f"prevalence in the pool: {prevalence:.1%}")

    if recall <= fpr:
        raise SystemExit("\nrecall <= FPR: the detector carries no signal and "
                         "the correction below would be meaningless")

    print("\n" + "=" * 74)
    print("BIAS-CORRECTED HALLUCINATION RATE PER CONFIGURATION")
    print("=" * 74)
    print(f"{'config':<22}{'detector says':<16}{'corrected':<14}{'shift'}")
    print("-" * 74)

    corrected = {}
    for config, metrics in summary["metrics"].items():
        observed = metrics["hallucination_rate"]
        estimate = min(1.0, max(0.0, (observed - fpr) / (recall - fpr)))
        corrected[config] = estimate
        print(f"{config:<22}{observed:<16.1%}{estimate:<14.1%}"
              f"{estimate - observed:+.1%}")

    print("-" * 74)
    print("correction: p = (observed - FPR) / (TPR - FPR)")

    if notes:
        print("\n" + "=" * 74)
        print("FAILURE TYPES among the human-labelled unsupported answers")
        print("=" * 74)
        failures = Counter(note for audit_id, note in notes.items()
                           if labels.get(audit_id) == 1)
        total = sum(failures.values())
        for tag, count in failures.most_common():
            print(f"  {tag:<32}{count:>3}  ({count / total:.0%})")

        premise = [a for a, n in notes.items()
                   if n == "question-premise-inaccurate"]
        if premise:
            print(f"\n  questions that misstate their source passage: "
                  f"{len(premise)}/{len(labels)} "
                  f"({len(premise) / len(labels):.1%})  {premise}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / "calibration.json"
    json.dump(
        {
            "threshold": threshold,
            "n_labelled": len(labels),
            "population_strata": dict(population),
            "sampled_strata": dict(sampled),
            "weights": weights,
            "precision": precision,
            "precision_ci": [p_low, p_high],
            "recall": recall,
            "specificity": specificity,
            "false_positive_rate": fpr,
            "f1": f1,
            "prevalence_estimate": prevalence,
            "weighted_confusion": {"tp": wtp, "fp": wfp, "fn": wfn, "tn": wtn},
            "corrected_hallucination_rate": corrected,
            "failure_types": dict(Counter(
                note for audit_id, note in notes.items()
                if labels.get(audit_id) == 1)),
        },
        out.open("w", encoding="utf-8"),
        indent=2,
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()