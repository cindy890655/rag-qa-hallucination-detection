"""
Evaluate and compare the baseline detector (contradiction only) against the
fusion detector (contradiction + entailment) on the cached NLI scores.

Baseline:  hallucinated iff contradiction >= 0.5
Fusion:    hallucinated iff contradiction >= 0.5 OR entailment < 0.3

Both are evaluated on the same cached scores, so no model re-run is needed.
Prints a side-by-side metric table, both confusion matrices, and the
threshold-independent AUC of the contradiction score.
"""
import json
from sklearn.metrics import roc_auc_score, average_precision_score

CACHE = 'cached_scores.json'
CONTRA_T = 0.5
ENTAIL_T = 0.3


def compute_metrics(y_pred, y_true):
    fp = fn = tp = tn = 0
    for yp, yt in zip(y_pred, y_true):
        if yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 1 and yp == 1:
            tp += 1
        else:
            tn += 1
    n = len(y_true)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def main():
    cache = json.load(open(CACHE))
    all_scores = cache["all_scores"]
    y_true = cache["y_true"]

    baseline = [1 if s["contradiction"] >= CONTRA_T else 0 for s in all_scores]
    fusion = [1 if (s["contradiction"] >= CONTRA_T or s["entailment"] < ENTAIL_T) else 0
              for s in all_scores]

    mb = compute_metrics(baseline, y_true)
    mf = compute_metrics(fusion, y_true)

    print("=" * 60)
    print(f"Hallucination detector: baseline vs fusion  (N={len(y_true)})")
    print("=" * 60)
    print(f"{'Metric':<12}{'Baseline':<15}{'Fusion (E<0.3)':<15}")
    print("-" * 60)
    for key, name in [("acc", "Accuracy"), ("prec", "Precision"),
                      ("rec", "Recall"), ("f1", "F1")]:
        print(f"{name:<12}{mb[key]:<15.4f}{mf[key]:<15.4f}")
    print("-" * 60)
    print(f"{'Baseline':<12}TN={mb['tn']} FP={mb['fp']} FN={mb['fn']} TP={mb['tp']}")
    print(f"{'Fusion':<12}TN={mf['tn']} FP={mf['fp']} FN={mf['fn']} TP={mf['tp']}")
    print("-" * 60)

    y_score = [s["contradiction"] for s in all_scores]
    print(f"ROC-AUC (contradiction score): {roc_auc_score(y_true, y_score):.4f}")
    print(f"PR-AUC  (contradiction score): {average_precision_score(y_true, y_score):.4f}")
    print("=" * 60)
    print(f"Fusion recovers {mb['fn'] - mf['fn']} hallucinations missed by the baseline "
          f"(FN {mb['fn']} -> {mf['fn']}).")


if __name__ == "__main__":
    main()