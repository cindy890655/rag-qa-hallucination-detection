"""
Sweep the contradiction threshold and compare against the argmax rule.

The baseline here is the obvious rule: call an answer hallucinated when
contradiction is the top NLI class. The sweep asks whether an explicit threshold
does better, and by how much. Both are computed from cached_scores.json, so no
model is loaded and the two are compared on identical samples.

The threshold this picks is F1-optimal on this subset, which is exactly the kind
of fit that failed to transfer to RAG output later; see the report's discussion
of recalibration.

Run from the project root:
    python3 detection/tune_threshold.py
"""

import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def main():
    # ---- read the cache written earlier (no need to re-run the model) ----
    with open("cached_scores.json") as f:
        cache = json.load(f)
    all_scores = cache["all_scores"]
    y_true = cache["y_true"]

    # SIGNAL = contradiction probability.
    # high contradiction -> knowledge refutes the answer -> hallucinated (1).
    contra_probs = [s.get("contradiction", 0.0) for s in all_scores]

    # ---- metrics at a given contradiction threshold ----
    def metrics_at(threshold):
        # predict hallucinated (1) when contradiction >= threshold
        y_pred = [1 if p >= threshold else 0 for p in contra_probs]
        return {
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    # ---- BASELINE = argmax rule: hallucinated (1) iff contradiction is the top class ----
    y_pred_base = [1 if max(s, key=s.get) == "contradiction" else 0 for s in all_scores]
    baseline = {
        "accuracy": accuracy_score(y_true, y_pred_base),
        "precision": precision_score(y_true, y_pred_base, zero_division=0),
        "recall": recall_score(y_true, y_pred_base, zero_division=0),
        "f1": f1_score(y_true, y_pred_base, zero_division=0),
    }

    # ---- sweep threshold 0.01..0.99, pick the F1-optimal operating point ----
    best = None
    for i in range(1, 100):
        m = metrics_at(i / 100)
        if best is None or m["f1"] > best["f1"]:
            best = m

    def show(title, m):
        print(f"=== {title} ===")
        for k, v in m.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print()

    show("BASELINE (contradiction argmax rule)", baseline)
    show("TUNED (F1-optimal contradiction threshold)", best)

    print(f"F1: baseline {baseline['f1']:.4f} -> tuned {best['f1']:.4f} "
          f"at contradiction threshold {best['threshold']:.2f}")
    print(f"Accuracy: baseline {baseline['accuracy']:.4f} -> tuned {best['accuracy']:.4f}")


if __name__ == "__main__":
    main()