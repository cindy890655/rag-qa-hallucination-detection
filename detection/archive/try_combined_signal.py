import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Reads the cache written by tune_threshold.py. No model re-run needed.
with open("cached_scores.json") as f:
    cache = json.load(f)
all_scores = cache["all_scores"]
y_true = cache["y_true"]

E = [s.get("entailment", 0.0) for s in all_scores]
C = [s.get("contradiction", 0.0) for s in all_scores]

# Three candidate "hallucination scores". Higher = more likely hallucinated.
signals = {
    "contradiction only        (current)": C,
    "1 - entailment            (unsupported)": [1 - e for e in E],
    "contradiction - entailment (combined)": [c - e for c, e in zip(C, E)],
}


def best_over_threshold(score):
    lo, hi = min(score), max(score)
    best = None
    steps = 100
    for i in range(1, steps):
        t = lo + (hi - lo) * i / steps
        y_pred = [1 if v >= t else 0 for v in score]
        m = {
            "threshold": t,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best


print(f"{'signal':<40} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'thr':>7}")
print("-" * 74)
for name, score in signals.items():
    b = best_over_threshold(score)
    print(f"{name:<40} {b['accuracy']:.3f} {b['precision']:>6.3f} "
          f"{b['recall']:>6.3f} {b['f1']:>6.3f} {b['threshold']:>7.3f}")