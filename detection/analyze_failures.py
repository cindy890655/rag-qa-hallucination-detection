import json
from load_data import load_halueval_qa
from prepare_data import build_samples
from sklearn.metrics import f1_score


def best_threshold(contra_probs, y_true):
    """Re-find the F1-optimal CONTRADICTION threshold from the cache (no model needed)."""
    best_t, best_f1 = 0.5, -1.0
    for i in range(1, 100):
        t = i / 100
        y_pred = [1 if p >= t else 0 for p in contra_probs]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def main():
    # ---- load the cache written by tune_threshold.py ----
    with open("cached_scores.json") as f:
        cache = json.load(f)
    all_scores = cache["all_scores"]
    y_true = cache["y_true"]
    # SIGNAL = contradiction: high contradiction -> hallucinated (1)
    contra_probs = [s.get("contradiction", 0.0) for s in all_scores]

    # ---- rebuild the SAME samples so we can read their text (same N, same order) ----
    data = load_halueval_qa("data/qa_data.json")
    samples = build_samples(data)[: len(y_true)]

    # ---- apply the F1-optimal contradiction threshold ----
    t = best_threshold(contra_probs, y_true)
    y_pred = [1 if p >= t else 0 for p in contra_probs]
    print(f"Using F1-optimal contradiction threshold = {t:.2f}\n")

    # ---- collect the two kinds of mistakes ----
    # false positive: true faithful (0) but predicted hallucinated (1)
    #   -> a correct answer wrongly flagged (knowledge didn't clearly support it)
    # false negative: true hallucinated (1) but predicted faithful (0)
    #   -> a hallucination that slipped through (knowledge didn't clearly refute it)
    false_pos, false_neg = [], []
    for smp, yt, yp, p in zip(samples, y_true, y_pred, contra_probs):
        if yt == 0 and yp == 1:
            false_pos.append((smp, p))
        elif yt == 1 and yp == 0:
            false_neg.append((smp, p))

    total_wrong = len(false_pos) + len(false_neg)
    print(f"Total mistakes: {total_wrong}  "
          f"(false positives: {len(false_pos)}, false negatives: {len(false_neg)})\n")

    def show(title, cases, k=3):
        print(f"===== {title} (showing up to {k}) =====")
        for smp, p in cases[:k]:
            print(f"[P(contradiction)={p:.2f}]")
            print(f"  Q:         {smp['question']}")
            print(f"  Answer:    {smp['answer']}")
            print(f"  Knowledge: {smp['knowledge'][:160]}...")
            print()

    show("FALSE POSITIVES  (faithful answer wrongly flagged as hallucinated)", false_pos)
    show("FALSE NEGATIVES  (hallucination that slipped through)", false_neg)

    # ---- save all mistakes for the report ----
    with open("failure_cases.json", "w") as f:
        json.dump({
            "threshold": t,
            "false_positives": [s for s, _ in false_pos],
            "false_negatives": [s for s, _ in false_neg],
        }, f, ensure_ascii=False, indent=2)
    print("Saved all mistakes -> failure_cases.json")


if __name__ == "__main__":
    main()