"""
Re-score the HaluEval subset with the CURRENT model in detector.py
and overwrite cached_scores.json. Run this whenever you change the model.
"""
from load_data import load_halueval_qa
from prepare_data import build_samples
from detector import HallucinationDetector
import json

def main():
    data = load_halueval_qa("data/qa_data.json")
    samples = build_samples(data)

    N = 400                      # same subset size as before, keep it comparable
    subset = samples[:N]
    print(f"Re-scoring {len(subset)} samples with the current model (slow part)...")

    detector = HallucinationDetector()

    all_scores = []
    y_true = []
    for i, s in enumerate(subset):
        sc = detector.scores(s["knowledge"], s["question"], s["answer"])
        all_scores.append(sc)
        y_true.append(s["label"])
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(subset)}")

    with open("cached_scores.json", "w") as f:
        json.dump({"all_scores": all_scores, "y_true": y_true}, f)
    print("Saved NEW cache -> cached_scores.json")

if __name__ == "__main__":
    main()