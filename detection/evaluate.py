from load_data import load_halueval_qa
from prepare_data import build_samples
from detector import HallucinationDetector
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def evaluate(detector, samples):
    """
    Run the detector on a list of samples and compute metrics.
    Input:
        detector - a HallucinationDetector instance
        samples  - list of dicts with keys: question / knowledge / answer / label
    Output:
        metrics - dict with accuracy / precision / recall / f1
    """
    y_true = []
    y_pred = []

    for i, sample in enumerate(samples):
        pred_label, _ = detector.detect(
            sample["knowledge"], sample["question"], sample["answer"]
        )
        y_true.append(sample["label"])
        y_pred.append(pred_label)

        # print progress every 20 samples
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(samples)} samples")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    return metrics


if __name__ == "__main__":
    # load data and build samples
    data = load_halueval_qa("data/qa_data.json")
    samples = build_samples(data)

    # use a small subset first (evaluating all 20000 is slow)
    subset = samples[:100]
    print(f"Evaluating on {len(subset)} samples...")

    detector = HallucinationDetector()
    metrics = evaluate(detector, subset)

    print("\nResults:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")