from transformers import pipeline


class HallucinationDetector:
    """
    NLI-based hallucination detector.
    It checks whether an answer is supported by the given knowledge.
    """

    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
        """
        Load the pretrained NLI model.
        top_k=None makes the pipeline return the score for ALL three classes
        (entailment / neutral / contradiction), not just the top one.
        We need the entailment probability to tune a threshold later.
        """
        self.nli = pipeline("text-classification", model=model_name, top_k=None)

    def scores(self, knowledge, question, answer):
        """
        Return the NLI class probabilities as a dict, e.g.
        {"entailment": 0.82, "neutral": 0.11, "contradiction": 0.07}
        """
        hypothesis = f"{question} {answer}"
        text = f"{knowledge} </s></s> {hypothesis}"
        out = self.nli(text)[0]                       # list of {label, score}
        return {s["label"].lower(): s["score"] for s in out}

    def detect(self, knowledge, question, answer, threshold=0.5):
        """
        Signal = contradiction probability.
        With the DeBERTa-FEVER model, faithful answers score near 0 and
        hallucinated answers score near 1 on contradiction, so a midpoint
        threshold cleanly separates them.

        hallucinated (1) iff P(contradiction) >= threshold, else faithful (0).

        Returns (label, contradiction_prob).
        """
        sc = self.scores(knowledge, question, answer)
        p_contra = sc.get("contradiction", 0.0)
        label = 1 if p_contra >= threshold else 0
        return label, p_contra

# quick check on a few examples
if __name__ == "__main__":
    detector = HallucinationDetector()

    knowledge = ("Arthur's Magazine (1844-1846) was an American literary periodical. "
                 "First for Women is a woman's magazine published by Bauer Media Group.")
    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    faithful_answer = "Arthur's Magazine"
    hallucinated_answer = "First for Women was started first."

    print("Faithful answer:    ", detector.detect(knowledge, question, faithful_answer))
    print("Hallucinated answer:", detector.detect(knowledge, question, hallucinated_answer))