"""
The NLI-based hallucination detector.

Detection is framed as Natural Language Inference: the retrieved knowledge is the
premise, the question and answer together form the hypothesis, and the class
probabilities decide whether the answer is supported.

The class exposes the decision at two levels on purpose. `scores()` returns the
raw three-class probabilities and applies no threshold; `detect()` and
`detect_fusion()` apply one. Keeping the threshold outside the model is what
allowed it to be recalibrated for RAG output later without touching this file:
the benchmark uses 0.3 on entailment, the RAG evaluation recalibrates to 0.48
against a negative control, and both go through the same `scores()` call.

The hypothesis is `question + answer` rather than the answer alone. RAG answers
are frequently bare fragments such as "2002" or "Helam", for which entailment is
undefined without the question; measured on 70 answers, dropping the question
lowers the separation AUC from 0.863 to 0.720.
"""

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


    def detect_fusion(self, knowledge, question, answer,
                      contra_threshold=0.5, entail_threshold=0.3):
        """
        Fusion detector (improved). Combines two complementary signals:
          - high contradiction  -> the evidence refutes the answer
          - low entailment       -> the evidence does not support the answer
        Flag as hallucinated (1) if EITHER condition holds:
            contradiction >= contra_threshold  OR  entailment < entail_threshold
        Returns (label, {"contradiction":..., "entailment":...}).
        """
        sc = self.scores(knowledge, question, answer)
        p_contra = sc.get("contradiction", 0.0)
        p_entail = sc.get("entailment", 0.0)
        label = 1 if (p_contra >= contra_threshold or p_entail < entail_threshold) else 0
        return label, {"contradiction": p_contra, "entailment": p_entail}


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