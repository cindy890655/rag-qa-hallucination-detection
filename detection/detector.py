from transformers import pipeline


class HallucinationDetector:
    """
    NLI-based hallucination detector.
    It checks whether an answer is supported by the given knowledge.
    """

    def __init__(self, model_name="facebook/bart-large-mnli"):
        """
        Load the pretrained NLI model.
        Input: model_name - the HuggingFace model id (string)
        """
        self.nli = pipeline("text-classification", model=model_name)

    def detect(self, knowledge, question, answer):
        """
        Judge whether an answer is hallucinated given the knowledge.
        Input:
            knowledge - the evidence text (string)
            question  - the question being answered (string)
            answer    - the answer to check (string)
        Output:
            label - 1 if hallucinated, 0 if faithful (int)
            nli_label - the raw NLI label (string): entailment / contradiction / neutral
        """
        # combine question + answer into a full hypothesis sentence
        hypothesis = f"{question} {answer}"
        # NLI input: premise (knowledge) + hypothesis
        text = f"{knowledge} </s></s> {hypothesis}"
        result = self.nli(text)[0]
        nli_label = result["label"].lower()

        # entailment -> knowledge supports answer -> faithful (0)
        # otherwise (contradiction / neutral) -> hallucinated (1)
        if nli_label == "entailment":
            label = 0
        else:
            label = 1

        return label, nli_label


# quick check on a few examples
if __name__ == "__main__":
    detector = HallucinationDetector()

    knowledge = "Arthur's Magazine (1844-1846) was an American literary periodical. First for Women is a woman's magazine published by Bauer Media Group."
    question = "Which magazine was started first Arthur's Magazine or First for Women?"

    faithful_answer = "Arthur's Magazine"
    hallucinated_answer = "First for Women was started first."

    print("Faithful answer:")
    print(detector.detect(knowledge, question, faithful_answer))

    print("Hallucinated answer:")
    print(detector.detect(knowledge, question, hallucinated_answer))