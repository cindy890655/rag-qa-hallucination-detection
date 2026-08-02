"""
Unit tests for the hallucination detector.

Run from the detection folder with:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 -m pytest test_detector.py -v
or simply:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 test_detector.py
"""
from detector import HallucinationDetector

# Load the model once and reuse it across all tests (loading is the slow part).
detector = HallucinationDetector()


def test_faithful_answer_is_passed():
    """A correct answer supported by the knowledge should be judged faithful (0)."""
    knowledge = "Python is a high-level programming language created by Guido van Rossum in 1991."
    question = "Who created Python?"
    answer = "Python was created by Guido van Rossum."
    label, score = detector.detect(knowledge, question, answer)
    assert label == 0, f"expected faithful (0), got {label} (score={score:.3f})"


def test_hallucinated_answer_is_flagged():
    """An answer contradicted by the knowledge should be flagged as hallucinated (1)."""
    knowledge = "Python is a high-level programming language created by Guido van Rossum in 1991."
    question = "Who created Python?"
    answer = "Python was created by Elon Musk."
    label, score = detector.detect(knowledge, question, answer)
    assert label == 1, f"expected hallucinated (1), got {label} (score={score:.3f})"


def test_return_format():
    """detect() must return (label, score): label is 0/1, score is a float in [0, 1]."""
    knowledge = "The Eiffel Tower is located in Paris, France."
    question = "Where is the Eiffel Tower?"
    answer = "The Eiffel Tower is in Paris."
    label, score = detector.detect(knowledge, question, answer)
    assert label in (0, 1), f"label must be 0 or 1, got {label}"
    assert isinstance(score, float), f"score must be a float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"score must be in [0, 1], got {score}"


def test_return_labels_are_the_three_nli_classes():
    """scores() must return the three NLI classes, each a valid probability."""
    knowledge = "Water boils at 100 degrees Celsius at sea level."
    sc = detector.scores(knowledge, "What temperature does water boil?", "Water boils at 100 C.")
    assert set(sc.keys()) == {"entailment", "neutral", "contradiction"}, \
        f"unexpected labels: {sc.keys()}"
    for name, value in sc.items():
        assert 0.0 <= value <= 1.0, f"{name} probability out of range: {value}"


def _run_all():
    """Simple runner so the file also works without pytest installed."""
    tests = [
        test_faithful_answer_is_passed,
        test_hallucinated_answer_is_flagged,
        test_return_format,
        test_return_labels_are_the_three_nli_classes,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()