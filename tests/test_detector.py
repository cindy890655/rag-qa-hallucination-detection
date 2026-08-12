"""
Unit tests for the NLI hallucination detector.

The model is loaded once and shared, because loading is the only slow part.

Run from the project root:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 -m tests.test_detector
or with pytest:
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 -m pytest tests/test_detector.py -v
"""
from detection.detector import HallucinationDetector

detector = HallucinationDetector()

PYTHON_FACT = ("Python is a high-level programming language created by Guido "
               "van Rossum and first released in 1991.")


def test_faithful_answer_is_passed():
    """A correct answer supported by the knowledge should be judged faithful (0)."""
    label, score = detector.detect(
        PYTHON_FACT, "Who created Python?",
        "Python was created by Guido van Rossum.")
    assert label == 0, f"expected faithful (0), got {label} (score={score:.3f})"


def test_hallucinated_answer_is_flagged():
    """An answer contradicted by the knowledge should be flagged (1)."""
    label, score = detector.detect(
        PYTHON_FACT, "Who created Python?",
        "Python was created by Elon Musk.")
    assert label == 1, f"expected hallucinated (1), got {label} (score={score:.3f})"


def test_return_format():
    """detect() must return (label, score): label is 0/1, score a float in [0, 1]."""
    label, score = detector.detect(
        "The Eiffel Tower is located in Paris, France.",
        "Where is the Eiffel Tower?", "The Eiffel Tower is in Paris.")
    assert label in (0, 1), f"label must be 0 or 1, got {label}"
    assert isinstance(score, float), f"score must be a float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"score must be in [0, 1], got {score}"


def test_return_labels_are_the_three_nli_classes():
    """scores() must return the three NLI classes, each a valid probability."""
    scores = detector.scores(
        "Water boils at 100 degrees Celsius at sea level.",
        "What temperature does water boil?", "Water boils at 100 C.")
    assert set(scores) == {"entailment", "neutral", "contradiction"}, \
        f"unexpected labels: {set(scores)}"
    for name, value in scores.items():
        assert 0.0 <= value <= 1.0, f"{name} probability out of range: {value}"


def test_fusion_catches_answers_the_baseline_misses():
    """
    The fusion rule fires on low entailment as well as high contradiction, which
    is what lets it catch answers the evidence fails to support rather than
    actively refutes. That is the case the contradiction baseline misses, and it
    is why fusion recovers 121 hallucinations on the benchmark.

    The example is chosen for margin, not for drama: the claim is true in the
    world but absent from the knowledge given, and measured at contradiction
    0.003 and entailment 0.046 it sits far from both thresholds. A borderline
    example would make this test depend on the model landing on one side of a
    cutoff rather than on the rule being correct.
    """
    question = "Under what licence is Python released?"
    answer = "Python is released under the PSF License."

    baseline_label, contradiction = detector.detect(
        PYTHON_FACT, question, answer)
    fusion_label, signals = detector.detect_fusion(
        PYTHON_FACT, question, answer)

    assert baseline_label == 0, (
        f"the contradiction baseline should not fire here "
        f"(contradiction={contradiction:.3f})")
    assert signals["entailment"] < 0.3, (
        f"the knowledge does not mention licensing, so entailment should be "
        f"low (got {signals['entailment']:.3f})")
    assert fusion_label == 1, (
        f"fusion should flag it on low entailment "
        f"(entailment={signals['entailment']:.3f})")


def test_fusion_returns_both_signals():
    """detect_fusion() must expose both probabilities it decided on."""
    label, signals = detector.detect_fusion(
        PYTHON_FACT, "Who created Python?",
        "Python was created by Guido van Rossum.")
    assert label in (0, 1)
    assert set(signals) == {"contradiction", "entailment"}, \
        f"unexpected signals: {set(signals)}"
    for name, value in signals.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def _run_all():
    """Simple runner so the file also works without pytest installed."""
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except AssertionError as error:
            print(f"FAIL  {test.__name__}: {error}")
        except Exception as error:
            print(f"ERROR {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)