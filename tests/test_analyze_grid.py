"""Unit tests for the per-configuration analysis.

    python3 -m tests.test_analyze_grid
"""
import sys

from scale_eval import analyze_grid


# ---------------------------------------------------------------------
# confidence intervals
# ---------------------------------------------------------------------

def test_interval_brackets_the_estimate():
    low, high = analyze_grid.wilson(15, 30)
    assert low < 0.5 < high
    assert 0.0 <= low and high <= 1.0


def test_interval_narrows_as_the_sample_grows():
    narrow = analyze_grid.wilson(500, 1000)
    wide = analyze_grid.wilson(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_interval_handles_zero_and_one_hundred_percent():
    """Wilson intervals should behave sensibly at the extremes."""
    low, high = analyze_grid.wilson(0, 20)
    assert low == 0.0 and high > 0.0

    low, high = analyze_grid.wilson(20, 20)
    assert high == 1.0 and low < 1.0


def test_empty_sample_does_not_raise():
    assert analyze_grid.wilson(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------

def test_symmetric_disagreement_means_no_difference():
    """Equal disagreement in both directions gives no evidence either way."""
    a = {f"q{i}": i < 10 for i in range(20)}
    b = {f"q{i}": i >= 10 for i in range(20)}
    only_a, only_b, p = analyze_grid.mcnemar(a, b)
    assert only_a == only_b == 10
    assert p == 1.0


def test_one_sided_disagreement_is_significant():
    a = {f"q{i}": True for i in range(15)}
    b = {f"q{i}": False for i in range(15)}
    only_a, only_b, p = analyze_grid.mcnemar(a, b)
    assert (only_a, only_b) == (15, 0)
    assert p < 0.001


def test_identical_outcomes_give_no_disagreement():
    """Identical outcomes should not break the paired test."""
    a = {f"q{i}": i % 2 == 0 for i in range(10)}
    only_a, only_b, p = analyze_grid.mcnemar(a, dict(a))
    assert (only_a, only_b, p) == (0, 0, 1.0)


def test_only_shared_questions_are_compared():
    """Only shared questions should be compared."""
    a = {"q1": True, "q2": False, "q3": True}
    b = {"q1": False, "q2": False}
    only_a, only_b, _ = analyze_grid.mcnemar(a, b)
    assert (only_a, only_b) == (1, 0), "q3 should have been ignored"


def test_agreement_is_ignored_by_the_test():
    """McNemar should only depend on discordant pairs."""
    a = {f"q{i}": True for i in range(100)}
    b = dict(a)
    b["q0"] = False
    only_a, only_b, _ = analyze_grid.mcnemar(a, b)
    assert (only_a, only_b) == (1, 0)


# ---------------------------------------------------------------------
# the grounded-answer definition
# ---------------------------------------------------------------------

def rows():
    return [
        {"id": "grounded", "answerable": True, "abstained": False,
         "best_entailment": 0.90},
        {"id": "flagged", "answerable": True, "abstained": False,
         "best_entailment": 0.10},
        {"id": "abstained", "answerable": True, "abstained": True,
         "best_entailment": None},
        {"id": "unanswerable", "answerable": False, "abstained": False,
         "best_entailment": 0.90},
    ]


def test_grounded_requires_an_answer_that_was_not_flagged():
    vector = analyze_grid.grounded_vector(rows(), threshold=0.48)
    assert vector == {"grounded": True, "flagged": False, "abstained": False}


def test_unanswerable_questions_are_excluded():
    """
    The grounded answer rate is reported over the 200 answerable questions, so
    the 40 unanswerable ones must not enter the denominator.
    """
    vector = analyze_grid.grounded_vector(rows(), threshold=0.48)
    assert "unanswerable" not in vector


def test_abstention_is_not_grounded_and_not_a_hallucination():
    """
    An abstention counts against the grounded rate but is never flagged, which
    is what stops a configuration that never answers from looking perfect.
    """
    vector = analyze_grid.grounded_vector(rows(), threshold=0.48)
    assert vector["abstained"] is False


def test_threshold_is_applied_at_the_boundary():
    """The rule is entailment >= threshold, so equality counts as grounded."""
    boundary = [{"id": "x", "answerable": True, "abstained": False,
                 "best_entailment": 0.48}]
    assert analyze_grid.grounded_vector(boundary, threshold=0.48)["x"] is True

    below = [{"id": "x", "answerable": True, "abstained": False,
              "best_entailment": 0.4799}]
    assert analyze_grid.grounded_vector(below, threshold=0.48)["x"] is False


def _run_all():
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
    sys.exit(0 if _run_all() else 1)