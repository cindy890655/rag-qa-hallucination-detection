"""Unit tests for chunk-wise scoring.

    python3 -m tests.test_score_grid
"""
import sys

from scale_eval import score_grid


class StubDetector:
    """Return canned scores one chunk at a time."""

    def __init__(self, per_chunk):
        self.per_chunk = per_chunk
        self.calls = 0

    def scores(self, knowledge, question, answer):
        entailment, contradiction = self.per_chunk[self.calls]
        self.calls += 1
        return {"entailment": entailment, "neutral": 0.0,
                "contradiction": contradiction}


def chunks(count):
    return [{"content": f"chunk {i}", "title": f"T{i}"} for i in range(count)]


def test_every_chunk_is_scored_separately():
    """No concatenation, so no truncation."""
    detector = StubDetector([(0.10, 0.80), (0.95, 0.02), (0.40, 0.30)])
    score_grid.score_chunks(detector, chunks(3), "Q?", "A")
    assert detector.calls == 3, \
        f"expected one call per chunk, got {detector.calls}"


def test_best_entailment_is_the_maximum_over_chunks():
    detector = StubDetector([(0.10, 0.80), (0.95, 0.02), (0.40, 0.30)])
    verdict = score_grid.score_chunks(detector, chunks(3), "Q?", "A")
    assert verdict["best_entailment"] == 0.95


def test_min_contradiction_is_the_minimum_over_chunks():
    """The contradiction summary is the minimum across chunks."""
    detector = StubDetector([(0.10, 0.80), (0.95, 0.02), (0.40, 0.30)])
    verdict = score_grid.score_chunks(detector, chunks(3), "Q?", "A")
    assert verdict["min_contradiction"] == 0.02


def test_per_chunk_detail_is_kept():
    """Keep the per-chunk scores for inspection."""
    detector = StubDetector([(0.10, 0.80), (0.95, 0.02)])
    verdict = score_grid.score_chunks(detector, chunks(2), "Q?", "A")
    assert len(verdict["per_chunk"]) == 2
    for entry in verdict["per_chunk"]:
        assert set(entry) == {"contradiction", "entailment", "flag"}


def test_one_supporting_chunk_grounds_the_answer():
    """Any supporting chunk should ground the answer."""
    detector = StubDetector([(0.01, 0.05), (0.99, 0.00)])
    verdict = score_grid.score_chunks(detector, chunks(2), "Q?", "A")
    assert verdict["flag"] is False


def test_an_answer_no_chunk_supports_is_flagged():
    detector = StubDetector([(0.05, 0.10), (0.08, 0.20)])
    verdict = score_grid.score_chunks(detector, chunks(2), "Q?", "A")
    assert verdict["flag"] is True


def test_a_contradicting_chunk_alone_does_not_ground_an_answer():
    """High entailment from one chunk cannot be produced by contradiction."""
    detector = StubDetector([(0.02, 0.97)])
    verdict = score_grid.score_chunks(detector, chunks(1), "Q?", "A")
    assert verdict["flag"] is True


def test_single_chunk_behaves_like_the_original_rule():
    """At k=1, chunk-wise scoring should match the benchmark rule."""
    supported = StubDetector([(0.90, 0.01)])
    assert score_grid.score_chunks(
        supported, chunks(1), "Q?", "A")["flag"] is False

    refuted = StubDetector([(0.05, 0.90)])
    assert score_grid.score_chunks(
        refuted, chunks(1), "Q?", "A")["flag"] is True


def test_scores_are_rounded_for_storage():
    """Stored to four decimals; enough for a 0.48 threshold, small on disk."""
    detector = StubDetector([(0.123456789, 0.987654321)])
    verdict = score_grid.score_chunks(detector, chunks(1), "Q?", "A")
    assert verdict["best_entailment"] == 0.1235
    assert verdict["min_contradiction"] == 0.9877


def test_control_offset_matches_the_llm_judge():
    """
    The negative control pairs each answer with another question's evidence. The
    NLI scorer and the LLM judge must use the same offset, or the two methods
    would be validated on different control pairs and could not be compared.
    """
    from llm_judge import llm_judge_grid
    assert score_grid.CONTROL_OFFSET == llm_judge_grid.CONTROL_OFFSET


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