"""
Unit tests for question-bank construction.

Two things are worth testing here. The reply validator decides which generated
questions enter the bank at all, and the unanswerable-set builder decides which
questions count as impossible for the corpus - a judgement the README reports as
one of the project's weaker points.

    python3 -m tests.test_question_bank
"""
import sys

from scale_eval import build_question_bank as qb


def test_a_well_formed_reply_is_accepted():
    reply = ('{"question": "In what year did Borland launch Sidekick?", '
             '"category": "technology"}')
    parsed = qb.parse_reply(reply)
    assert parsed is not None
    assert parsed["question"].endswith("?")
    assert parsed["category"] == "technology"


def test_json_wrapped_in_code_fences_is_still_parsed():
    """The prompt forbids fences; models add them anyway."""
    reply = ('```json\n{"question": "Who wrote Animal Farm in 1945?", '
             '"category": "arts"}\n```')
    assert qb.parse_reply(reply) is not None


def test_context_dependent_questions_are_rejected():
    """
    A question that only makes sense while looking at the source passage becomes
    unanswerable once retrieval decides what evidence the model actually sees.
    """
    for reply in [
        '{"question": "What does the passage say about the moa?", "category": "science"}',
        '{"question": "Which year is mentioned above?", "category": "history"}',
        '{"question": "According to the article, who won?", "category": "history"}',
    ]:
        assert qb.parse_reply(reply) is None, f"should be rejected: {reply}"


def test_statements_and_stubs_are_rejected():
    for reply in [
        '{"question": "Borland launched Sidekick in 1984.", "category": "technology"}',
        '{"question": "Why?", "category": "other"}',
        'not json at all',
        '{"category": "other"}',
    ]:
        assert qb.parse_reply(reply) is None, f"should be rejected: {reply}"


def test_unknown_category_falls_back_to_other():
    """A category outside the fixed list must not leak into the bank."""
    reply = ('{"question": "Where was Alfred Russel Wallace born?", '
             '"category": "biology"}')
    parsed = qb.parse_reply(reply)
    assert parsed is not None
    assert parsed["category"] == "other"


def test_unanswerable_set_excludes_subjects_the_corpus_covers():
    """
    A question is only unanswerable if the corpus has no article on its subject.
    Pretending the corpus covers Mount Everest must drop that question.
    """
    corpus = {"Mount Everest", "Taj Mahal", "Blockchain"}
    questions = qb.build_unanswerable(corpus, wanted=60)
    texts = " ".join(q["question"] for q in questions)

    assert "Mount Everest" not in texts
    assert "Taj Mahal" not in texts
    assert "blockchain" not in texts.lower()


def test_unanswerable_questions_are_marked_and_have_no_gold_chunk():
    questions = qb.build_unanswerable(set(), wanted=20)
    assert len(questions) == 20
    assert all(q["answerable"] is False for q in questions)
    assert all(q["gold_chunk_index"] is None for q in questions)
    assert all(q["gold_title"] is None for q in questions)
    assert all(q["category"] == "unanswerable" for q in questions)


def test_unanswerable_reasons_are_labelled():
    """
    The two kinds are reported separately, because only one of them is airtight:
    future events cannot be answered by any corpus, whereas absent-from-corpus
    was verified at title level only.
    """
    questions = qb.build_unanswerable(set(), wanted=60)
    reasons = {q["reason"] for q in questions}
    assert reasons <= {"absent-from-corpus", "unknowable"}, \
        f"unexpected reason: {reasons}"


def test_sampling_is_deterministic():
    """A fixed seed keeps the bank reproducible across runs."""
    first = qb.build_unanswerable(set(), wanted=15)
    second = qb.build_unanswerable(set(), wanted=15)
    assert [q["question"] for q in first] == [q["question"] for q in second]


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