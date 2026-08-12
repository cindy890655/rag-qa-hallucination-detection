"""Unit tests for the generator loader helpers.

    python3 -m tests.test_fast_generator
"""
import sys

from scale_eval import fast_generator


def test_short_aliases_resolve_to_model_ids():
    """Short aliases should resolve to full model IDs."""
    assert fast_generator.resolve_model_name("phi3") == \
        "microsoft/Phi-3-mini-4k-instruct"
    assert fast_generator.resolve_model_name("flan-base") == \
        "google/flan-t5-base"
    assert fast_generator.resolve_model_name("flan-large") == \
        "google/flan-t5-large"


def test_unknown_names_pass_through():
    """Unknown names should pass through unchanged."""
    assert fast_generator.resolve_model_name("some-org/some-model") == \
        "some-org/some-model"


def test_aliases_match_the_configurations_that_were_run():
    """Every generator used in the results tables should resolve."""
    for name in ("phi3", "flan-base", "flan-large"):
        assert name in fast_generator.MODEL_ALIASES


def test_device_choice_is_a_valid_torch_device():
    assert fast_generator.pick_device() in ("mps", "cuda", "cpu")


def test_input_budget_matches_the_pipeline():
    """The generator should use the same input budget as the pipeline."""
    assert fast_generator.MAX_INPUT_TOKENS == 512


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