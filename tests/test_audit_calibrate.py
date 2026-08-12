"""Unit tests for the human-audit calibration.

    python3 -m tests.test_audit_calibrate
"""
import sys

from scale_eval import audit_calibrate


def test_a_perfect_detector_needs_no_correction():
    """With recall 1 and no false alarms, the observed rate is the true rate."""
    for observed in (0.0, 0.15, 0.5, 1.0):
        assert audit_calibrate.correct_prevalence(
            observed, recall=1.0, fpr=0.0) == observed


def test_missed_detections_push_the_estimate_up():
    """A detector with half recall should double the observed rate."""
    assert audit_calibrate.correct_prevalence(
        0.10, recall=0.5, fpr=0.0) == 0.2


def test_false_alarms_push_the_estimate_down():
    """False alarms should push the corrected prevalence toward zero."""
    assert audit_calibrate.correct_prevalence(
        0.10, recall=0.5, fpr=0.10) == 0.0


def test_the_measured_operating_point_reproduces_the_reported_figure():
    """Check one headline prevalence correction from the report."""
    estimate = audit_calibrate.correct_prevalence(
        0.143, recall=0.452, fpr=0.118)
    assert abs(estimate - 0.075) < 0.005, f"got {estimate:.4f}"


def test_the_correction_preserves_the_ordering_of_configurations():
    """The correction should preserve the ordering of configurations."""
    observed = [0.143, 0.148, 0.150, 0.232, 0.255]
    corrected = [audit_calibrate.correct_prevalence(r, 0.452, 0.118)
                 for r in observed]
    assert corrected == sorted(corrected)


def test_the_estimate_stays_a_probability():
    """The corrected rate should stay within [0, 1]."""
    assert audit_calibrate.correct_prevalence(0.0, 0.452, 0.118) == 0.0
    assert audit_calibrate.correct_prevalence(1.0, 0.452, 0.118) == 1.0


def test_a_detector_with_no_signal_is_refused():
    """Reject detectors with no usable signal."""
    for recall, fpr in ((0.3, 0.3), (0.2, 0.5)):
        try:
            audit_calibrate.correct_prevalence(0.15, recall, fpr)
        except ValueError:
            continue
        raise AssertionError(
            f"should have refused recall={recall}, fpr={fpr}")


def test_interval_helper_matches_the_analysis_module():
    """Both modules should agree on Wilson intervals."""
    from scale_eval import analyze_grid
    for successes, total in ((0, 10), (5, 15), (30, 30), (11, 15)):
        assert audit_calibrate.wilson(successes, total) == \
            analyze_grid.wilson(successes, total)


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