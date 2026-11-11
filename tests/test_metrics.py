import pytest
from qa_suite.metrics import cohens_kappa, fleiss_kappa, interpret_kappa, wilson_interval


def test_cohens_kappa_known_value():
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(0.5)


def test_cohens_kappa_perfect_and_degenerate():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert cohens_kappa(["a", "a"], ["a", "a"]) == 1.0  # pe == 1 edge case


def test_cohens_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        cohens_kappa([], [])
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_fleiss_kappa_known_values():
    assert fleiss_kappa([["a", "a", "a"], ["b", "b", "b"]]) == pytest.approx(1.0)
    assert fleiss_kappa([["a", "a", "b"], ["a", "b", "b"]]) == pytest.approx(-1 / 3)


def test_interpretation_bands():
    assert interpret_kappa(-0.1) == "poor"
    assert interpret_kappa(0.35) == "fair"
    assert interpret_kappa(0.7) == "substantial"
    assert interpret_kappa(0.9) == "almost perfect"


def test_wilson_interval():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(15, 30)
    assert lo < 0.5 < hi
