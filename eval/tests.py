import numpy as np

from eval.aggregate import aggregate_probs
from eval.metrics import ece, macro_f1
from eval.score import gated_score, gated_scores


def test_macro_f1_perfect():
    assert macro_f1([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]) == 1.0


def test_gated_perfect():
    b = [0, 1, 2]
    d = ["A", "B", "C"]
    q = ["UOQ", "LIQ", "CENTER"]
    assert np.allclose(gated_scores(b, d, q, b, d, q), 1.0)


def test_gated_wrong_birads_is_zero():
    assert gated_scores([0], ["A"], ["UOQ"], [1], ["A"], ["UOQ"])[0] == 0.0


def test_gated_birads_only():
    assert np.isclose(gated_scores([0], ["B"], ["LIQ"], [0], ["A"], ["UOQ"])[0], 0.70)


def test_gated_missing_quadrant_full():
    assert np.isclose(gated_scores([1], ["A"], [None], [1], ["A"], [""])[0], 1.0)


def test_gated_missing_quadrant_skip():
    assert np.isclose(gated_scores([1], ["A"], [None], [1], ["A"], [""],
                                   quadrant_missing="skip")[0], 1.0)
    assert np.isclose(gated_scores([1], ["B"], [None], [1], ["A"], [""],
                                   quadrant_missing="skip")[0], 0.70 / 0.85)


def test_gated_score_mean():
    b = [0, 1]
    d = ["A", "B"]
    q = ["UOQ", "LIQ"]
    assert np.isclose(gated_score(b, d, q, b, d, q), 1.0)


def test_aggregate_mean():
    probs = np.array([[0.2, 0.8], [0.4, 0.6]])
    order, out = aggregate_probs(probs, ["a", "a"], "mean")
    assert order == ["a"]
    assert np.allclose(out, [[0.3, 0.7]])


def test_aggregate_max():
    probs = np.array([[0.2, 0.8], [0.4, 0.6]])
    _, out = aggregate_probs(probs, ["a", "a"], "max")
    assert np.allclose(out, [[0.4, 0.8]])


def test_ece_perfect_calibration():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(probs, [0, 1]) < 1e-9
