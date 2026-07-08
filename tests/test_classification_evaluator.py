"""
Pytest port of the self-checks in classification_evaluator.py.
Run: venv/bin/python -m pytest tests/
"""
import pytest

from classification_evaluator import ClassificationEvaluator


def test_perfect_classification():
    ev = ClassificationEvaluator([1, 0, 1, 0], [1, 0, 1, 0])
    r = ev.binary_report()
    assert r["accuracy"] == 1.0
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0
    assert r["f1"] == 1.0
    assert r["TP"] == 2 and r["TN"] == 2 and r["FP"] == 0 and r["FN"] == 0


def test_all_wrong():
    ev = ClassificationEvaluator([1, 1, 0, 0], [0, 0, 1, 1])
    r = ev.binary_report()
    assert r["accuracy"] == 0.0
    assert r["f1"] == 0.0


def test_imbalanced_trap():
    """90% accuracy but 0% F1 — the reason accuracy alone is not enough."""
    ev = ClassificationEvaluator(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    r = ev.binary_report()
    assert r["accuracy"] == 0.9
    assert r["f1"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        ClassificationEvaluator([1, 0], [1])


def test_empty_raises():
    with pytest.raises(ValueError):
        ClassificationEvaluator([], [])


def test_binary_methods_reject_multiclass():
    ev = ClassificationEvaluator([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError):
        ev.binary_report()


def test_multiclass_report():
    ev = ClassificationEvaluator([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 1])
    r = ev.multiclass_report()
    assert r["accuracy"] == pytest.approx(5 / 6)
    assert r["per_class"][0]["f1"] == 1.0
    assert 0.0 < r["macro_avg"]["f1"] <= 1.0
