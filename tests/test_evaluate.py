"""Evaluation metrics."""

import numpy as np
import pytest

from brain_tumor.evaluate import bootstrap_ci, classification_metrics


def test_confusion_matrix_precision_recall():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])
    probs = np.eye(4)[pred] * 0.7 + 0.075
    m = classification_metrics(y, probs)
    assert m["confusion_matrix"] == [[1, 1, 0, 0], [0, 2, 0, 0], [1, 0, 1, 0], [0, 0, 0, 2]]
    assert m["recall"] == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert m["precision"] == pytest.approx([0.5, 2 / 3, 1.0, 1.0])
    assert m["macro_recall"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_auc_is_one_for_perfect_scores_and_half_for_constant():
    y = np.repeat(np.arange(4), 5)
    perfect = classification_metrics(y, np.eye(4)[y])
    assert perfect["auc"] == pytest.approx([1.0] * 4)
    assert perfect["macro_auc"] == pytest.approx(1.0)
    flat = classification_metrics(y, np.full((20, 4), 0.25))
    assert flat["auc"] == pytest.approx([0.5] * 4)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    pred = np.where(rng.random(400) < 0.8, y, rng.integers(0, 4, 400))
    lo, hi = bootstrap_ci(y, pred, n=300, seed=0)
    point = classification_metrics(y, np.eye(4)[pred])["macro_recall"]
    assert lo < point < hi and hi - lo < 0.15
