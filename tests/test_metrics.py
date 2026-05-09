"""Metrics correctness on hand-built predictions."""

from __future__ import annotations

import numpy as np

from tedsds.metrics import summarize


def test_summarize_perfect_predictions() -> None:
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    out = summarize(y, y.copy())
    assert out["accuracy"] == 1.0
    assert out["weighted_f1"] == 1.0
    assert out["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert out["per_label"][0]["support"] == 2


def test_summarize_known_confusion_matrix() -> None:
    # 4 labels of class 0 (3 correct, 1 misclassified as 1)
    # 4 labels of class 1 (2 correct, 2 misclassified as 0)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1], dtype=np.int64)
    out = summarize(y_true, y_pred)

    assert out["confusion_matrix"] == [[3, 1], [2, 2]]
    assert out["accuracy"] == 5 / 8

    # precision(0) = 3 / (3 + 2) = 0.6
    np.testing.assert_allclose(out["per_label"][0]["precision"], 0.6, rtol=1e-9)
    # recall(0) = 3 / 4
    np.testing.assert_allclose(out["per_label"][0]["recall"], 0.75, rtol=1e-9)
    # precision(1) = 2 / (2 + 1)
    np.testing.assert_allclose(out["per_label"][1]["precision"], 2 / 3, rtol=1e-9)


def test_summarize_is_json_serialisable() -> None:
    import json

    y_true = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    y_pred = np.array([0, 1, 2, 0, 0], dtype=np.int64)
    out = summarize(y_true, y_pred)
    json.dumps(out)  # must not raise
