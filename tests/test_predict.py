"""Tests for tedsds.predict and tedsds.evaluate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tedsds.evaluate import evaluate_pipeline
from tedsds.predict import apply_pipeline
from tedsds.sensors import SETTINGS
from tedsds.train_lr import train as train_lr


def _write_synthetic_features(
    path: Path, n_per_class: int = 60, seed: int = 0, with_label2: bool = True
) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for cls, centre in enumerate([(0.1,) * 21, (0.5,) * 21, (0.9,) * 21]):
        for _ in range(n_per_class):
            rows.append((cls, np.array(centre) + rng.normal(0, 0.02, size=21)))

    data: dict[str, object] = {
        "id": list(range(1, len(rows) + 1)),
        "cycle": [1] * len(rows),
        **{s: 0.0 for s in SETTINGS},
        **{f"a{i}": [r[1][i - 1] for r in rows] for i in range(1, 22)},
        **{f"sd{i}": 0.0 for i in range(1, 22)},
        "rul": [10] * len(rows),
        "label1": [int(r[0] >= 1) for r in rows],
    }
    if with_label2:
        data["label2"] = [int(r[0]) for r in rows]
    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), path)  # type: ignore[no-untyped-call]


def test_apply_pipeline_returns_predictions_and_truth(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    _write_synthetic_features(p)
    pipeline = train_lr(p, label="label2", val_fraction=0.0, seed=42).pipeline

    res = apply_pipeline(pipeline, p, label="label2")
    assert res.y_pred.shape == res.ids.shape == res.cycles.shape
    assert res.y_true is not None
    assert res.proba is not None
    assert res.proba.shape == (len(res.y_pred), 3)
    # 3 classes for our synthetic data
    assert set(res.classes) == {0, 1, 2}
    # Probability rows sum to 1
    np.testing.assert_allclose(res.proba.sum(axis=1), 1.0, atol=1e-6)


def test_apply_pipeline_returns_none_truth_if_label_missing(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    _write_synthetic_features(p, with_label2=False)
    pipeline = train_lr(p, label="label1", val_fraction=0.0, seed=42).pipeline
    res = apply_pipeline(pipeline, p, label="label2")  # label2 absent
    assert res.y_true is None


def test_evaluate_pipeline_high_accuracy(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    _write_synthetic_features(p)
    pipeline = train_lr(p, label="label2", val_fraction=0.0, seed=42).pipeline
    metrics = evaluate_pipeline(pipeline, p, label="label2")
    assert metrics["accuracy"] > 0.95
    assert metrics["n"] == 180


def test_evaluate_pipeline_raises_when_label_missing(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    _write_synthetic_features(p, with_label2=False)
    pipeline = train_lr(p, label="label1", val_fraction=0.0, seed=42).pipeline
    with pytest.raises(ValueError, match="label2"):
        evaluate_pipeline(pipeline, p, label="label2")
