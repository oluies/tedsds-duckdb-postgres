"""Trainer smoke tests on synthetic separable features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tedsds.sensors import SETTINGS
from tedsds.train_lr import train as train_lr
from tedsds.train_rf import train as train_rf
from tedsds.training import feature_columns, train_classifier


def _write_synthetic_features(path: Path, n_per_class: int = 60, seed: int = 0) -> None:
    """Three well-separated clouds — easy for any classifier."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls, centre in enumerate([(0.1,) * 21, (0.5,) * 21, (0.9,) * 21]):
        for _ in range(n_per_class):
            rows.append((cls, np.array(centre) + rng.normal(0, 0.02, size=21)))
    df = pd.DataFrame(
        {
            "id": list(range(1, len(rows) + 1)),
            "cycle": [1] * len(rows),
            **{s: 0.0 for s in SETTINGS},
            **{f"a{i}": [r[1][i - 1] for r in rows] for i in range(1, 22)},
            **{f"sd{i}": 0.0 for i in range(1, 22)},
            "rul": [10] * len(rows),
            "label1": [r[0] >= 1 for r in rows],
            "label2": [r[0] for r in rows],
        }
    )
    df["label1"] = df["label1"].astype(int)
    df["label2"] = df["label2"].astype(int)
    pq.write_table(pa.Table.from_pandas(df), path)  # type: ignore[no-untyped-call]


def test_feature_columns_excludes_id_label_rul(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    _write_synthetic_features(p)
    df = pd.read_parquet(p)
    cols = feature_columns(df)
    assert "id" not in cols
    assert "cycle" not in cols
    assert "rul" not in cols
    assert "label1" not in cols
    assert "label2" not in cols
    assert "a1" in cols
    assert "sd21" in cols


def test_train_rf_on_separable_data_high_accuracy(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    _write_synthetic_features(p)
    result = train_rf(p, label="label2", val_fraction=0.25, n_estimators=20, seed=42)
    assert result.train_metrics["accuracy"] > 0.95
    assert result.val_metrics is not None
    assert result.val_metrics["accuracy"] > 0.95


def test_train_lr_on_separable_data_high_accuracy(tmp_path: Path) -> None:
    p = tmp_path / "f.parquet"
    _write_synthetic_features(p)
    result = train_lr(p, label="label2", val_fraction=0.25, seed=42)
    assert result.train_metrics["accuracy"] > 0.95


def test_train_classifier_no_val_fraction_returns_none(tmp_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

    p = tmp_path / "f.parquet"
    _write_synthetic_features(p)
    result = train_classifier(p, estimator=LogisticRegression(), val_fraction=0.0)
    assert result.val_metrics is None
    assert result.n_val == 0


def test_train_handles_nan_features(tmp_path: Path) -> None:
    """First sd column is NaN-heavy in real data; trainer must not error."""
    p = tmp_path / "f.parquet"
    _write_synthetic_features(p)
    df = pd.read_parquet(p)
    df.loc[df.index < 10, "sd1"] = np.nan
    pq.write_table(pa.Table.from_pandas(df), p)  # type: ignore[no-untyped-call]
    result = train_rf(p, label="label2", val_fraction=0.0, n_estimators=10, seed=42)
    assert result.train_metrics["accuracy"] > 0.5
