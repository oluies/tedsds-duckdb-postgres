"""Training pipeline for tedsds classifiers.

Reads a features Parquet (output of :mod:`tedsds.features`), drops non-feature
columns, splits into train/validation if requested, and fits a
``MinMaxScaler -> estimator`` sklearn Pipeline. Returns the fitted pipeline
plus train and (optional) validation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]

from tedsds.metrics import summarize

LabelColumn = Literal["label1", "label2"]
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {"id", "cycle", "rul", "label1", "label2", "operationmode"}
)


@dataclass(frozen=True)
class TrainResult:
    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any] | None
    n_train: int
    n_val: int


def feature_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """Return columns to feed into the model — everything that isn't an id/label."""
    return tuple(c for c in df.columns if c not in NON_FEATURE_COLUMNS)


def train_classifier(
    features_path: Path,
    *,
    estimator: BaseEstimator,
    label: LabelColumn = "label2",
    val_fraction: float = 0.0,
    seed: int = 42,
) -> TrainResult:
    """Train ``estimator`` on the features Parquet at ``features_path``.

    Args:
        features_path: Parquet produced by ``tedsds features``.
        estimator: an unfitted sklearn classifier.
        label: which label column to predict.
        val_fraction: if > 0, hold out this fraction (stratified) for validation.
        seed: random seed for the split.
    """
    df = pd.read_parquet(features_path)
    if label not in df.columns:
        raise ValueError(f"label column {label!r} missing from {features_path}")

    cols = feature_columns(df)
    # Replace any NaN (e.g. first-row STDDEV_SAMP of a single-row window) with 0.
    X: NDArray[np.float64] = np.nan_to_num(df[list(cols)].to_numpy(dtype=np.float64))
    y: NDArray[np.int64] = df[label].to_numpy(dtype=np.int64)

    if val_fraction > 0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_fraction, random_state=seed, stratify=y
        )
    else:
        X_tr, X_val, y_tr, y_val = X, None, y, None

    pipeline = Pipeline([("scaler", MinMaxScaler()), ("clf", estimator)])
    pipeline.fit(X_tr, y_tr)

    train_pred: NDArray[np.int64] = pipeline.predict(X_tr).astype(np.int64)
    train_metrics = summarize(y_tr, train_pred)

    val_metrics: dict[str, Any] | None = None
    if X_val is not None and y_val is not None:
        val_pred: NDArray[np.int64] = pipeline.predict(X_val).astype(np.int64)
        val_metrics = summarize(y_val, val_pred)

    return TrainResult(
        pipeline=pipeline,
        feature_columns=cols,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        n_train=len(y_tr),
        n_val=0 if y_val is None else len(y_val),
    )
