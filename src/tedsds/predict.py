"""Inference utilities: apply a fitted pipeline to a features parquet.

Shared by both :mod:`tedsds.evaluate` (which compares predictions to ground
truth) and the prediction CLI (which writes them to the ``predictions``
table).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tedsds.training import LabelColumn, feature_columns


@dataclass(frozen=True)
class PredictResult:
    ids: NDArray[np.int64]
    cycles: NDArray[np.int64]
    y_true: NDArray[np.int64] | None
    y_pred: NDArray[np.int64]
    proba: NDArray[np.float64] | None  # shape (n, n_classes) when supported
    classes: tuple[int, ...]  # in proba column order


def apply_pipeline(
    pipeline: Any,
    features_path: Path,
    *,
    label: LabelColumn | None = "label2",
) -> PredictResult:
    """Run ``pipeline`` over the features parquet at ``features_path``.

    Args:
        pipeline: a fitted sklearn ``Pipeline`` (e.g. from the registry).
        features_path: parquet from ``tedsds features``.
        label: if non-None and the column is present, ground-truth y is
            included in the result.
    """
    df = pd.read_parquet(features_path)
    cols = feature_columns(df)
    X: NDArray[np.float64] = np.nan_to_num(df[list(cols)].to_numpy(dtype=np.float64))

    y_true: NDArray[np.int64] | None = None
    if label is not None and label in df.columns:
        y_true = df[label].to_numpy(dtype=np.int64)

    y_pred = pipeline.predict(X).astype(np.int64)

    proba: NDArray[np.float64] | None = None
    classes: tuple[int, ...] = tuple(int(c) for c in getattr(pipeline, "classes_", ()))
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X).astype(np.float64)

    return PredictResult(
        ids=df["id"].to_numpy(dtype=np.int64),
        cycles=df["cycle"].to_numpy(dtype=np.int64),
        y_true=y_true,
        y_pred=y_pred,
        proba=proba,
        classes=classes,
    )
