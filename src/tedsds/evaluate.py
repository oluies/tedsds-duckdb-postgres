"""Standalone evaluator: load a registered model, score it on a feature
parquet, and produce the metrics dict used elsewhere in the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tedsds.metrics import summarize
from tedsds.predict import apply_pipeline
from tedsds.training import LabelColumn


def evaluate_pipeline(
    pipeline: Any,
    features_path: Path,
    *,
    label: LabelColumn = "label2",
) -> dict[str, Any]:
    """Run ``pipeline`` against features and return ``metrics.summarize``.

    Raises:
        ValueError: if the features parquet is missing the requested label column.
    """
    result = apply_pipeline(pipeline, features_path, label=label)
    if result.y_true is None:
        raise ValueError(
            f"label column {label!r} missing from {features_path}; evaluation requires ground truth"
        )
    return summarize(result.y_true, result.y_pred)
