"""Multiclass classification metrics, mirroring what Spark MLlib's
``MulticlassMetrics`` produced for the original tedsds project.

The output of :func:`summarize` is a JSON-serialisable dict suitable for
storing in the Postgres ``models.metrics`` JSONB column.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def summarize(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> dict[str, Any]:
    """Compute the standard set of multiclass metrics."""
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_label = {
        int(lab): {
            "precision": float(
                precision_score(y_true, y_pred, labels=[lab], average="micro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, labels=[lab], average="micro", zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, labels=[lab], average="micro", zero_division=0)),
            "support": int((y_true == lab).sum()),
        }
        for lab in labels
    }

    return {
        "labels": [int(lab) for lab in labels],
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cast(list[list[int]], cm.tolist()),
        "per_label": per_label,
    }
