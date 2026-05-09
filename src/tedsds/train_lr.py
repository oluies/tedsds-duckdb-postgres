"""Logistic Regression (LBFGS) training wrapper.

Sklearn's ``LogisticRegression(solver='lbfgs')`` is the closest analogue to
Spark MLlib's ``LogisticRegressionWithLBFGS`` used in the original tedsds
project. ``multi_class='auto'`` selects multinomial for >2 classes.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from tedsds.training import LabelColumn, TrainResult, train_classifier


def train(
    features_path: Path,
    *,
    label: LabelColumn = "label2",
    max_iter: int = 200,
    C: float = 1.0,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> TrainResult:
    """Fit a LogisticRegression(solver='lbfgs') inside a MinMaxScaler pipeline."""
    estimator = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        C=C,
        random_state=seed,
    )
    return train_classifier(
        features_path,
        estimator=estimator,
        label=label,
        val_fraction=val_fraction,
        seed=seed,
    )
