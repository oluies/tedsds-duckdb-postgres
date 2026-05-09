"""Random Forest training wrapper."""

from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

from tedsds.training import LabelColumn, TrainResult, train_classifier


def train(
    features_path: Path,
    *,
    label: LabelColumn = "label2",
    n_estimators: int = 100,
    max_depth: int = 10,
    val_fraction: float = 0.0,
    seed: int = 42,
    n_jobs: int = -1,
) -> TrainResult:
    """Fit a RandomForestClassifier inside a MinMaxScaler pipeline."""
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return train_classifier(
        features_path,
        estimator=estimator,
        label=label,
        val_fraction=val_fraction,
        seed=seed,
    )
