"""KMeans clustering of NASA turbofan operational settings into op-modes.

The dataset has three operational settings per cycle. The original Spark/MLlib
project clustered them into k=6 ``operationmode`` labels and carried that
column through the feature table (without using it in the windowed
aggregations). We reproduce the same intent here with scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import duckdb
import joblib  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

from tedsds.sensors import SETTINGS


@dataclass(frozen=True)
class OpModeModel:
    """A fitted KMeans model plus the inputs used to fit it."""

    kmeans: KMeans
    feature_columns: tuple[str, ...] = SETTINGS

    def predict(self, settings: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict cluster id for each row of ``settings``."""
        return cast(NDArray[np.int64], self.kmeans.predict(settings).astype(np.int64))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> OpModeModel:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"expected {cls.__name__} at {path}, got {type(obj).__name__}")
        return obj


def fit(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    source: str = "pg.public.sensor_readings",
    k: int = 6,
    random_state: int = 42,
    n_init: int = 10,
) -> OpModeModel:
    """Fit KMeans on the settings columns of ``run_id``."""
    cols = ", ".join(SETTINGS)
    df = con.execute(f"SELECT {cols} FROM {source} WHERE run_id = ?", [run_id]).df()
    arr = df.to_numpy(dtype=np.float64)
    if arr.shape[0] < k:
        raise ValueError(f"need at least {k} rows to fit KMeans(k={k}), got {arr.shape[0]}")

    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    model.fit(arr)
    return OpModeModel(kmeans=model)


def predict_table(
    con: duckdb.DuckDBPyConnection,
    model: OpModeModel,
    *,
    run_id: str,
    source: str = "pg.public.sensor_readings",
) -> pa.Table:
    """Return a table of ``(id, cycle, operationmode)`` for ``run_id``."""
    cols = ", ".join(SETTINGS)
    df = con.execute(
        f"SELECT id, cycle, {cols} FROM {source} WHERE run_id = ? ORDER BY id, cycle",
        [run_id],
    ).df()
    settings = df[list(SETTINGS)].to_numpy(dtype=np.float64)
    op_mode = model.predict(settings)
    return pa.table(
        {
            "id": pa.array(df["id"].to_numpy()),
            "cycle": pa.array(df["cycle"].to_numpy()),
            "operationmode": pa.array(op_mode),
        }
    )
