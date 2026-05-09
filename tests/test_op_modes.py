"""Tests for the KMeans op-modes module.

Two layers:

1. Pure scikit-learn / joblib tests that need no Postgres (most of the value).
2. End-to-end test that goes through ``ATTACH postgres`` and the features
   pipeline — gated on ``@needs_pg``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import psycopg
import pyarrow as pa
import pytest

from tedsds.features import build_features
from tedsds.ingest import ingest_readings
from tedsds.op_modes import OpModeModel, fit, predict_table

from .conftest import needs_pg

DATA_DIR = Path(__file__).resolve().parent / "data"


def _synthetic_with_clusters() -> tuple[pa.Table, np.ndarray]:
    """6 cluster centres, 30 points (5 per centre, tiny noise)."""
    rng = np.random.default_rng(0)
    centres = np.array(
        [
            [0.0, 0.0, 100.0],
            [10.0, 0.0, 100.0],
            [0.0, 10.0, 100.0],
            [10.0, 10.0, 100.0],
            [5.0, 5.0, 90.0],
            [5.0, 5.0, 110.0],
        ]
    )
    rows = []
    truth = []
    for c_idx, c in enumerate(centres):
        for _ in range(5):
            rows.append(c + rng.normal(0, 0.01, size=3))
            truth.append(c_idx)
    arr = np.array(rows)
    return (
        pa.table(
            {
                "run_id": ["X"] * len(arr),
                "id": list(range(1, len(arr) + 1)),
                "cycle": [1] * len(arr),
                "setting1": arr[:, 0],
                "setting2": arr[:, 1],
                "setting3": arr[:, 2],
            }
        ),
        np.array(truth),
    )


def test_fit_recovers_well_separated_clusters() -> None:
    table, truth = _synthetic_with_clusters()
    con = duckdb.connect(":memory:")
    con.register("readings", table)
    con.execute("CREATE VIEW vr AS SELECT * FROM readings")

    model = fit(con, run_id="X", source="vr", k=6, random_state=42)

    assert model.kmeans.cluster_centers_.shape == (6, 3)

    settings = table.select(["setting1", "setting2", "setting3"]).to_pandas().to_numpy()
    pred = model.predict(settings)

    # KMeans is permutation-invariant in cluster ids; check that all rows in a
    # true cluster get the same predicted id, and that there are 6 distinct ids.
    assert len(np.unique(pred)) == 6
    for true_id in np.unique(truth):
        same = pred[truth == true_id]
        assert len(np.unique(same)) == 1, f"cluster {true_id} split across predictions"


def test_fit_is_deterministic_with_seed() -> None:
    table, _ = _synthetic_with_clusters()
    con = duckdb.connect(":memory:")
    con.register("readings", table)

    a = fit(con, run_id="X", source="readings", k=6, random_state=42)
    b = fit(con, run_id="X", source="readings", k=6, random_state=42)

    np.testing.assert_array_equal(a.kmeans.labels_, b.kmeans.labels_)
    np.testing.assert_allclose(a.kmeans.cluster_centers_, b.kmeans.cluster_centers_)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    table, _ = _synthetic_with_clusters()
    con = duckdb.connect(":memory:")
    con.register("readings", table)

    model = fit(con, run_id="X", source="readings", k=6, random_state=42)
    path = tmp_path / "op_modes.joblib"
    model.save(path)

    loaded = OpModeModel.load(path)
    assert loaded.feature_columns == model.feature_columns

    settings = table.select(["setting1", "setting2", "setting3"]).to_pandas().to_numpy()
    np.testing.assert_array_equal(loaded.predict(settings), model.predict(settings))


def test_fit_rejects_too_few_rows() -> None:
    table = pa.table(
        {
            "run_id": ["X"] * 3,
            "setting1": [1.0, 2.0, 3.0],
            "setting2": [1.0, 2.0, 3.0],
            "setting3": [1.0, 2.0, 3.0],
        }
    )
    con = duckdb.connect(":memory:")
    con.register("readings", table)
    with pytest.raises(ValueError, match="need at least 6 rows"):
        fit(con, run_id="X", source="readings", k=6)


@needs_pg
def test_features_includes_operationmode_when_model_provided(
    pg: psycopg.Connection, pg_dsn: str
) -> None:
    ingest_readings(
        pg,
        run_id="SYN-train",
        dataset="SYN",
        kind="train",
        path=DATA_DIR / "synthetic_train.txt",
    )

    from tedsds.db import duckdb_with_pg

    con = duckdb_with_pg(pg_dsn)
    # Synthetic file has only 9 rows, so use k=2 here.
    model = fit(con, run_id="SYN-train", k=2, random_state=42)
    op_modes = predict_table(con, model, run_id="SYN-train")

    table = build_features(con, run_id="SYN-train", kind="train", window_rows=5, op_modes=op_modes)
    assert "operationmode" in table.column_names

    # Without the model: column absent.
    table_plain = build_features(con, run_id="SYN-train", kind="train", window_rows=5)
    assert "operationmode" not in table_plain.column_names
