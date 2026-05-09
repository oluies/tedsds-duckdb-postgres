"""Feature-pipeline tests.

Two flavours of validation:

1. **Synthetic exact-value**: with hand-computed expected mean/sd/RUL/labels for
   the synthetic dataset, assert exact agreement.
2. **Pandas cross-check (optional)**: if ``$TEDSDS_FD001_TRAIN`` points to the
   real NASA FD001 train file, ingest it, run the DuckDB pipeline, and compare
   row-for-row against the same logic implemented with pandas — proving the SQL
   matches an independent reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pytest

from tedsds.db import duckdb_with_pg
from tedsds.features import build_features
from tedsds.ingest import ingest_readings
from tedsds.sensors import SENSORS

from .conftest import needs_pg

DATA_DIR = Path(__file__).resolve().parent / "data"


@needs_pg
def test_synthetic_features_have_correct_rul_labels_and_window(
    pg: psycopg.Connection, pg_dsn: str
) -> None:
    ingest_readings(
        pg,
        run_id="SYN-train",
        dataset="SYN",
        kind="train",
        path=DATA_DIR / "synthetic_train.txt",
    )

    con = duckdb_with_pg(pg_dsn)
    table = build_features(con, run_id="SYN-train", kind="train", window_rows=5)
    df = table.to_pandas()

    engine1 = df[df.id == 1].sort_values("cycle").reset_index(drop=True)
    assert engine1.cycle.tolist() == [1, 2, 3, 4, 5, 6]

    # rul = max(cycle) per id - cycle  → 6,5,4,3,2,1,0 ... wait max=6 so rul = 5,4,3,2,1,0
    assert engine1.rul.tolist() == [5, 4, 3, 2, 1, 0]
    # label1: rul <= 30 always true here
    assert engine1.label1.tolist() == [1] * 6
    # label2: rul <= 15 → 2; rul <= 30 → 1; else 0. All rul<=5 here so all 2.
    assert engine1.label2.tolist() == [2] * 6

    # s1 for engine 1 over cycles 1..6 is 1,2,3,4,5,6.
    # Window ROWS BETWEEN 4 PRECEDING AND CURRENT ROW (size up to 5):
    #   cycle 1 → [1]                    mean=1.0   sd=NaN  (sample sd of 1 point)
    #   cycle 2 → [1,2]                  mean=1.5   sd=stddev_samp = 0.7071...
    #   cycle 3 → [1,2,3]                mean=2.0   sd=1.0
    #   cycle 4 → [1,2,3,4]              mean=2.5   sd=1.2909944...
    #   cycle 5 → [1,2,3,4,5]            mean=3.0   sd=1.5811388...
    #   cycle 6 → [2,3,4,5,6]            mean=4.0   sd=1.5811388...
    expected_a1 = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    expected_sd1 = [
        float("nan"),
        np.std([1, 2], ddof=1),
        np.std([1, 2, 3], ddof=1),
        np.std([1, 2, 3, 4], ddof=1),
        np.std([1, 2, 3, 4, 5], ddof=1),
        np.std([2, 3, 4, 5, 6], ddof=1),
    ]
    np.testing.assert_allclose(engine1.a1.to_numpy(), expected_a1, rtol=1e-12)

    # First sd1 entry is NULL → NaN; compare with equal_nan
    np.testing.assert_allclose(
        engine1.sd1.to_numpy(dtype=float), expected_sd1, rtol=1e-12, equal_nan=True
    )

    # Engine 2: s1 cycles 1..3 are 10, 20, 30
    engine2 = df[df.id == 2].sort_values("cycle").reset_index(drop=True)
    np.testing.assert_allclose(engine2.a1.to_numpy(), [10.0, 15.0, 20.0], rtol=1e-12)


def _pandas_features(readings: pd.DataFrame, window: int) -> pd.DataFrame:
    """Reference implementation of the feature pipeline using pandas only."""
    df = readings.sort_values(["id", "cycle"]).reset_index(drop=True)
    df["rul"] = df.groupby("id").cycle.transform("max") - df.cycle
    df["label1"] = (df.rul <= 30).astype(int)
    df["label2"] = np.where(df.rul <= 15, 2, np.where(df.rul <= 30, 1, 0)).astype(int)
    g = df.groupby("id", sort=False)
    for i, s in enumerate(SENSORS, start=1):
        roll = g[s].rolling(window=window, min_periods=1)
        df[f"a{i}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"sd{i}"] = roll.std(ddof=1).reset_index(level=0, drop=True)
    return df


@needs_pg
@pytest.mark.skipif(
    not os.environ.get("TEDSDS_FD001_TRAIN"),
    reason="set TEDSDS_FD001_TRAIN=/path/to/train_FD001.txt(.gz) to run cross-check",
)
def test_fd001_matches_pandas_reference(pg: psycopg.Connection, pg_dsn: str) -> None:
    src = Path(os.environ["TEDSDS_FD001_TRAIN"])
    n = ingest_readings(pg, run_id="FD001-train", dataset="FD001", kind="train", path=src)
    assert n > 0

    con = duckdb_with_pg(pg_dsn)
    duck_df = build_features(con, run_id="FD001-train", kind="train", window_rows=5).to_pandas()

    raw = pd.read_csv(
        src,
        sep=r"\s+",
        header=None,
        names=["id", "cycle", "setting1", "setting2", "setting3", *SENSORS],
        compression="infer" if src.suffix == ".gz" else None,
    )
    pandas_df = _pandas_features(raw, window=5)

    assert len(duck_df) == len(pandas_df)
    duck_sorted = duck_df.sort_values(["id", "cycle"]).reset_index(drop=True)
    pd_sorted = pandas_df.sort_values(["id", "cycle"]).reset_index(drop=True)

    for col in ["rul", "label1", "label2"]:
        np.testing.assert_array_equal(duck_sorted[col].to_numpy(), pd_sorted[col].to_numpy())

    for i in range(1, 22):
        np.testing.assert_allclose(
            duck_sorted[f"a{i}"].to_numpy(dtype=float),
            pd_sorted[f"a{i}"].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
            err_msg=f"a{i} mismatch",
        )
        np.testing.assert_allclose(
            duck_sorted[f"sd{i}"].to_numpy(dtype=float),
            pd_sorted[f"sd{i}"].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
            err_msg=f"sd{i} mismatch",
        )
