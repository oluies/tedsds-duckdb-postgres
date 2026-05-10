"""Ingest tests.

Two layers:

1. **DuckDB-only parsing**: synthetic CSVs parsed via the same ``read_csv``
   expressions ``tedsds.ingest`` uses against Postgres. No live Postgres needed.
2. **Round-trip against live Postgres**: gated on ``@needs_pg``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import psycopg
import pytest

from tedsds.ingest import (
    _maybe_load_httpfs,
    ingest_readings,
    ingest_truth,
    sensor_read_csv_sql,
    truth_read_csv_sql,
)
from tedsds.sensors import READING_COLUMNS

from .conftest import needs_pg

DATA_DIR = Path(__file__).resolve().parent / "data"


# -- DuckDB-only parsing tests -----------------------------------------------


def test_sensor_read_csv_parses_26_columns(tmp_path: Path) -> None:
    csv = tmp_path / "train.txt"
    rows = [
        [1, 1, 0.0023, 0.0003, 100.0, *[float(i) for i in range(1, 22)]],
        [1, 2, 0.0010, -0.0001, 100.0, *[float(i) * 1.1 for i in range(1, 22)]],
        [2, 1, -0.0020, 0.0002, 100.0, *[float(i) * 0.9 for i in range(1, 22)]],
    ]
    csv.write_text("\n".join(" ".join(f"{v:g}" for v in r) for r in rows) + "\n")

    con = duckdb.connect(":memory:")
    cols = ", ".join(READING_COLUMNS)
    sql = f"SELECT {cols} FROM {sensor_read_csv_sql()}"
    result = con.execute(sql, {"path": str(csv)}).fetchall()

    assert len(result) == 3
    assert [r[0] for r in result] == [1, 1, 2]
    assert [r[1] for r in result] == [1, 2, 1]
    assert result[0][5:] == tuple(float(i) for i in range(1, 22))


def test_sensor_read_csv_tolerates_trailing_space(tmp_path: Path) -> None:
    csv = tmp_path / "trailing.txt"
    line = " ".join(["1", "1", "0.0", "0.0", "100.0", *[str(i) for i in range(1, 22)]])
    csv.write_text(line + " \n")

    con = duckdb.connect(":memory:")
    cols = ", ".join(READING_COLUMNS)
    sql = f"SELECT {cols} FROM {sensor_read_csv_sql()}"
    rows = con.execute(sql, {"path": str(csv)}).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][-1] == 21.0


def test_sensor_read_csv_handles_signed_and_scientific(tmp_path: Path) -> None:
    csv = tmp_path / "sci.txt"
    csv.write_text("1 1 -0.0023 1.5e-3 100.0 " + " ".join(["1.0"] * 21) + "\n")

    con = duckdb.connect(":memory:")
    sql = f"SELECT setting1, setting2, setting3 FROM {sensor_read_csv_sql()}"
    rows = con.execute(sql, {"path": str(csv)}).fetchall()
    assert rows == [(-0.0023, 0.0015, 100.0)]


def test_truth_read_csv_assigns_row_numbers(tmp_path: Path) -> None:
    csv = tmp_path / "RUL.txt"
    csv.write_text("112\n98\n69\n82\n")

    con = duckdb.connect(":memory:")
    sql = f"SELECT ROW_NUMBER() OVER () AS id, rul_at_maxcycle FROM {truth_read_csv_sql()}"
    rows = con.execute(sql, {"path": str(csv)}).fetchall()
    assert rows == [(1, 112), (2, 98), (3, 69), (4, 82)]


def test_ingest_readings_rejects_bad_kind() -> None:
    with pytest.raises(ValueError, match="kind must be 'train' or 'test'"):
        ingest_readings(
            None,  # type: ignore[arg-type]
            run_id="r",
            dataset="FD001",
            kind="validation",  # type: ignore[arg-type]
            path=Path("ignored.txt"),
        )


def test_maybe_load_httpfs_skips_local_paths() -> None:
    con = MagicMock(spec=duckdb.DuckDBPyConnection)
    _maybe_load_httpfs(con, "/tmp/train_FD001.txt")
    con.execute.assert_not_called()


def test_maybe_load_httpfs_loads_for_remote_schemes() -> None:
    for url in ("https://x/y.csv", "s3://b/k.csv", "gs://b/k.csv", "azure://b/k.csv"):
        con = MagicMock(spec=duckdb.DuckDBPyConnection)
        _maybe_load_httpfs(con, url)
        statements = [c.args[0] for c in con.execute.call_args_list]
        assert statements == ["INSTALL httpfs", "LOAD httpfs"], url


# -- Live Postgres round-trip tests ------------------------------------------


@needs_pg
def test_ingest_readings_inserts_rows(pg: psycopg.Connection) -> None:
    n = ingest_readings(
        pg,
        run_id="SYN-train",
        dataset="SYN",
        kind="train",
        path=DATA_DIR / "synthetic_train.txt",
    )
    assert n == 9

    with pg.cursor() as cur:
        cur.execute("SELECT count(*) FROM sensor_readings WHERE run_id = 'SYN-train'")
        (count,) = cur.fetchone()  # type: ignore[misc]
        assert count == 9

        cur.execute("SELECT count(DISTINCT id) FROM sensor_readings WHERE run_id = 'SYN-train'")
        (n_engines,) = cur.fetchone()  # type: ignore[misc]
        assert n_engines == 2


@needs_pg
def test_ingest_truth_assigns_ids_in_order(pg: psycopg.Connection, tmp_path: Path) -> None:
    truth_file = tmp_path / "rul.txt"
    truth_file.write_text("100\n50\n25\n")

    with pg.cursor() as cur:
        cur.execute("INSERT INTO runs (run_id, dataset, kind) VALUES ('SYN-test','SYN','test')")
    pg.commit()

    n = ingest_truth(pg, run_id="SYN-test", path=truth_file)
    assert n == 3

    with pg.cursor() as cur:
        cur.execute("SELECT id, rul_at_maxcycle FROM truth WHERE run_id = 'SYN-test' ORDER BY id")
        rows = cur.fetchall()
    assert rows == [(1, 100), (2, 50), (3, 25)]
