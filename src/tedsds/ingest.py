"""CSV → Postgres ingest powered by DuckDB extensions.

The NASA Turbofan files are space-separated text with no header. We read them
through DuckDB's native ``read_csv`` (with ``httpfs`` autoloaded for URLs and
gzip handled transparently) and stream rows straight into Postgres via the
``postgres`` extension's ATTACHed catalog — no intermediate Python rows.

The functions take an open ``psycopg.Connection`` for API parity with the
rest of the package; only the connection's DSN is used (DuckDB opens its
own connection underneath).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import duckdb
import psycopg

from tedsds.db import dsn_from_conn, duckdb_with_pg_rw
from tedsds.sensors import READING_COLUMNS

RunKind = Literal["train", "test"]

# Two trailing pad columns absorb the trailing whitespace some NASA dumps
# carry after the 26th value. They are read as VARCHAR and discarded.
_PAD_COLS: tuple[str, ...] = ("_pad1", "_pad2")


def _sensor_types() -> dict[str, str]:
    types: dict[str, str] = {"id": "INTEGER", "cycle": "INTEGER"}
    for col in READING_COLUMNS[2:]:
        types[col] = "DOUBLE"
    for col in _PAD_COLS:
        types[col] = "VARCHAR"
    return types


def _sql_str_list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(f"'{v}'" for v in values) + "]"


def _sql_str_map(mapping: dict[str, str]) -> str:
    return "{" + ", ".join(f"'{k}': '{v}'" for k, v in mapping.items()) + "}"


def sensor_read_csv_sql(path_param: str = "$path") -> str:
    """Build the ``read_csv`` expression that yields the 26 sensor columns."""
    names = READING_COLUMNS + _PAD_COLS
    return (
        f"read_csv({path_param}, delim=' ', header=false, "
        f"names={_sql_str_list(names)}, "
        f"types={_sql_str_map(_sensor_types())}, "
        f"null_padding=true, ignore_errors=false)"
    )


def truth_read_csv_sql(path_param: str = "$path") -> str:
    """Build the ``read_csv`` expression for a NASA RUL_*.txt truth file."""
    return f"read_csv({path_param}, header=false, columns={{'rul_at_maxcycle': 'INTEGER'}})"


def _maybe_load_httpfs(con: duckdb.DuckDBPyConnection, path: str) -> None:
    if path.startswith(("http://", "https://", "s3://", "gs://", "azure://")):
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")


def ingest_readings(
    conn: psycopg.Connection,
    *,
    run_id: str,
    dataset: str,
    kind: RunKind,
    path: Path,
) -> int:
    """Ingest a sensor-reading file into ``runs`` + ``sensor_readings`` via DuckDB.

    Returns the number of rows inserted into ``sensor_readings``.
    """
    if kind not in ("train", "test"):
        raise ValueError(f"kind must be 'train' or 'test', got {kind!r}")

    csv_path = str(path)
    sensor_cols_sql = ", ".join(READING_COLUMNS)
    insert_sensor_sql = f"""
        INSERT INTO pg.sensor_readings (run_id, {sensor_cols_sql})
        SELECT $run_id, {sensor_cols_sql}
        FROM {sensor_read_csv_sql()}
    """

    duck = duckdb_with_pg_rw(dsn_from_conn(conn))
    try:
        _maybe_load_httpfs(duck, csv_path)
        duck.execute("BEGIN")
        try:
            # DuckDB's postgres extension emits INSERTs as COPY, which
            # bypasses column DEFAULTs — supply ``ingested_at`` explicitly.
            duck.execute(
                "INSERT INTO pg.runs (run_id, dataset, kind, source_uri, ingested_at) "
                "VALUES ($run_id, $dataset, $kind, $source_uri, current_timestamp) "
                "ON CONFLICT (run_id) DO UPDATE SET source_uri = EXCLUDED.source_uri",
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "kind": kind,
                    "source_uri": csv_path,
                },
            )
            cur = duck.execute(insert_sensor_sql, {"run_id": run_id, "path": csv_path})
            row = cur.fetchone()
            inserted = int(row[0]) if row else 0
            duck.execute("COMMIT")
            return inserted
        except Exception:
            duck.execute("ROLLBACK")
            raise
    finally:
        duck.close()


def ingest_truth(
    conn: psycopg.Connection,
    *,
    run_id: str,
    path: Path,
) -> int:
    """Ingest a RUL truth file (one rul_at_maxcycle per id, in id order).

    The file holds one integer per line; line numbers (1-indexed) are the unit id.
    Returns the number of rows inserted.
    """
    csv_path = str(path)
    insert_sql = f"""
        INSERT INTO pg.truth (run_id, id, rul_at_maxcycle)
        SELECT $run_id, ROW_NUMBER() OVER () AS id, rul_at_maxcycle
        FROM {truth_read_csv_sql()}
    """

    duck = duckdb_with_pg_rw(dsn_from_conn(conn))
    try:
        _maybe_load_httpfs(duck, csv_path)
        duck.execute("BEGIN")
        try:
            cur = duck.execute(insert_sql, {"run_id": run_id, "path": csv_path})
            row = cur.fetchone()
            inserted = int(row[0]) if row else 0
            duck.execute("COMMIT")
            return inserted
        except Exception:
            duck.execute("ROLLBACK")
            raise
    finally:
        duck.close()
