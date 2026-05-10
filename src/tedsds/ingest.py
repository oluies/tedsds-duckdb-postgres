"""CSV (or .gz) → Postgres ingest for the NASA turbofan dataset.

The raw files are space-delimited with 26 columns for sensor readings and
2 columns for truth files. We use ``COPY ... FROM STDIN`` for speed.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import psycopg

from tedsds.sensors import READING_COLUMNS, TRUTH_COLUMNS

RunKind = Literal["train", "test"]


def _open(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield stripped


def _normalise_row(line: str, expected_cols: int) -> list[str]:
    parts = line.split()
    if len(parts) != expected_cols:
        raise ValueError(f"expected {expected_cols} columns, got {len(parts)}: {line!r}")
    return parts


def ingest_readings(
    conn: psycopg.Connection,
    *,
    run_id: str,
    dataset: str,
    kind: RunKind,
    path: Path,
) -> int:
    """Ingest a sensor-reading file into ``runs`` and ``sensor_readings``.

    Returns the number of rows inserted into ``sensor_readings``.
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO runs (run_id, dataset, kind, source_uri) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (run_id) DO UPDATE SET source_uri = EXCLUDED.source_uri",
            (run_id, dataset, kind, str(path)),
        )

    columns = ("run_id", *READING_COLUMNS)
    copy_sql = f"COPY sensor_readings ({', '.join(columns)}) FROM STDIN"

    n = 0
    with conn.cursor() as cur, cur.copy(copy_sql) as copy:
        for line in _open(path):
            parts = _normalise_row(line, expected_cols=len(READING_COLUMNS))
            copy.write_row((run_id, *parts))
            n += 1
    conn.commit()
    return n


def ingest_truth(
    conn: psycopg.Connection,
    *,
    run_id: str,
    path: Path,
) -> int:
    """Ingest a RUL truth file (one rul_at_maxcycle per id, in id order).

    The NASA truth files have a single column (RUL) with the engine ``id``
    being implicit (1-based row number). We infer it.
    """
    columns = ("run_id", *TRUTH_COLUMNS)
    copy_sql = f"COPY truth ({', '.join(columns)}) FROM STDIN"

    n = 0
    with conn.cursor() as cur, cur.copy(copy_sql) as copy:
        for engine_id, line in enumerate(_open(path), start=1):
            parts = line.split()
            if len(parts) != 1:
                raise ValueError(f"expected 1 column in truth file, got {len(parts)}: {line!r}")
            copy.write_row((run_id, str(engine_id), parts[0]))
            n += 1
    conn.commit()
    return n
