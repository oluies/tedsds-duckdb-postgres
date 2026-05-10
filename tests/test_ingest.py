"""Ingest round-trip tests against a live Postgres."""

from __future__ import annotations

from pathlib import Path

import psycopg

from tedsds.ingest import ingest_readings, ingest_truth

from .conftest import needs_pg

DATA_DIR = Path(__file__).resolve().parent / "data"


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
