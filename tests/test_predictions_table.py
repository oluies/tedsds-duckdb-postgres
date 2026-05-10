"""End-to-end test for writing predictions to Postgres."""

from __future__ import annotations

import json
from pathlib import Path

import psycopg

from tedsds.predict import apply_pipeline
from tedsds.predictions_table import write_predictions
from tedsds.registry import save_model
from tedsds.train_lr import train as train_lr

from .conftest import needs_pg
from .test_predict import _write_synthetic_features


@needs_pg
def test_write_predictions_round_trip(pg: psycopg.Connection, tmp_path: Path) -> None:
    features_path = tmp_path / "f.parquet"
    _write_synthetic_features(features_path)

    pipeline = train_lr(features_path, label="label2", val_fraction=0.0, seed=42).pipeline

    with pg.cursor() as cur:
        cur.execute("INSERT INTO runs (run_id, dataset, kind) VALUES ('PRED-train','PRED','train')")
    pg.commit()

    model_id = save_model(
        pg,
        name="lr-pred-test",
        algo="logistic_regression",
        params={"seed": 42},
        metrics={"train": {"accuracy": 1.0}},
        pipeline=pipeline,
        trained_on="PRED-train",
    )

    result = apply_pipeline(pipeline, features_path, label="label2")
    n_written = write_predictions(pg, result, model_id=model_id, run_id="PRED-train")
    assert n_written == len(result.y_pred)

    with pg.cursor() as cur:
        cur.execute(
            "SELECT id, cycle, label_pred, proba FROM predictions "
            "WHERE model_id = %s AND run_id = %s ORDER BY id, cycle",
            (str(model_id), "PRED-train"),
        )
        rows = cur.fetchall()

    assert len(rows) == n_written
    # Spot-check first row matches in-memory result
    rid, rcycle, rlabel, rproba_raw = rows[0]
    assert rid == int(result.ids[0])
    assert rcycle == int(result.cycles[0])
    assert rlabel == int(result.y_pred[0])

    rproba = rproba_raw if isinstance(rproba_raw, dict) else json.loads(rproba_raw)
    assert set(rproba.keys()) == {"0", "1", "2"}
    assert abs(sum(rproba.values()) - 1.0) < 1e-6


@needs_pg
def test_write_predictions_replaces_existing_rows(pg: psycopg.Connection, tmp_path: Path) -> None:
    features_path = tmp_path / "f.parquet"
    _write_synthetic_features(features_path)
    pipeline = train_lr(features_path, label="label2", val_fraction=0.0, seed=42).pipeline

    with pg.cursor() as cur:
        cur.execute("INSERT INTO runs (run_id, dataset, kind) VALUES ('PRED-train','PRED','train')")
    pg.commit()
    model_id = save_model(
        pg,
        name="lr-replace",
        algo="logistic_regression",
        params={},
        metrics={},
        pipeline=pipeline,
        trained_on="PRED-train",
    )

    result = apply_pipeline(pipeline, features_path, label="label2")
    write_predictions(pg, result, model_id=model_id, run_id="PRED-train")
    # Second write must not double the rows
    n_second = write_predictions(pg, result, model_id=model_id, run_id="PRED-train")

    with pg.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM predictions WHERE model_id = %s AND run_id = %s",
            (str(model_id), "PRED-train"),
        )
        (count,) = cur.fetchone()  # type: ignore[misc]
    assert count == n_second
