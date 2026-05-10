"""Persist :class:`tedsds.predict.PredictResult` rows into the
Postgres ``predictions`` table.

Stored as ``(model_id, run_id, id, cycle, label_pred, proba)`` where ``proba``
is a JSONB object mapping each class label to its predicted probability,
e.g. ``{"0": 0.12, "1": 0.81, "2": 0.07}``.
"""

from __future__ import annotations

import uuid

import psycopg
from psycopg.types.json import Jsonb

from tedsds.predict import PredictResult


def write_predictions(
    conn: psycopg.Connection,
    result: PredictResult,
    *,
    model_id: uuid.UUID,
    run_id: str,
) -> int:
    """Bulk-insert predictions; returns rows written. Replaces any existing
    rows for ``(model_id, run_id)``."""
    proba = result.proba
    classes = result.classes

    rows: list[tuple[str, str, int, int, int, Jsonb | None]] = []
    for i in range(len(result.y_pred)):
        proba_obj: Jsonb | None = None
        if proba is not None:
            proba_obj = Jsonb({str(int(c)): float(proba[i, j]) for j, c in enumerate(classes)})
        rows.append(
            (
                str(model_id),
                run_id,
                int(result.ids[i]),
                int(result.cycles[i]),
                int(result.y_pred[i]),
                proba_obj,
            )
        )

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM predictions WHERE model_id = %s AND run_id = %s",
            (str(model_id), run_id),
        )
        cur.executemany(
            "INSERT INTO predictions (model_id, run_id, id, cycle, label_pred, proba) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )
    conn.commit()
    return len(rows)
