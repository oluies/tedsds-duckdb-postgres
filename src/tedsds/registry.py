"""Postgres-backed model registry.

Stores fitted sklearn pipelines as joblib bytes in the ``models`` table
along with parameters and metrics as JSONB. Round-trips via
:func:`save_model` / :func:`load_model`.
"""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass
from typing import Any

import joblib  # type: ignore[import-untyped]
import psycopg
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class StoredModel:
    model_id: uuid.UUID
    name: str
    algo: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    pipeline: Any  # sklearn.pipeline.Pipeline
    trained_on: str


def _dump(pipeline: Any) -> bytes:
    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    return buf.getvalue()


def _load(blob: bytes | memoryview) -> Any:
    if isinstance(blob, memoryview):
        blob = bytes(blob)
    return joblib.load(io.BytesIO(blob))


def save_model(
    conn: psycopg.Connection,
    *,
    name: str,
    algo: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    pipeline: Any,
    trained_on: str,
) -> uuid.UUID:
    """Insert a model row and return its uuid."""
    model_id = uuid.uuid4()
    artifact = _dump(pipeline)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO models (model_id, name, algo, params, metrics, artifact, trained_on) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (str(model_id), name, algo, Jsonb(params), Jsonb(metrics), artifact, trained_on),
        )
    conn.commit()
    return model_id


def load_model(conn: psycopg.Connection, *, model_id: uuid.UUID) -> StoredModel:
    """Fetch a previously stored model by uuid."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT name, algo, params, metrics, artifact, trained_on FROM models "
            "WHERE model_id = %s",
            (str(model_id),),
        )
        row = cur.fetchone()
    if row is None:
        raise KeyError(f"no model with id {model_id}")

    name, algo, params, metrics, artifact, trained_on = row
    if isinstance(params, str):
        params = json.loads(params)
    if isinstance(metrics, str):
        metrics = json.loads(metrics)

    return StoredModel(
        model_id=model_id,
        name=name,
        algo=algo,
        params=params,
        metrics=metrics,
        pipeline=_load(artifact),
        trained_on=trained_on,
    )
