"""Postgres model-registry round-trip."""

from __future__ import annotations

import numpy as np
import psycopg
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import MinMaxScaler  # type: ignore[import-untyped]

from tedsds.registry import load_model, save_model

from .conftest import needs_pg


def _toy_pipeline() -> Pipeline:
    pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", LogisticRegression())])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    pipe.fit(X, y)
    return pipe


@needs_pg
def test_save_load_roundtrip_predicts_identically(pg: psycopg.Connection) -> None:
    with pg.cursor() as cur:
        cur.execute("INSERT INTO runs (run_id, dataset, kind) VALUES ('REG-train','REG','train')")
    pg.commit()

    pipe = _toy_pipeline()
    model_id = save_model(
        pg,
        name="test-lr",
        algo="logistic_regression",
        params={"C": 1.0},
        metrics={"train": {"accuracy": 0.9}},
        pipeline=pipe,
        trained_on="REG-train",
    )

    loaded = load_model(pg, model_id=model_id)
    assert loaded.name == "test-lr"
    assert loaded.algo == "logistic_regression"
    assert loaded.params == {"C": 1.0}
    assert loaded.trained_on == "REG-train"

    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 4))
    np.testing.assert_array_equal(pipe.predict(X), loaded.pipeline.predict(X))


@needs_pg
def test_load_unknown_id_raises(pg: psycopg.Connection) -> None:
    import uuid

    import pytest

    with pytest.raises(KeyError):
        load_model(pg, model_id=uuid.uuid4())
