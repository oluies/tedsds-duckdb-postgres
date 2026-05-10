from pathlib import Path
from typing import Annotated

import typer

from tedsds import __version__

app = typer.Typer(help="tedsds — predictive maintenance on Postgres + DuckDB + scikit-learn")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def ping() -> None:
    """Verify connectivity to Postgres."""
    from tedsds.db import pg_conn

    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        (one,) = cur.fetchone()  # type: ignore[misc]
        typer.echo(f"postgres ok: SELECT 1 = {one}")


@app.command()
def ingest_readings(
    run_id: Annotated[str, typer.Option(help="Unique run identifier, e.g. 'FD001-train'")],
    dataset: Annotated[str, typer.Option(help="Dataset name, e.g. 'FD001'")],
    kind: Annotated[str, typer.Option(help="'train' or 'test'")],
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Ingest a NASA turbofan sensor file (.txt or .txt.gz) into Postgres."""
    if kind not in {"train", "test"}:
        raise typer.BadParameter("kind must be 'train' or 'test'")

    from tedsds.db import pg_conn
    from tedsds.ingest import ingest_readings as do_ingest

    with pg_conn() as conn:
        n = do_ingest(conn, run_id=run_id, dataset=dataset, kind=kind, path=path)  # type: ignore[arg-type]
    typer.echo(f"inserted {n} rows into sensor_readings for run_id={run_id}")


@app.command()
def ingest_truth(
    run_id: Annotated[str, typer.Option(help="Run identifier the truth belongs to")],
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Ingest a NASA turbofan RUL truth file."""
    from tedsds.db import pg_conn
    from tedsds.ingest import ingest_truth as do_ingest

    with pg_conn() as conn:
        n = do_ingest(conn, run_id=run_id, path=path)
    typer.echo(f"inserted {n} rows into truth for run_id={run_id}")


@app.command()
def features(
    run_id: Annotated[str, typer.Option(help="Run identifier to build features for")],
    kind: Annotated[str, typer.Option(help="'train' or 'test'")],
    out: Annotated[Path, typer.Option(help="Where to write the resulting Parquet file")],
    window_rows: Annotated[int, typer.Option(help="Window size including current row")] = 5,
    op_modes_model: Annotated[
        Path | None,
        typer.Option(
            help="Optional joblib path of a fitted OpModeModel; adds operationmode column"
        ),
    ] = None,
) -> None:
    """Build the feature table via DuckDB and write it to Parquet."""
    if kind not in {"train", "test"}:
        raise typer.BadParameter("kind must be 'train' or 'test'")

    import pyarrow.parquet as pq

    from tedsds.db import duckdb_with_pg
    from tedsds.features import build_features

    con = duckdb_with_pg()

    op_modes = None
    if op_modes_model is not None:
        from tedsds.op_modes import OpModeModel, predict_table

        model = OpModeModel.load(op_modes_model)
        op_modes = predict_table(con, model, run_id=run_id)

    table = build_features(
        con,
        run_id=run_id,
        kind=kind,  # type: ignore[arg-type]
        window_rows=window_rows,
        op_modes=op_modes,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)  # type: ignore[no-untyped-call]
    typer.echo(f"wrote {table.num_rows} rows x {table.num_columns} cols to {out}")


@app.command()
def train_op_modes(
    run_id: Annotated[str, typer.Option(help="Run identifier to fit on")],
    out: Annotated[Path, typer.Option(help="Joblib path to save the fitted model")],
    k: Annotated[int, typer.Option(help="Number of clusters")] = 6,
    seed: Annotated[int, typer.Option(help="random_state for reproducibility")] = 42,
) -> None:
    """Fit a KMeans model on the operational settings of ``run_id``."""
    from tedsds.db import duckdb_with_pg
    from tedsds.op_modes import fit

    con = duckdb_with_pg()
    model = fit(con, run_id=run_id, k=k, random_state=seed)
    model.save(out)
    inertia = model.kmeans.inertia_
    typer.echo(f"fitted KMeans(k={k}, seed={seed}) inertia={inertia:.4f} -> {out}")


def _persist_classifier(
    *,
    name: str,
    algo: str,
    params: dict[str, object],
    result: object,
    out: Path | None,
    register: bool,
    trained_on: str | None,
) -> None:
    import json

    import joblib  # type: ignore[import-untyped]

    from tedsds.training import TrainResult

    assert isinstance(result, TrainResult)
    metrics_summary: dict[str, object] = {"train": result.train_metrics}
    if result.val_metrics is not None:
        metrics_summary["val"] = result.val_metrics

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.pipeline, out)
        out.with_suffix(out.suffix + ".metrics.json").write_text(
            json.dumps(metrics_summary, indent=2)
        )

    if register:
        if trained_on is None:
            raise typer.BadParameter("--register requires --trained-on")
        from tedsds.db import pg_conn
        from tedsds.registry import save_model

        with pg_conn() as conn:
            model_id = save_model(
                conn,
                name=name,
                algo=algo,
                params=params,
                metrics=metrics_summary,
                pipeline=result.pipeline,
                trained_on=trained_on,
            )
        typer.echo(f"registered model_id={model_id}")

    train_acc = result.train_metrics["accuracy"]
    val_line = (
        "" if result.val_metrics is None else f"  val_accuracy={result.val_metrics['accuracy']:.4f}"
    )
    typer.echo(
        f"trained {algo} train_accuracy={train_acc:.4f}{val_line}  "
        f"n_train={result.n_train} n_val={result.n_val}"
    )


@app.command()
def train_rf(
    features: Annotated[Path, typer.Option(help="Features Parquet from `tedsds features`")],
    name: Annotated[str, typer.Option(help="Human-readable model name")] = "rf",
    label: Annotated[str, typer.Option(help="'label1' or 'label2'")] = "label2",
    n_estimators: Annotated[int, typer.Option()] = 100,
    max_depth: Annotated[int, typer.Option()] = 10,
    val_fraction: Annotated[float, typer.Option(help="Validation hold-out fraction")] = 0.2,
    seed: Annotated[int, typer.Option()] = 42,
    out: Annotated[Path | None, typer.Option(help="Optional joblib path")] = None,
    register: Annotated[bool, typer.Option("--register/--no-register")] = False,
    trained_on: Annotated[str | None, typer.Option(help="run_id for the model registry")] = None,
) -> None:
    """Train a RandomForestClassifier."""
    if label not in {"label1", "label2"}:
        raise typer.BadParameter("label must be 'label1' or 'label2'")

    from tedsds.train_rf import train

    result = train(
        features,
        label=label,  # type: ignore[arg-type]
        n_estimators=n_estimators,
        max_depth=max_depth,
        val_fraction=val_fraction,
        seed=seed,
    )
    _persist_classifier(
        name=name,
        algo="random_forest",
        params={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "label": label,
            "seed": seed,
            "val_fraction": val_fraction,
        },
        result=result,
        out=out,
        register=register,
        trained_on=trained_on,
    )


@app.command()
def train_lr(
    features: Annotated[Path, typer.Option(help="Features Parquet from `tedsds features`")],
    name: Annotated[str, typer.Option(help="Human-readable model name")] = "lr",
    label: Annotated[str, typer.Option(help="'label1' or 'label2'")] = "label2",
    max_iter: Annotated[int, typer.Option()] = 200,
    C: Annotated[float, typer.Option(help="Inverse regularisation strength")] = 1.0,
    val_fraction: Annotated[float, typer.Option(help="Validation hold-out fraction")] = 0.2,
    seed: Annotated[int, typer.Option()] = 42,
    out: Annotated[Path | None, typer.Option(help="Optional joblib path")] = None,
    register: Annotated[bool, typer.Option("--register/--no-register")] = False,
    trained_on: Annotated[str | None, typer.Option(help="run_id for the model registry")] = None,
) -> None:
    """Train a LogisticRegression (LBFGS) classifier."""
    if label not in {"label1", "label2"}:
        raise typer.BadParameter("label must be 'label1' or 'label2'")

    from tedsds.train_lr import train

    result = train(
        features,
        label=label,  # type: ignore[arg-type]
        max_iter=max_iter,
        C=C,
        val_fraction=val_fraction,
        seed=seed,
    )
    _persist_classifier(
        name=name,
        algo="logistic_regression",
        params={
            "max_iter": max_iter,
            "C": C,
            "label": label,
            "seed": seed,
            "val_fraction": val_fraction,
        },
        result=result,
        out=out,
        register=register,
        trained_on=trained_on,
    )


@app.command()
def evaluate(
    model_id: Annotated[str, typer.Option(help="Registered model UUID")],
    features: Annotated[Path, typer.Option(help="Features Parquet from `tedsds features`")],
    label: Annotated[str, typer.Option(help="'label1' or 'label2'")] = "label2",
    out: Annotated[
        Path | None, typer.Option(help="Optional JSON path for the metrics dict")
    ] = None,
) -> None:
    """Evaluate a registered model against a features parquet."""
    if label not in {"label1", "label2"}:
        raise typer.BadParameter("label must be 'label1' or 'label2'")

    import json
    import uuid as _uuid

    from tedsds.db import pg_conn
    from tedsds.evaluate import evaluate_pipeline
    from tedsds.registry import load_model

    with pg_conn() as conn:
        stored = load_model(conn, model_id=_uuid.UUID(model_id))

    metrics = evaluate_pipeline(stored.pipeline, features, label=label)  # type: ignore[arg-type]
    summary = json.dumps(metrics, indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary)
    typer.echo(
        f"model={stored.name} algo={stored.algo} "
        f"accuracy={metrics['accuracy']:.4f} weighted_f1={metrics['weighted_f1']:.4f} "
        f"n={metrics['n']}"
    )


@app.command()
def predict(
    model_id: Annotated[str, typer.Option(help="Registered model UUID")],
    features: Annotated[Path, typer.Option(help="Features Parquet from `tedsds features`")],
    run_id: Annotated[str, typer.Option(help="Run identifier to attribute predictions to")],
    label: Annotated[
        str, typer.Option(help="Label column to compare against, if present")
    ] = "label2",
    write: Annotated[
        bool, typer.Option("--write/--no-write", help="Persist to predictions table")
    ] = True,
    out: Annotated[Path | None, typer.Option(help="Optional Parquet path for predictions")] = None,
) -> None:
    """Run inference with a registered model and write to the predictions table."""
    if label not in {"label1", "label2"}:
        raise typer.BadParameter("label must be 'label1' or 'label2'")

    import uuid as _uuid

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tedsds.db import pg_conn
    from tedsds.predict import apply_pipeline
    from tedsds.predictions_table import write_predictions
    from tedsds.registry import load_model

    with pg_conn() as conn:
        stored = load_model(conn, model_id=_uuid.UUID(model_id))
        result = apply_pipeline(stored.pipeline, features, label=label)  # type: ignore[arg-type]

        n_written = 0
        if write:
            n_written = write_predictions(conn, result, model_id=stored.model_id, run_id=run_id)

    if out is not None:
        df = pd.DataFrame({"id": result.ids, "cycle": result.cycles, "label_pred": result.y_pred})
        if result.proba is not None:
            for j, cls in enumerate(result.classes):
                df[f"proba_{int(cls)}"] = result.proba[:, j]
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df), out)  # type: ignore[no-untyped-call]

    typer.echo(
        f"predicted {len(result.y_pred)} rows with model {stored.name} "
        f"(written={n_written}, parquet={'yes' if out else 'no'})"
    )


if __name__ == "__main__":
    app()
