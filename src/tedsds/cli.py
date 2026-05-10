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
) -> None:
    """Build the feature table via DuckDB and write it to Parquet."""
    if kind not in {"train", "test"}:
        raise typer.BadParameter("kind must be 'train' or 'test'")

    import pyarrow.parquet as pq

    from tedsds.db import duckdb_with_pg
    from tedsds.features import build_features

    con = duckdb_with_pg()
    table = build_features(con, run_id=run_id, kind=kind, window_rows=window_rows)  # type: ignore[arg-type]
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)  # type: ignore[no-untyped-call]
    typer.echo(f"wrote {table.num_rows} rows x {table.num_columns} cols to {out}")


if __name__ == "__main__":
    app()
