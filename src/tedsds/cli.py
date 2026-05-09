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


if __name__ == "__main__":
    app()
