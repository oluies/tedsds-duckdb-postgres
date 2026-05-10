from collections.abc import Iterator
from contextlib import contextmanager

import duckdb
import psycopg

from tedsds.config import settings


@contextmanager
def pg_conn(dsn: str | None = None) -> Iterator[psycopg.Connection]:
    """Yield a Postgres connection (autocommit off)."""
    with psycopg.connect(dsn or settings.pg_dsn) as conn:
        yield conn


def duckdb_with_pg(dsn: str | None = None, alias: str = "pg") -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with Postgres ATTACHed read-only."""
    con = duckdb.connect(":memory:")
    con.install_extension("postgres")
    con.load_extension("postgres")
    con.execute(f"ATTACH '{dsn or settings.pg_dsn}' AS {alias} (TYPE postgres, READ_ONLY)")
    return con


def duckdb_with_pg_rw(dsn: str | None = None, alias: str = "pg") -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with Postgres ATTACHed read-write."""
    con = duckdb.connect(":memory:")
    con.install_extension("postgres")
    con.load_extension("postgres")
    con.execute(f"ATTACH '{dsn or settings.pg_dsn}' AS {alias} (TYPE postgres)")
    return con


def dsn_from_conn(conn: psycopg.Connection) -> str:
    """Build a libpq-style DSN from an open psycopg connection.

    DuckDB's postgres extension accepts the same string. We rebuild it from
    ``conn.info`` rather than relying on ``conn.info.dsn``, which redacts the
    password.
    """
    info = conn.info
    parts: list[str] = []
    for key, val in (
        ("host", info.host),
        ("port", info.port),
        ("dbname", info.dbname),
        ("user", info.user),
        ("password", info.password),
    ):
        if val:
            parts.append(f"{key}={val}")
    return " ".join(parts)
