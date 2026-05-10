"""Shared test fixtures.

Tests that need Postgres expect a server reachable at ``$TEST_PG_DSN`` (defaults
to the local docker-compose Postgres). Each test gets a freshly migrated DB.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "migrations"

DEFAULT_DSN = "postgresql://tedsds:tedsds@localhost:5432/tedsds"


def _pg_dsn() -> str:
    return os.environ.get("TEST_PG_DSN", DEFAULT_DSN)


def _pg_available(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout):
        return False


needs_pg = pytest.mark.skipif(
    not _pg_available(_pg_dsn()),
    reason=f"Postgres not reachable at {_pg_dsn()}; set TEST_PG_DSN or run `make pg-up`",
)


def _terminate_other_backends(conn: psycopg.Connection) -> None:
    """Kill any other connections to this DB so DROP SCHEMA can proceed.

    A previous test that opened a DuckDB->Postgres ATTACH and didn't tear
    it down cleanly can leave a backend holding read locks indefinitely.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
        )
    conn.commit()


def _drop_all(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DO $$ BEGIN "
            "EXECUTE 'DROP SCHEMA public CASCADE'; "
            "EXECUTE 'CREATE SCHEMA public'; "
            "END $$;"
        )
    conn.commit()


def _apply_migrations(conn: psycopg.Connection) -> None:
    for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        with conn.cursor() as cur:
            cur.execute(sql_file.read_text())
    conn.commit()


@pytest.fixture
def pg() -> Iterator[psycopg.Connection]:
    """Yield a connection to a freshly migrated Postgres database."""
    dsn = _pg_dsn()
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        _terminate_other_backends(conn)
        _drop_all(conn)
        _apply_migrations(conn)
        yield conn


@pytest.fixture
def pg_dsn() -> str:
    return _pg_dsn()
