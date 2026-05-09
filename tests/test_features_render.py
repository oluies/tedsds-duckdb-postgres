"""SQL-rendering tests that do not touch a database.

These guard against template-whitespace bugs (tokens accidentally fused) and
let local CI catch issues without a live Postgres.
"""

from __future__ import annotations

import re

import duckdb
import pytest

from tedsds.features import render_features_sql


def test_render_train_sql_has_no_fused_tokens() -> None:
    sql = render_features_sql(run_id="X-train", kind="train")

    # Every SQL keyword used in the template must appear surrounded by whitespace.
    for keyword in ("SELECT", "FROM", "WITH", "WHERE", "AS", "AVG", "STDDEV_SAMP"):
        assert re.search(rf"(?<!\w){keyword}(?!\w)", sql), f"{keyword!r} fused with neighbour"

    # No identifier directly followed by FROM/WHERE/AS — catches the
    # `sd21FROM labeled` bug we just fixed.
    assert not re.search(r"\w(FROM|WHERE|WINDOW)\b", sql), "identifier fused with keyword"


def test_render_emits_all_21_sensor_aggregates() -> None:
    sql = render_features_sql(run_id="X-train", kind="train")
    for i in range(1, 22):
        assert f"AS a{i}" in sql, f"missing a{i}"
        assert f"AS sd{i}" in sql, f"missing sd{i}"


def test_render_window_size_is_parameterised() -> None:
    sql_5 = render_features_sql(run_id="X-train", kind="train", window_rows=5)
    sql_10 = render_features_sql(run_id="X-train", kind="train", window_rows=10)
    assert "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW" in sql_5
    assert "ROWS BETWEEN 9 PRECEDING AND CURRENT ROW" in sql_10


@pytest.mark.parametrize("kind", ["train", "test"])
@pytest.mark.parametrize("with_op_modes", [False, True])
def test_rendered_sql_parses_in_duckdb(kind: str, with_op_modes: bool) -> None:
    """In-memory DuckDB parser smoke test — no Postgres required.

    We replace the FROM sources with empty inline tables so DuckDB can fully
    parse and bind the query. Catches any syntax bug the rendering may have
    introduced across the four (kind, with_op_modes) combinations.
    """
    sql = render_features_sql(
        run_id="X",
        kind=kind,  # type: ignore[arg-type]
        source="readings",
        truth_source="truth",
        op_modes_source="op_modes" if with_op_modes else None,
    )

    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE readings (run_id VARCHAR, id INT, cycle INT, "
        + ", ".join(f"setting{i} DOUBLE" for i in range(1, 4))
        + ", "
        + ", ".join(f"s{i} DOUBLE" for i in range(1, 22))
        + ")"
    )
    con.execute("CREATE TABLE truth (run_id VARCHAR, id INT, rul_at_maxcycle INT)")
    if with_op_modes:
        con.execute("CREATE TABLE op_modes (id INT, cycle INT, operationmode BIGINT)")
    con.execute(sql).fetchall()  # parses, binds, runs on empty data
