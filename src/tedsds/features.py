"""DuckDB feature pipeline for tedsds.

Reads sensor readings (and optionally truth) from Postgres via DuckDB's
``postgres`` extension, computes RUL, binary/multiclass labels, and a
windowed mean / sample stddev for each of the 21 sensor columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import duckdb
import jinja2
import pyarrow as pa

from tedsds.sensors import SENSORS, SETTINGS

RunKind = Literal["train", "test"]

_SQL_DIR = Path(__file__).resolve().parents[2] / "sql" / "features"
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_SQL_DIR),
    autoescape=False,
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_features_sql(
    *,
    run_id: str,
    kind: RunKind,
    source: str = "pg.public.sensor_readings",
    truth_source: str = "pg.public.truth",
    window_rows: int = 5,
    op_modes_source: str | None = None,
) -> str:
    """Render the feature-engineering SQL with the given parameters."""
    template = _jinja_env.get_template("build_features.sql.j2")
    return template.render(
        run_id=run_id,
        rul_source=kind,
        source=source,
        truth_source=truth_source,
        window_rows=window_rows,
        op_modes_source=op_modes_source,
        settings=SETTINGS,
        sensors=SENSORS,
    )


def build_features(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    kind: RunKind,
    source: str = "pg.public.sensor_readings",
    truth_source: str = "pg.public.truth",
    window_rows: int = 5,
    op_modes: pa.Table | None = None,
) -> pa.Table:
    """Run the feature pipeline and return the result as a pyarrow.Table.

    If ``op_modes`` is provided (columns ``id, cycle, operationmode``), it is
    registered as a temporary view and LEFT JOINed onto the output.
    """
    op_modes_source: str | None = None
    if op_modes is not None:
        con.register("_op_modes_view", op_modes)
        op_modes_source = "_op_modes_view"

    sql = render_features_sql(
        run_id=run_id,
        kind=kind,
        source=source,
        truth_source=truth_source,
        window_rows=window_rows,
        op_modes_source=op_modes_source,
    )
    try:
        return con.execute(sql).fetch_arrow_table()
    finally:
        if op_modes_source is not None:
            con.unregister(op_modes_source)
