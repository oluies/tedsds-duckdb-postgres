# tedsds-duckdb-postgres

Predictive maintenance on the NASA Turbofan Engine Degradation dataset, rewritten as
**Postgres + DuckDB + scikit-learn** — a port of the Spark/MLlib project at
[oluies/tedsds](https://github.com/oluies/tedsds).

## Architecture

```
CSV ──┬─► Postgres (source of truth, model registry, predictions)
      │       ▲
      │       │ ATTACH (postgres_scanner, predicate pushdown)
      ▼       │
   DuckDB ────┘
      │
      │ pyarrow (zero-copy)
      ▼
  scikit-learn ──► joblib bytea ──► Postgres `models` table
```

| Layer | Tool |
|---|---|
| Source of truth | Postgres 16 |
| Feature engineering | DuckDB 1.x (windowed mean/sd, RUL, labels) |
| ML | scikit-learn (KMeans, RandomForest, LogisticRegression) |
| Glue | PyArrow |
| Pkg mgmt | uv |
| CLI | Typer |

## Quickstart

```bash
# 1. Postgres
make pg-up                 # docker compose up -d postgres
make migrate               # apply migrations/

# 2. Python env
uv sync

# 3. Sanity check
uv run tedsds --help
make test
```

## Status

- [x] Phase 1 — repo bootstrap, Postgres schema, Docker, CI, Dependabot
- [ ] Phase 2 — CSV → Postgres ingest
- [ ] Phase 3 — DuckDB feature pipeline + parity test vs. Spark golden output
- [ ] Phase 4 — KMeans for op-modes
- [ ] Phase 5 — RandomForest + LogisticRegression training
- [ ] Phase 6 — Evaluator (confusion matrix, weighted F1)
- [ ] Phase 7 — Inference CLI + predictions table writer

## Reference

NASA Ames Prognostics Data Repository — Saxena & Goebel (2008),
"Turbofan Engine Degradation Simulation Data Set."

## License

MIT — see [LICENSE](LICENSE).
