-- Initial schema for tedsds-duckdb-postgres.

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    dataset     TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('train', 'test')),
    source_uri  TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    run_id   TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    id       INT  NOT NULL,
    cycle    INT  NOT NULL,
    setting1 DOUBLE PRECISION,
    setting2 DOUBLE PRECISION,
    setting3 DOUBLE PRECISION,
    s1  DOUBLE PRECISION, s2  DOUBLE PRECISION, s3  DOUBLE PRECISION, s4  DOUBLE PRECISION,
    s5  DOUBLE PRECISION, s6  DOUBLE PRECISION, s7  DOUBLE PRECISION, s8  DOUBLE PRECISION,
    s9  DOUBLE PRECISION, s10 DOUBLE PRECISION, s11 DOUBLE PRECISION, s12 DOUBLE PRECISION,
    s13 DOUBLE PRECISION, s14 DOUBLE PRECISION, s15 DOUBLE PRECISION, s16 DOUBLE PRECISION,
    s17 DOUBLE PRECISION, s18 DOUBLE PRECISION, s19 DOUBLE PRECISION, s20 DOUBLE PRECISION,
    s21 DOUBLE PRECISION,
    PRIMARY KEY (run_id, id, cycle)
);

CREATE TABLE IF NOT EXISTS truth (
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    id              INT  NOT NULL,
    rul_at_maxcycle INT  NOT NULL,
    PRIMARY KEY (run_id, id)
);

CREATE TABLE IF NOT EXISTS models (
    model_id   UUID PRIMARY KEY,
    name       TEXT NOT NULL,
    algo       TEXT NOT NULL,
    params     JSONB NOT NULL,
    metrics    JSONB NOT NULL,
    artifact   BYTEA NOT NULL,
    trained_on TEXT NOT NULL REFERENCES runs(run_id),
    trained_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    model_id   UUID NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    run_id     TEXT NOT NULL REFERENCES runs(run_id)    ON DELETE CASCADE,
    id         INT  NOT NULL,
    cycle      INT  NOT NULL,
    label_pred INT  NOT NULL,
    proba      JSONB,
    PRIMARY KEY (model_id, run_id, id, cycle)
);
