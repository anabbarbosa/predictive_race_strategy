CREATE TABLE IF NOT EXISTS analytics_stint_summary (
    run_id              BIGINT NOT NULL REFERENCES analytics_runs ON DELETE CASCADE,
    session_key         BIGINT NOT NULL,
    circuit_short_name  TEXT,
    compound            TEXT NOT NULL,
    n_stints            INT,
    avg_length          DOUBLE PRECISION,
    median_length       DOUBLE PRECISION,
    max_length          INT,
    PRIMARY KEY (run_id, session_key, compound)
);
