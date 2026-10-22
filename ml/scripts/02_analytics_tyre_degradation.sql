CREATE TABLE IF NOT EXISTS analytics_tyre_degradation (
    run_id              BIGINT NOT NULL REFERENCES analytics_runs ON DELETE CASCADE,
    session_key         BIGINT NOT NULL,
    circuit_short_name  TEXT,
    compound            TEXT NOT NULL,
    deg_s_per_lap       DOUBLE PRECISION,
    n_laps              INT,
    PRIMARY KEY (run_id, session_key, compound)
);
