CREATE TABLE IF NOT EXISTS analytics_team_pace (
    run_id              BIGINT NOT NULL REFERENCES analytics_runs ON DELETE CASCADE,
    session_key         BIGINT NOT NULL,
    circuit_short_name  TEXT,
    team_name           TEXT NOT NULL,
    avg_lap             DOUBLE PRECISION,
    n_laps              INT,
    PRIMARY KEY (run_id, session_key, team_name)
);
