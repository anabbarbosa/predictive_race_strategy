CREATE TABLE IF NOT EXISTS analytics_driver_pace (
    run_id              BIGINT NOT NULL REFERENCES analytics_runs ON DELETE CASCADE,
    session_key         BIGINT NOT NULL,
    circuit_short_name  TEXT,
    driver_number       INT NOT NULL,
    broadcast_name      TEXT,
    team_name           TEXT,
    avg_lap             DOUBLE PRECISION,
    median_lap          DOUBLE PRECISION,
    n_laps              INT,
    PRIMARY KEY (run_id, session_key, driver_number)
);
