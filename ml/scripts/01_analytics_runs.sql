CREATE TABLE IF NOT EXISTS analytics_runs (
    run_id      BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    label       TEXT,
    git_commit  TEXT
);
