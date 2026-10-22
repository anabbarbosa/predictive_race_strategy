import subprocess
from pathlib import Path

import polars as pl
from sqlalchemy import Engine, text

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"  # run in file-name order

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MIN_LAPS = 10  # drivers/teams with fewer clean laps are left out of fastest/slowest


def tyre_degradation(laps_df: pl.DataFrame) -> pl.DataFrame:
    """Seconds lost per lap of tyre life, per race and compound (linear fit). Behind lap_vs_tyre_life."""
    return (
        laps_df.group_by("session_key", "circuit_short_name", "compound")
        .agg(
            deg_s_per_lap=pl.cov("tyre_life", "lap_duration") / pl.col("tyre_life").var(),
            n_laps=pl.len(),
        )
        .filter(pl.col("n_laps") >= 10)
    )


def driver_pace(laps_df: pl.DataFrame) -> pl.DataFrame:
    """Lap time summary per race and driver. Behind lap_per_driver."""
    return laps_df.group_by(
        "session_key", "circuit_short_name", "driver_number", "broadcast_name", "team_name"
    ).agg(
        avg_lap=pl.col("lap_duration").mean(),
        median_lap=pl.col("lap_duration").median(),
        n_laps=pl.len(),
    )


def team_pace(laps_df: pl.DataFrame) -> pl.DataFrame:
    """Average lap time per race and team. Behind average_lap_by_team."""
    return (
        laps_df.filter(pl.col("team_name").is_not_null())
        .group_by("session_key", "circuit_short_name", "team_name")
        .agg(avg_lap=pl.col("lap_duration").mean(), n_laps=pl.len())
    )


def stint_summary(stints_df: pl.DataFrame) -> pl.DataFrame:
    """Stint length summary per race and compound. Behind stint_by_compound."""
    return stints_df.group_by("session_key", "circuit_short_name", "compound").agg(
        n_stints=pl.len(),
        avg_length=pl.col("stint_length").mean(),
        median_length=pl.col("stint_length").median(),
        max_length=pl.col("stint_length").max(),
    )


def race_summary(
    team_df: pl.DataFrame, driver_df: pl.DataFrame, stint_df: pl.DataFrame, deg_df: pl.DataFrame
) -> pl.DataFrame:
    """One row per race comparing teams, drivers, stints and tyre degradation."""

    def fastest_slowest(df: pl.DataFrame, name_col: str, prefix: str) -> pl.DataFrame:
        name, avg = pl.col(name_col).sort_by("avg_lap"), pl.col("avg_lap").sort()
        return (
            df.filter(pl.col("n_laps") >= MIN_LAPS)
            .group_by("session_key")
            .agg(
                **{
                    f"fastest_{prefix}": name.first(),
                    f"fastest_{prefix}_avg": avg.first(),
                    f"slowest_{prefix}": name.last(),
                    f"slowest_{prefix}_avg": avg.last(),
                }
            )
            .with_columns(
                (pl.col(f"slowest_{prefix}_avg") - pl.col(f"fastest_{prefix}_avg")).alias(f"{prefix}_spread_s")
            )
        )

    totals = driver_df.group_by("session_key", "circuit_short_name").agg(
        n_drivers=pl.len(), n_laps=pl.col("n_laps").sum()
    )
    stints = stint_df.group_by("session_key").agg(
        **{f"avg_{c.lower()}_stint": pl.col("avg_length").filter(pl.col("compound") == c).first() for c in COMPOUNDS}
    )
    deg = deg_df.group_by("session_key").agg(
        **{f"{c.lower()}_deg_s_per_lap": pl.col("deg_s_per_lap").filter(pl.col("compound") == c).first() for c in COMPOUNDS}
    )
    return (
        totals.join(fastest_slowest(team_df, "team_name", "team"), on="session_key", how="left")
        .join(fastest_slowest(driver_df, "broadcast_name", "driver"), on="session_key", how="left")
        .join(stints, on="session_key", how="left")
        .join(deg, on="session_key", how="left")
        .sort("session_key")
    )


def save_results(engine: Engine, laps_df: pl.DataFrame, stints_df: pl.DataFrame, label: str | None = None) -> int:
    """Store this run's results in the analytics_* tables, tagged with a new run_id."""
    results = {
        "tyre_degradation": tyre_degradation(laps_df),
        "driver_pace": driver_pace(laps_df),
        "team_pace": team_pace(laps_df),
        "stint_summary": stint_summary(stints_df),
    }
    results["race_summary"] = race_summary(
        results["team_pace"], results["driver_pace"], results["stint_summary"], results["tyre_degradation"]
    )
    with engine.begin() as conn:
        for script in sorted(SCRIPTS_DIR.glob("*.sql")):
            conn.execute(text(script.read_text(encoding="utf-8")))
        run_id = conn.execute(
            text("INSERT INTO analytics_runs (label, git_commit) VALUES (:label, :commit) RETURNING run_id"),
            {"label": label, "commit": git_commit()},
        ).scalar_one()
        for name, df in results.items():
            df.with_columns(run_id=pl.lit(run_id)).write_database(
                f"analytics_{name}", connection=conn, if_table_exists="append"
            )
            print(f"Saved {df.height} rows to analytics_{name}")
    print(f"Analytics run {run_id} saved.")
    return run_id


def git_commit() -> str | None:
    """Short hash of HEAD, with '-dirty' if there are uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return None
