import json
from pathlib import Path

import numpy as np
import polars as pl

from ml.controller import MLController

FEATURE_QUERY = """
    SELECT session_key, driver_number, stint_number, circuit_short_name, lap_duration, tyre_life, compound,
           lap_number, track_temperature, team_name, year
    FROM laps
    WHERE track_temperature IS NOT NULL
"""

MODELS_DIR = Path(__file__).resolve().parent / "columns"
FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.json"
CIRCUIT_FEATURES_FILE = MODELS_DIR / "circuit_features.csv"
CIRCUIT_COLUMNS = ["circuit_lap_time"]
DEG_PRIOR_LAPS = 300

PRACTICE_QUERY = """
    SELECT session_key, meeting_key, circuit_short_name, driver_number, stint_number, compound, lap_number, lap_duration,
           tyre_life, year
    FROM practice_laps
"""
LONG_RUN_LAPS = 8
FP_PRIOR_RUNS = 3

LAP_TIME = "lap_duration"
TARGET = "lap_delta"
FUEL_SECONDS_PER_LAP = 0.05
COMPOUND_ORDER = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
TEAM_ALIASES = {"Kick Sauber": "Audi", "RB": "Racing Bulls"}  


class MLfeatures(MLController):
    def load_laps(self) -> pl.DataFrame:
        return self.fetch_data(FEATURE_QUERY)

    def encode(self, laps_df: pl.DataFrame) -> pl.DataFrame:
        return (
            laps_df.with_columns(
                compound=pl.col("compound").replace_strict(COMPOUND_ORDER),
                team_name=pl.col("team_name").replace(TEAM_ALIASES),
            )
            .to_dummies("team_name")
        )

    def fuel_corrected(self) -> pl.Expr:
        laps_remaining = pl.col("lap_number").max().over("session_key") - pl.col("lap_number")
        return pl.col(LAP_TIME) - FUEL_SECONDS_PER_LAP * laps_remaining

    def add_race_delta(self, laps_df: pl.DataFrame) -> pl.DataFrame:
        corrected = self.fuel_corrected()
        return laps_df.with_columns((corrected - corrected.median().over("session_key", "driver_number")).alias(TARGET))

    def circuit_features(self, laps_df: pl.DataFrame, test_year: int) -> pl.DataFrame:
        train = laps_df.filter(pl.col("year") < test_year).with_columns(corrected=self.fuel_corrected())
        stint = ["session_key", "driver_number", "stint_number"]
        demeaned = train.with_columns(
            dx=pl.col("tyre_life") - pl.col("tyre_life").mean().over(stint),
            dy=pl.col("corrected") - pl.col("corrected").mean().over(stint),
        )
        slope = ((pl.col("dx") * pl.col("dy")).sum() / (pl.col("dx") ** 2).sum())
        compound_deg = demeaned.group_by("compound").agg(compound_deg=slope)
        circuit_deg = (
            demeaned.group_by("circuit_short_name", "compound")
            .agg(raw_deg=slope, n_laps=pl.len())
            .join(compound_deg, on="compound")
            .with_columns(weight=pl.col("n_laps") / (pl.col("n_laps") + DEG_PRIOR_LAPS))
            .select("circuit_short_name", "compound",
                    circuit_deg=pl.col("weight") * pl.col("raw_deg").fill_nan(None).fill_null(pl.col("compound_deg"))
                    + (1 - pl.col("weight")) * pl.col("compound_deg"))
        )
        lap_time = laps_df.group_by("circuit_short_name").agg(
            circuit_lap_time=pl.when(pl.col("year") < test_year).then(pl.col(LAP_TIME)).median()
            .fill_null(pl.col(LAP_TIME).median())
        )
        return (
            lap_time.join(compound_deg, how="cross")
            .join(circuit_deg, on=["circuit_short_name", "compound"], how="left")
            .with_columns(pl.col("circuit_deg").fill_null(pl.col("compound_deg")))
            .select("circuit_short_name", "compound", "circuit_lap_time", "circuit_deg")
            .sort("circuit_short_name", "compound")
        )

    def long_runs(self, practice_df: pl.DataFrame) -> pl.DataFrame:
        stint = ["session_key", "driver_number", "stint_number"]
        # every lap burns fuel and gets ~0.05 s faster: add it back so the slope is tyre wear only
        return (
            practice_df.with_columns(corrected=pl.col(LAP_TIME) + FUEL_SECONDS_PER_LAP * pl.col("lap_number"))
            .with_columns(
                dx=pl.col("tyre_life") - pl.col("tyre_life").mean().over(stint),
                dy=pl.col("corrected") - pl.col("corrected").mean().over(stint),
            )
            .group_by(*stint, "meeting_key", "circuit_short_name", "compound", "year")
            .agg(
                n_laps=pl.len(),
                span=pl.col("tyre_life").max() - pl.col("tyre_life").min(),
                deg=(pl.col("dx") * pl.col("dy")).sum() / (pl.col("dx") ** 2).sum(),
            )
            .filter(pl.col("n_laps") >= LONG_RUN_LAPS, pl.col("span") >= LONG_RUN_LAPS - 1)
        )

    def practice_features(self, practice_df: pl.DataFrame, test_year: int) -> pl.DataFrame:
        runs = self.long_runs(practice_df)
        prior = runs.filter(pl.col("year") < test_year).group_by("compound").agg(compound_deg=pl.col("deg").median())
        weekend = runs.group_by("meeting_key", "compound").agg(raw_deg=pl.col("deg").median(), n_runs=pl.len())
        return (
            runs.select("meeting_key", "circuit_short_name", "year").unique()
            .join(prior, how="cross")
            .join(weekend, on=["meeting_key", "compound"], how="left")
            .with_columns(weight=pl.col("n_runs").fill_null(0) / (pl.col("n_runs").fill_null(0) + FP_PRIOR_RUNS))
            .select(
                "meeting_key", "circuit_short_name", "year", "compound",
                fp_deg=pl.col("weight") * pl.col("raw_deg").fill_null(0) + (1 - pl.col("weight")) * pl.col("compound_deg"),
                n_runs=pl.col("n_runs").fill_null(0),
                compound_deg="compound_deg",
            )
            .sort("meeting_key", "compound")
        )

    def add_circuit_features(self, laps_df: pl.DataFrame, circuits: pl.DataFrame) -> pl.DataFrame:
        return laps_df.join(circuits.select("circuit_short_name", "compound", *CIRCUIT_COLUMNS),
                            on=["circuit_short_name", "compound"], how="left", maintain_order="left")

    def load_practice(self) -> pl.DataFrame:
        return self.fetch_data(PRACTICE_QUERY)

    def build_features(self, laps_df: pl.DataFrame, test_year: int = 2026) -> tuple[pl.DataFrame, pl.Series]:
        self.circuits = self.circuit_features(laps_df, test_year)
        encoded_df = self.encode(self.add_race_delta(self.add_circuit_features(laps_df, self.circuits)))
        y = encoded_df[TARGET]
        x = encoded_df.drop(TARGET, LAP_TIME, "session_key", "driver_number", "stint_number", "circuit_short_name")
        self.check_features(x)
        return x, y

    def split_by_season(self, laps_df: pl.DataFrame, x: pl.DataFrame, y: pl.Series, test_year: int = 2026):
        train = x["year"] < test_year

        x_train, y_train = x.filter(train), y.filter(train)
        x_test, y_test = x.filter(~train), y.filter(~train)

        train_races = set(laps_df.filter(train)["session_key"])
        test_races = set(laps_df.filter(~train)["session_key"])

        print(f"Train: {x_train.height} laps from {len(train_races)} races (before {test_year})")
        print(f"Test: {x_test.height} laps from {len(test_races)} races ({test_year} onwards)")
        if train_races & test_races:
            raise ValueError(f"Races in both train and test: {train_races & test_races}")

        return x_train, x_test, y_train, y_test


    def check_features(self, x: pl.DataFrame):
        null_cols = [col for col, n in zip(x.columns, x.null_count().row(0)) if n > 0]
        if null_cols:
            raise ValueError(f"Features with nulls: {null_cols}")
        non_numeric = [col for col, dtype in x.schema.items() if not dtype.is_numeric()]
        if non_numeric:
            raise ValueError(f"Non-numeric features: {non_numeric}")

    def save_feature_columns(self, x: pl.DataFrame, path: Path = FEATURE_COLUMNS_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(x.columns, indent=2))
        print(f"{len(x.columns)} feature columns saved to {path}")
        self.circuits.write_csv(CIRCUIT_FEATURES_FILE)
        print(f"Circuit features of {self.circuits['circuit_short_name'].n_unique()} circuits saved to {CIRCUIT_FEATURES_FILE}")

    def load_feature_columns(self, path: Path = FEATURE_COLUMNS_FILE) -> list[str]:
        return json.loads(path.read_text())

    def load_circuit_features(self, path: Path = CIRCUIT_FEATURES_FILE) -> pl.DataFrame:
        return pl.read_csv(path)

    def run(self, test_year: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        laps_df = self.load_laps()
        x, y = self.build_features(laps_df, test_year)
        self.save_feature_columns(x)
        x_train, x_test, y_train, y_test = self.split_by_season(laps_df, x, y, test_year)
        return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
