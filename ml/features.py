import json
from pathlib import Path

import numpy as np
import polars as pl

from ml.controller import MLController

FEATURE_QUERY = """
    SELECT session_key, lap_duration, tyre_life, compound, lap_number,
           track_temperature, team_name, year
    FROM laps
    WHERE track_temperature IS NOT NULL
"""

MODELS_DIR = Path(__file__).resolve().parent / "models"
FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.json"

TARGET = "lap_duration"
COMPOUND_ORDER = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
TEAM_ALIASES = {"Kick Sauber": "Audi", "RB": "Racing Bulls"} #renamed team    


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

    def build_features(self, laps_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
        encoded_df = self.encode(laps_df)
        y = encoded_df[TARGET]
        x = encoded_df.drop(TARGET, "session_key")
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

    def load_feature_columns(self, path: Path = FEATURE_COLUMNS_FILE) -> list[str]:
        return json.loads(path.read_text())

    def run(self, test_year: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        laps_df = self.load_laps()
        x, y = self.build_features(laps_df)
        self.save_feature_columns(x)
        x_train, x_test, y_train, y_test = self.split_by_season(laps_df, x, y, test_year)
        return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
