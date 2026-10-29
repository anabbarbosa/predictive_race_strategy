from pathlib import Path

import joblib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

from ml.features import CIRCUIT_COLUMNS, COMPOUND_ORDER
from ml.models import MLModels

TRAINED_DIR = Path(__file__).resolve().parent / "trained"
COMPOUND_COLORS = {"SOFT": "#FF0000", "MEDIUM": "#F504AD", "HARD": "#2600FF"}

RF_PARAMS = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5, 20, 50],
    "max_features": [1.0, 0.5, "sqrt"],
}
XGB_PARAMS = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.1],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "min_child_weight": [1, 5, 20],
}


class MLTuning(MLModels):

    def race_groups(self, laps_df: pl.DataFrame, x: pl.DataFrame, test_year: int) -> np.ndarray:
        return laps_df.filter(x["year"] < test_year)["session_key"].to_numpy()

    def tune(self, name: str, model, params: dict, x_train, y_train, groups, n_iter: int):
        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            n_iter=n_iter,
            cv=GroupKFold(n_splits=5),
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(x_train, y_train, groups=groups)
        print(f"{name}: CV MAE {-search.best_score_:.3f} with {search.best_params_}")
        return search.best_estimator_

    def monotone_xgb(self, tuned: XGBRegressor, columns: list[str], x_train, y_train) -> XGBRegressor:
        constraints = tuple(1 if col == "tyre_life" else 0 for col in columns)
        model = XGBRegressor(**{**tuned.get_params(), "monotone_constraints": constraints, "n_jobs": -1})
        return model.fit(x_train, y_train)

    def tyre_life_limits(self, laps_df: pl.DataFrame, test_year: int, quantile: float = 0.95) -> dict[str, float]:
        limits = (
            laps_df.filter(pl.col("year") < test_year)
            .group_by("compound")
            .agg(pl.col("tyre_life").quantile(quantile))
        )
        return dict(limits.iter_rows())

    def degradation_grid(self, columns: list[str], lap_number: int, track_temperature: float, year: int, team: str,
                         circuit_values: dict[str, float]) -> pl.DataFrame:
        team_columns = [col for col in columns if col.startswith("team_name_")]
        return (
            pl.DataFrame({"tyre_life": range(1, 41)})
            .join(pl.DataFrame({"compound": list(COMPOUND_ORDER.values())}), how="cross")
            .with_columns(
                pl.lit(lap_number).alias("lap_number"),
                pl.lit(track_temperature).alias("track_temperature"),
                pl.lit(year).alias("year"),
                *[pl.lit(int(col == f"team_name_{team}")).alias(col) for col in team_columns],
                *[pl.lit(value).alias(col) for col, value in circuit_values.items()],
            )
            .select(columns)
        )

    def plot_feature_importance(self, models: dict, columns: list[str], output_dir: str = "plots"):
        fig, axes = plt.subplots(1, len(models), figsize=(12, 6))
        for ax, (name, model) in zip(axes, models.items()):
            importance = pl.DataFrame({"feature": columns, "importance": model.feature_importances_}).sort("importance")
            ax.barh(importance["feature"], importance["importance"])
            ax.set_xlabel("Importance")
            ax.set_title(name)
        fig.suptitle("Feature importance (tuned models)")
        file_path = Path(output_dir) / "feature_importance.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {file_path}")

    def plot_degradation(self, models: dict, grid: pl.DataFrame, limits: dict[str, float], output_dir: str = "plots"):
        tyre_life = grid["tyre_life"].to_numpy()
        compound = grid["compound"].to_numpy()

        fig, axes = plt.subplots(1, len(models), figsize=(15, 5), sharey=True)
        for ax, (name, model) in zip(axes, models.items()):
            pred = model.predict(grid.to_numpy())
            for label, code in COMPOUND_ORDER.items():
                rows = compound == code
                ax.plot(tyre_life[rows], pred[rows], label=label, color=COMPOUND_COLORS[label])
                ax.axvline(limits[label], color=COMPOUND_COLORS[label], linestyle=":", linewidth=1)
            ax.set_xlabel("Tyre life (laps)")
            ax.set_title(name)
        axes[0].set_ylabel("Predicted lap delta to race median (s)")
        axes[0].legend(title="Compound (dotted = p95 tyre life in training)")
        fig.suptitle("Predicted degradation curves (lap 40, McLaren, fixed temperature)")
        file_path = Path(output_dir) / "degradation_curves.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {file_path}")

    def save_models(self, models: dict[str, object]):
        TRAINED_DIR.mkdir(parents=True, exist_ok=True)
        for file_name, model in models.items():
            path = TRAINED_DIR / file_name
            if isinstance(model, XGBRegressor):
                model.save_model(path)
            else:
                joblib.dump(model, path, compress=3)
            print(f"Saved {path}")

    def run(self, test_year: int = 2026, n_iter: int = 20) -> pl.DataFrame:
        laps_df = self.load_laps()
        x, y = self.build_features(laps_df, test_year)
        self.save_feature_columns(x)
        x_train, x_test, y_train, y_test = self.split_by_season(laps_df, x, y, test_year)
        columns = x.columns
        groups = self.race_groups(laps_df, x, test_year)

        np_x_train, np_y_train = self.convert_numpy(x_train, y_train)
        np_x_test = x_test.to_numpy()

        pred, xg_pred = self.predict(np_x_train, np_y_train, np_x_test)
        rf = self.tune("Random Forest", RandomForestRegressor(random_state=42, n_jobs=1), RF_PARAMS, np_x_train, np_y_train, groups, n_iter)
        xgb = self.tune("XGBoost", XGBRegressor(random_state=42, n_jobs=1), XGB_PARAMS, np_x_train, np_y_train, groups, n_iter)
        xgb_mono = self.monotone_xgb(xgb, columns, np_x_train, np_y_train)

        results = pl.DataFrame([
            self.evaluate("Baseline B: mean per compound", y_test, self.baseline_compound_mean(x_train, y_train, x_test)),
            *self.add_table(y_test, pred, xg_pred),
            self.evaluate("Random Forest (tuned)", y_test, rf.predict(np_x_test)),
            self.evaluate("XGBoost (tuned)", y_test, xgb.predict(np_x_test)),
            self.evaluate("XGBoost (tuned, monotone)", y_test, xgb_mono.predict(np_x_test)),
        ])
        print(results)

        tuned = {"Random Forest (tuned)": rf, "XGBoost (tuned)": xgb}
        self.plot_feature_importance(tuned, columns)
        grid = self.degradation_grid(columns, lap_number=40, track_temperature=x_train["track_temperature"].median(),
                                     year=test_year - 1, team="McLaren",
                                     circuit_values={col: x_train[col].median() for col in CIRCUIT_COLUMNS})
        self.plot_degradation({**tuned, "XGBoost (tuned, monotone)": xgb_mono}, grid, self.tyre_life_limits(laps_df, test_year))

        self.save_models({"random_forest.joblib": rf, "xgboost.json": xgb, "xgboost_monotone.json": xgb_mono})
        return results
