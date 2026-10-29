from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from ml.baseline import MLbaseline


class MLModels(MLbaseline):

    def convert_numpy(self, x: pl.DataFrame, y: pl.Series) -> tuple[np.ndarray, np.ndarray]:
        return x.to_numpy(), y.to_numpy()

    def predict(self, x_train, y_train, x_test) -> tuple[np.ndarray, np.ndarray]:
        regr = RandomForestRegressor(n_jobs=-1, random_state=42)
        pred = regr.fit(x_train, y_train).predict(x_test)

        xg_regr = XGBRegressor(random_state=42)
        xg_pred = xg_regr.fit(x_train, y_train).predict(x_test)

        return pred, xg_pred

    def add_table(self, y_test, pred, xg_pred) -> list[dict]:
        return [
            self.evaluate("Random Forest", y_test, pred),
            self.evaluate("XGBoost", y_test, xg_pred),
        ]

    def plot(self, x_test: pl.DataFrame, y_test: np.ndarray, predictions: dict[str, np.ndarray], output_dir: str = "plots"):
        """Save predicted vs actual and residuals vs tyre life, one panel per model."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tyre_life = x_test["tyre_life"].to_numpy()

        fig, axes = plt.subplots(1, len(predictions), figsize=(12, 6), sharex=True, sharey=True)
        for ax, (name, pred) in zip(axes, predictions.items()):
            ax.scatter(y_test, pred, s=2, alpha=0.3)
            ax.axline((0, 0), slope=1, color="black", linewidth=1)
            ax.set_xlabel("Actual lap delta to race median (s)")
            ax.set_ylabel("Predicted lap delta to race median (s)")
            ax.set_title(name)
        fig.suptitle("Predicted vs actual lap delta (test season)")
        plt.savefig(out / "predicted_vs_actual.png", bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, len(predictions), figsize=(12, 6), sharex=True, sharey=True)
        for ax, (name, pred) in zip(axes, predictions.items()):
            ax.scatter(tyre_life, y_test - pred, s=2, alpha=0.3)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_xlabel("Tyre life (laps)")
            ax.set_ylabel("Residual: actual - predicted (s)")
            ax.set_title(name)
        fig.suptitle("Residuals vs tyre life (test season)")
        plt.savefig(out / "residuals_vs_tyre_life.png", bbox_inches="tight")
        plt.close()

        print(f"Saved {out / 'predicted_vs_actual.png'} and {out / 'residuals_vs_tyre_life.png'}")

    def run(self, test_year: int = 2026) -> pl.DataFrame:
        laps_df = self.load_laps()
        x, y = self.build_features(laps_df, test_year)
        x_train, x_test, y_train, y_test = self.split_by_season(laps_df, x, y, test_year)

        np_x_train, np_y_train = self.convert_numpy(x_train, y_train)
        np_x_test, np_y_test = self.convert_numpy(x_test, y_test)
        pred, xg_pred = self.predict(np_x_train, np_y_train, np_x_test)

        results = pl.DataFrame([
            self.evaluate("Baseline A: global mean", y_test, self.baseline_mean(y_train, x_test)),
            self.evaluate("Baseline B: mean per compound", y_test, self.baseline_compound_mean(x_train, y_train, x_test)),
            *self.add_table(y_test, pred, xg_pred),
        ])
        print(results)

        self.plot(x_test, np_y_test, {"Random Forest": pred, "XGBoost": xg_pred})
        return results
