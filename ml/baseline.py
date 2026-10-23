import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from ml.features import TARGET, MLfeatures


class MLbaseline(MLfeatures):
    def evaluate(self, name: str, y_true: pl.Series, y_pred: np.ndarray) -> dict:
        return {
            "model": name,
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
        }

    def baseline_mean(self, y_train: pl.Series, x_test: pl.DataFrame) -> np.ndarray:
        return np.full(x_test.height, y_train.mean())

    def baseline_compound_mean(self, x_train: pl.DataFrame, y_train: pl.Series, x_test: pl.DataFrame) -> np.ndarray:
        global_mean = y_train.mean()
        compound_means = x_train.select("compound").with_columns(y_train).group_by("compound").agg(pl.col(TARGET).mean())
        predictions = x_test.select("compound").join(compound_means, on="compound", how="left", maintain_order="left")
        return predictions[TARGET].fill_null(global_mean).to_numpy()

    def run(self, test_year: int = 2026) -> pl.DataFrame:
        laps_df = self.load_laps()
        x, y = self.build_features(laps_df)
        x_train, x_test, y_train, y_test = self.split_by_season(laps_df, x, y, test_year)

        results = pl.DataFrame([
            self.evaluate("Baseline A: global mean", y_test, self.baseline_mean(y_train, x_test)),
            self.evaluate("Baseline B: mean per compound", y_test, self.baseline_compound_mean(x_train, y_train, x_test)),
        ])
        print(results)
        return results
