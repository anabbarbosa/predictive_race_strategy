from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap

from ml.features import COMPOUND_ORDER
from ml.simulate import MLSimulate
from ml.tuning import COMPOUND_COLORS

SAMPLE_LAPS = 5000


class MLExplain(MLSimulate):
    def shap_values(self, model, x: pl.DataFrame) -> shap.Explanation:
        return shap.TreeExplainer(model)(x.to_pandas())

    def importance(self, explanation: shap.Explanation, columns: list[str]) -> pl.DataFrame:
        return (
            pl.DataFrame({"feature": columns, "mean_abs_shap_s": np.abs(explanation.values).mean(axis=0)})
            .sort("mean_abs_shap_s", descending=True)
        )

    def plot_summary(self, explanation: shap.Explanation, output_dir: str = "plots"):
        shap.plots.beeswarm(explanation, max_display=12, show=False)
        plt.title("SHAP values of the monotone XGBoost (test season sample)")
        file_path = Path(output_dir) / "shap_summary.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {file_path}")

    def plot_tyre_life(self, explanation: shap.Explanation, x: pl.DataFrame, output_dir: str = "plots"):
        tyre_life_shap = explanation[:, x.columns.index("tyre_life")].values
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, code in COMPOUND_ORDER.items():
            rows = (x["compound"] == code).to_numpy()
            ax.scatter(x["tyre_life"].to_numpy()[rows], tyre_life_shap[rows], s=4, alpha=0.4,
                       color=COMPOUND_COLORS[label], label=label)
        ax.set_xlabel("Tyre life (laps)")
        ax.set_ylabel("SHAP value of tyre_life (s)")
        ax.set_title("Contribution of tyre life to the predicted lap delta (test season sample)")
        ax.legend(title="Compound", markerscale=4)
        file_path = Path(output_dir) / "shap_tyre_life.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {file_path}")

    def run(self, test_year: int = 2026) -> pl.DataFrame:
        model = self.load_model()
        laps_df = self.load_laps()
        x, _ = self.build_features(laps_df, test_year)
        x_test = x.filter(pl.col("year") >= test_year).select(self.columns)
        sample = x_test.sample(min(SAMPLE_LAPS, x_test.height), seed=42)

        explanation = self.shap_values(model, sample)
        rebuilt = explanation.base_values + explanation.values.sum(axis=1)
        gap = np.abs(rebuilt - model.predict(sample.to_numpy())).max()
        print(f"Base value {float(np.mean(explanation.base_values)):.3f} s, largest gap to model.predict: {gap:.2e} s")

        importance = self.importance(explanation, sample.columns)
        with pl.Config(tbl_rows=-1):
            print(importance)
        self.plot_summary(explanation)
        self.plot_tyre_life(explanation, sample)
        return importance
