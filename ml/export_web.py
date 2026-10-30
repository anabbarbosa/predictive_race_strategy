import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from ml.explain import MLExplain
from ml.features import CIRCUIT_COLUMNS, COMPOUND_ORDER
from ml.models import MLModels
from ml.pit_loss import MLPitLoss
from ml.safety_car import MLSafetyCar
from ml.simulate import WINDOW_SECONDS, MLSimulate
from ml.traffic import MLTraffic
from ml.tuning import TRAINED_DIR, MLTuning
from ml.undercut import MLUndercut
from ml.validate import RESULTS_DIR

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
EXAMPLE_CIRCUIT, EXAMPLE_TEAM = "Shanghai", "Mercedes"
SHAP_POINTS = 1500
EXPERIMENT_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.7, colsample_bytree=1.0,
                         min_child_weight=20, random_state=42)


def rows(df: pl.DataFrame, digits: int = 3) -> list[dict]:
    return df.with_columns(pl.selectors.float().cast(pl.Float64).round(digits)).to_dicts()


class MLExportWeb(MLModels):
    """Everything the web page shows, computed from the database, the trained models and results/, into web/data.js."""

    def models(self, laps_df, x, y, test_year: int) -> list[dict]:
        x_train, x_test, y_train, y_test = self.split_by_season(laps_df, x, y, test_year)
        np_x_test = x_test.to_numpy()
        rf, xgb = self.predict(x_train.to_numpy(), y_train.to_numpy(), np_x_test)
        tuned_xgb, monotone = XGBRegressor(), XGBRegressor()
        tuned_xgb.load_model(TRAINED_DIR / "xgboost.json")
        monotone.load_model(TRAINED_DIR / "xgboost_monotone.json")
        results = [
            ("baseline", self.evaluate("Baseline A: global mean", y_test, self.baseline_mean(y_train, x_test))),
            ("baseline", self.evaluate("Baseline B: mean per compound", y_test, self.baseline_compound_mean(x_train, y_train, x_test))),
            ("default", self.evaluate("Random Forest", y_test, rf)),
            ("default", self.evaluate("XGBoost", y_test, xgb)),
            ("tuned", self.evaluate("Random Forest (tuned)", y_test, joblib.load(TRAINED_DIR / "random_forest.joblib").predict(np_x_test))),
            ("tuned", self.evaluate("XGBoost (tuned)", y_test, tuned_xgb.predict(np_x_test))),
            ("tuned", self.evaluate("XGBoost (tuned, monotone)", y_test, monotone.predict(np_x_test))),
        ]
        return [{"group": group, **{k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}}
                for group, result in results]

    def circuit_experiment(self, laps_df, x, y, test_year: int) -> list[dict]:
        """Same fixed XGBoost with and without each circuit feature: the comparison behind keeping only circuit_lap_time."""
        circuits = self.circuit_features(laps_df, test_year)
        deg = laps_df.join(circuits, on=["circuit_short_name", "compound"], how="left", maintain_order="left")["circuit_deg"]
        x = x.with_columns(circuit_deg=deg)
        base = [col for col in x.columns if col not in (*CIRCUIT_COLUMNS, "circuit_deg")]
        train = x["year"] < test_year
        out = []
        for name, cols in [("No circuit features", base), ("+ circuit_lap_time (kept)", base + ["circuit_lap_time"]),
                           ("+ circuit_deg", base + ["circuit_deg"]),
                           ("+ both", base + ["circuit_lap_time", "circuit_deg"])]:
            model = XGBRegressor(**EXPERIMENT_PARAMS).fit(x.filter(train).select(cols).to_numpy(), y.filter(train).to_numpy())
            mae = mean_absolute_error(y.filter(~train), model.predict(x.filter(~train).select(cols).to_numpy()))
            out.append({"features": name, "mae": round(mae, 4)})
        return out

    def explain(self, test_year: int) -> dict:
        explain = MLExplain()
        model = explain.load_model(lap_model="xgboost")
        x, _ = explain.build_features(explain.load_laps(), test_year)
        sample = x.filter(pl.col("year") >= test_year).select(explain.columns).sample(SHAP_POINTS, seed=42)
        explanation = explain.shap_values(model, sample)
        tyre_life_shap = explanation[:, sample.columns.index("tyre_life")].values
        names = {code: name for name, code in COMPOUND_ORDER.items()}
        return {
            "importance": rows(explain.importance(explanation, sample.columns), 4),
            "tyre_life": [{"tyre_life": int(t), "compound": names[int(c)], "shap": round(float(s), 3)}
                          for t, c, s in zip(sample["tyre_life"], sample["compound"], tyre_life_shap)],
        }

    def degradation(self, laps_df, x, test_year: int) -> dict:
        tuning = MLTuning()
        model = MLSimulate().load_model(lap_model="xgboost")
        x_train = x.filter(pl.col("year") < test_year)
        grid = tuning.degradation_grid(x.columns, lap_number=40, track_temperature=x_train["track_temperature"].median(),
                                       year=test_year - 1, team="McLaren",
                                       circuit_values={col: x_train[col].median() for col in CIRCUIT_COLUMNS})
        pred = model.predict(grid.to_numpy())
        names = {code: name for name, code in COMPOUND_ORDER.items()}
        curves = (
            grid.select("tyre_life", "compound").with_columns(delta=pred)
            .with_columns(pl.col("compound").replace_strict(names, return_dtype=pl.String))
        )
        return {
            "curves": rows(curves),
            "limits": {k: round(v, 1) for k, v in tuning.tyre_life_limits(laps_df, test_year).items()},
        }

    def strategies(self, year: int) -> list[dict]:
        """Best strategy of the example team at every circuit, with the pit window of its best 1-stop compound pair."""
        sim = MLSimulate()
        model, limits = sim.load_model(), sim.stint_limits(year)
        circuits = sim.fetch_data("SELECT DISTINCT circuit_short_name FROM results ORDER BY 1")["circuit_short_name"]
        out = []
        for circuit in circuits:
            context = sim.race_context(circuit)
            results, strategies = sim.best_strategies(model, context["total_laps"], EXAMPLE_TEAM, circuit,
                                                      context["track_temperature"], year, context["pit_loss"], 2, limits)
            pair = results.filter(pl.col("n_stops") == 1)["compounds"][0]
            curve = (
                results.filter(pl.col("compounds") == pair)
                .select(pit_lap=pl.col("pit_laps").list.first(), gap=pl.col("total_time") - pl.col("total_time").min())
                .sort("pit_lap")
            )
            window = curve.filter(pl.col("gap") <= WINDOW_SECONDS)["pit_lap"]
            top = results.head(8).with_columns(
                strategy=pl.col("strategy_id").map_elements(lambda i: sim.describe(strategies[i]), return_dtype=pl.String)
            )
            out.append({
                "circuit": circuit,
                "total_laps": int(context["total_laps"]),
                "track_temperature": round(context["track_temperature"], 1),
                "pit_loss": round(context["pit_loss"], 2),
                "n_strategies": len(strategies),
                "best": top["strategy"][0],
                "best_2_stop_gap": round(results.filter(pl.col("n_stops") == 2)["gap_to_best"].min(), 2),
                "pair": pair,
                "window": [int(window.min()), int(window.max())],
                "curve": rows(curve),
                "top": rows(top.select("strategy", "n_stops", "gap_to_best"), 2),
            })
        return out

    def safety_car(self, year: int) -> dict:
        mc = MLSafetyCar()
        summary = mc.run(EXAMPLE_CIRCUIT, EXAMPLE_TEAM, year, plot=False)
        periods = mc.neutral_periods()
        top = summary.head(6)
        boxes = [
            {"strategy": strategy, **{f"p{q}": round(float(np.percentile(gaps, q)), 3) for q in (5, 25, 50, 75, 95)},
             "mean": round(float(gaps.mean()), 3)}
            for strategy, strategy_id in top.select("strategy", "strategy_id").iter_rows()
            for gaps in [mc.sims.filter(pl.col("strategy_id") == strategy_id)["gap"].to_numpy()]
        ]
        return {
            "rates": {k: round(v, 2) for k, v in mc.deployment_rates(periods, EXAMPLE_CIRCUIT).items()},
            "neutral_pit_loss": rows(mc.neutral_pit_loss(), 3),
            "summary": rows(summary.drop("strategy_id").head(12), 2),
            "regret": boxes,
            "n_sims": mc.sims["sim"].n_unique(),
            "neutralised_pct": round(mc.sims.group_by("sim").agg(pl.col("neutralised").first())["neutralised"].mean() * 100, 1),
        }

    def undercut(self, year: int) -> dict:
        undercut = MLUndercut()
        table = undercut.run(EXAMPLE_CIRCUIT, EXAMPLE_TEAM, year, plot=False)
        return {"old": "MEDIUM", "new": "HARD", "cells": rows(table)}

    def traffic(self, year: int) -> dict:
        traffic = MLTraffic()
        ranking = traffic.run(EXAMPLE_CIRCUIT, EXAMPLE_TEAM, year, plot=False)
        return {
            "dirty_air": rows(traffic.dirty_air().select(pl.col("gap_bin").cast(pl.String), "n_laps", "penalty_s"), 3),
            "ranking": rows(ranking.select("strategy", "free_time", "traffic_time", "traffic_loss_s", "laps_held",
                                           "position", "free_rank"), 2),
        }

    def validation(self) -> dict:
        drivers = pl.read_csv(RESULTS_DIR / "validation_drivers.csv").with_columns(
            actual_gap=pl.col("actual") - pl.col("actual").mean().over("session_key"),
            predicted_gap=pl.col("predicted") - pl.col("predicted").mean().over("session_key"),
        )
        return {
            "summary": rows(pl.read_csv(RESULTS_DIR / "validation_summary.csv"), 3),
            "drivers": rows(drivers.filter(pl.col("gap_error").is_not_null()).select(
                "circuit_short_name", "group", "position", "driver_number", "team_name", "strategy", "actual_gap",
                "predicted_gap", "error"), 2),
            "optimal": rows(pl.read_csv(RESULTS_DIR / "validation_optimal.csv"), 2),
        }

    def dataset(self) -> dict:
        counts = self.fetch_data("""
            SELECT (SELECT COUNT(DISTINCT session_key) FROM laps) AS races,
                   (SELECT COUNT(*) FROM laps) AS clean_laps,
                   (SELECT COUNT(*) FROM stints) AS stints,
                   (SELECT COUNT(*) FROM pit_stops) AS pit_stops,
                   (SELECT COUNT(*) FROM pit_stops WHERE is_representative) AS representative_stops,
                   (SELECT COUNT(*) FROM safety_car_periods) AS neutral_periods
        """)
        return counts.to_dicts()[0]

    def run(self, test_year: int = 2026) -> Path:
        laps_df = self.load_laps()
        x, y = self.build_features(laps_df, test_year)
        data = {
            "generated": date.today().isoformat(),
            "test_year": test_year,
            "example": {"circuit": EXAMPLE_CIRCUIT, "team": EXAMPLE_TEAM},
            "dataset": self.dataset(),
            "models": self.models(laps_df, x, y, test_year),
            "circuit_experiment": self.circuit_experiment(laps_df, x, y, test_year),
            "shap": self.explain(test_year),
            "degradation": self.degradation(laps_df, x, test_year),
            "pit_loss": rows(MLPitLoss().pit_loss_table(), 2),
            "strategies": self.strategies(test_year),
            "validation": self.validation(),
            "safety_car": self.safety_car(test_year),
            "undercut": self.undercut(test_year),
            "traffic": self.traffic(test_year),
        }
        WEB_DIR.mkdir(exist_ok=True)
        path = WEB_DIR / "data.js"
        # A script instead of JSON so index.html also works opened straight from disk, where fetch() is blocked
        path.write_text(f"window.RESULTS = {json.dumps(data, ensure_ascii=False)};\n", encoding="utf-8")
        print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB)")
        return path
