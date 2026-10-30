from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from xgboost import XGBRegressor

from ml.degradation import DegradationModel, MLDegradation
from ml.features import COMPOUND_ORDER, FUEL_SECONDS_PER_LAP, TEAM_ALIASES, MLfeatures
from ml.pit_loss import MLPitLoss
from ml.tuning import TRAINED_DIR, MLTuning

Strategy = list[tuple[str, int]]

MIN_STINT = 5
STINT_QUANTILE = 0.99
WINDOW_SECONDS = 1.0
LAP_MODEL = "xgboost"  # or "degradation" (ml/degradation.py); main.py sets it from --lap-model


class MLSimulate(MLfeatures):
    def __init__(self):
        super().__init__()
        self.columns = self.load_feature_columns()
        self.circuits = self.load_circuit_features()

    def load_model(self, file_name: str = "xgboost_monotone.json", lap_model: str | None = None) -> XGBRegressor | DegradationModel:
        if (lap_model or LAP_MODEL) == "degradation":
            return MLDegradation().load()
        model = XGBRegressor()
        model.load_model(TRAINED_DIR / file_name)
        return model

    def check_strategy(self, strategy: Strategy, total_laps: int):
        unknown = [compound for compound, _ in strategy if compound not in COMPOUND_ORDER]
        if unknown:
            raise ValueError(f"Unknown compounds {unknown}, expected one of {list(COMPOUND_ORDER)}")
        if any(length < 1 for _, length in strategy):
            raise ValueError(f"Every stint needs at least one lap: {strategy}")
        laps = sum(length for _, length in strategy)
        if laps != total_laps:
            raise ValueError(f"Stints add up to {laps} laps, the race has {total_laps}: {strategy}")

    def strategy_laps(self, strategies: list[Strategy]) -> pl.DataFrame:
        stints = pl.DataFrame(
            [(strategy_id, compound, length)
             for strategy_id, strategy in enumerate(strategies) for compound, length in strategy],
            schema={"strategy_id": pl.Int64, "compound": pl.String, "length": pl.Int64},
            orient="row",
        )
        return (
            stints.with_columns(tyre_life=pl.int_ranges("length"))
            .explode("tyre_life")
            .with_columns(lap_number=pl.int_range(1, pl.len() + 1).over("strategy_id"))
            .drop("length")
        )

    def build_rows(self, laps_df: pl.DataFrame, team: str, circuit: str, track_temperature: float, year: int) -> pl.DataFrame:
        circuit_rows = self.circuits.filter(pl.col("circuit_short_name") == circuit)
        if circuit_rows.is_empty():
            raise ValueError(f"No circuit features for {circuit!r}, run --features 1 after loading its races")
        team_column = f"team_name_{TEAM_ALIASES.get(team, team)}"
        team_columns = [col for col in self.columns if col.startswith("team_name_")]
        if team_column not in team_columns:
            known = [col.removeprefix("team_name_") for col in team_columns]
            raise ValueError(f"Unknown team {team!r}, the model knows {known}")
        return (
            self.add_circuit_features(laps_df.with_columns(circuit_short_name=pl.lit(circuit)), circuit_rows)
            .with_columns(
                *[pl.lit(int(col == team_column)).alias(col) for col in team_columns],
                compound=pl.col("compound").replace_strict(COMPOUND_ORDER),
                track_temperature=pl.lit(track_temperature),
                year=pl.lit(year),
            )
            .select(self.columns)
        )

    def predict(self, model, rows: pl.DataFrame, circuit: str) -> np.ndarray:
        if isinstance(model, DegradationModel):
            return model.predict_circuit(rows.to_numpy(), circuit)
        return model.predict(rows.to_numpy())

    def lap_times(self, strategies: list[Strategy], model, total_laps: int, team: str, circuit: str,
                  track_temperature: float, year: int, base_pace: float = 0.0) -> pl.DataFrame:
        for strategy in strategies:
            self.check_strategy(strategy, total_laps)
        laps_df = self.strategy_laps(strategies)
        lap_delta = pl.Series(self.predict(model, self.build_rows(laps_df, team, circuit, track_temperature, year), circuit))
        if base_pace and isinstance(model, DegradationModel):
            lap_delta = lap_delta - laps_df.with_columns(d=lap_delta).select(pl.col("d").median().over("strategy_id"))["d"]
        return laps_df.with_columns(
            lap_delta=lap_delta,
            lap_time=base_pace + lap_delta + FUEL_SECONDS_PER_LAP * (total_laps - pl.col("lap_number")),
        )

    def simulate_race(self, strategy: Strategy, model, total_laps: int, team: str, circuit: str,
                      track_temperature: float, year: int, pit_loss: float, base_pace: float = 0.0) -> float:
        laps_df = self.lap_times([strategy], model, total_laps, team, circuit, track_temperature, year, base_pace)
        return laps_df["lap_time"].sum() + (len(strategy) - 1) * pit_loss

    def lap_delta_grid(self, model, total_laps: int, team: str, circuit: str, track_temperature: float,
                       year: int) -> np.ndarray:
        grid = (
            pl.DataFrame({"compound": list(COMPOUND_ORDER)})
            .join(pl.DataFrame({"tyre_life": range(total_laps)}), how="cross")
            .join(pl.DataFrame({"lap_number": range(1, total_laps + 1)}), how="cross")
        )
        delta = self.predict(model, self.build_rows(grid, team, circuit, track_temperature, year), circuit)
        fuel = FUEL_SECONDS_PER_LAP * (total_laps - grid["lap_number"].to_numpy())
        return (delta + fuel).reshape(len(COMPOUND_ORDER), total_laps, total_laps)

    def stint_time(self, grid: np.ndarray, compound: str, first_lap: int, last_lap: int) -> float:
        laps = np.arange(first_lap, last_lap + 1)
        return grid[COMPOUND_ORDER[compound], laps - first_lap, laps - 1].sum()

    def race_time(self, grid: np.ndarray, compounds: list[str], pit_laps: list[int], total_laps: int,
                  stop_costs: list[float]) -> float:
        bounds = [0, *pit_laps, total_laps]
        laps = sum(self.stint_time(grid, c, start + 1, end) for c, start, end in zip(compounds, bounds, bounds[1:]))
        return laps + sum(stop_costs)

    def stint_limits(self, test_year: int) -> dict[str, int]:
        laps_df = self.fetch_data("SELECT compound, tyre_life, year FROM laps")
        limits = MLTuning().tyre_life_limits(laps_df, test_year, quantile=STINT_QUANTILE)
        return {compound: int(limit) + 1 for compound, limit in limits.items()}

    def generate_strategies(self, total_laps: int, max_stops: int = 2, min_stint: int = MIN_STINT,
                            max_stint: dict[str, int] | None = None):
        max_stint = max_stint or {}
        for stops in range(1, max_stops + 1):
            for pit_laps in combinations(range(1, total_laps), stops):
                bounds = (0, *pit_laps, total_laps)
                lengths = [end - start for start, end in zip(bounds, bounds[1:])]
                if min(lengths) < min_stint:
                    continue
                for compounds in product(COMPOUND_ORDER, repeat=stops + 1):
                    if len(set(compounds)) < 2:
                        continue
                    if any(length > max_stint.get(c, total_laps) for c, length in zip(compounds, lengths)):
                        continue
                    yield list(zip(compounds, lengths))

    def simulate_many(self, strategies: list[Strategy], model, total_laps: int, team: str, circuit: str,
                      track_temperature: float, year: int, pit_loss: float) -> pl.DataFrame:
        totals = (
            self.lap_times(strategies, model, total_laps, team, circuit, track_temperature, year)
            .group_by("strategy_id")
            .agg(lap_time=pl.col("lap_time").sum())
        )
        summary = pl.DataFrame({
            "strategy_id": range(len(strategies)),
            "compounds": ["-".join(c[0] for c, _ in s) for s in strategies],
            "pit_laps": [self.pit_laps(s) for s in strategies],
            "n_stops": [len(s) - 1 for s in strategies],
        })
        return (
            summary.join(totals, on="strategy_id")
            .with_columns(total_time=pl.col("lap_time") + pl.col("n_stops") * pit_loss)
            .with_columns(gap_to_best=pl.col("total_time") - pl.col("total_time").min())
            .drop("lap_time")
            .sort("total_time")
        )

    def pit_laps(self, strategy: Strategy) -> list[int]:
        laps, total = [], 0
        for _, length in strategy[:-1]:
            total += length
            laps.append(total)
        return laps

    def describe(self, strategy: Strategy) -> str:
        return " -> ".join(f"{compound} {length}" for compound, length in strategy)

    def race_context(self, circuit: str) -> dict:
        races = self.fetch_data(f"""
            SELECT session_key, year, MAX(laps_completed) AS total_laps
            FROM results WHERE circuit_short_name = '{circuit.replace("'", "''")}'
            GROUP BY session_key, year ORDER BY year DESC
        """)
        if races.is_empty():
            known = self.fetch_data("SELECT DISTINCT circuit_short_name FROM results ORDER BY 1")["circuit_short_name"].to_list()
            raise ValueError(f"Unknown circuit {circuit!r}, expected one of {known}")
        session_key = races["session_key"][0]
        track_temperature = self.fetch_data(
            f"SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY track_temperature) AS t FROM laps WHERE session_key = {session_key}"
        )["t"].item()
        return {
            "session_key": session_key,
            "total_laps": races["total_laps"][0],
            "track_temperature": track_temperature,
            "pit_loss": MLPitLoss().pit_loss(circuit),
        }

    def plot_pit_window(self, results: pl.DataFrame, circuit: str, team: str, output_dir: str = "plots"):
        one_stop = results.filter(pl.col("n_stops") == 1)
        if one_stop.is_empty():
            return
        pair = one_stop["compounds"][0]
        curve = (
            one_stop.filter(pl.col("compounds") == pair)
            .with_columns(pit_lap=pl.col("pit_laps").list.first(), gap=pl.col("total_time") - pl.col("total_time").min())
            .sort("pit_lap")
        )
        window = curve.filter(pl.col("gap") <= WINDOW_SECONDS)["pit_lap"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axvspan(window.min() - 0.5, window.max() + 0.5, color="#E10600", alpha=0.12,
                   label=f"Pit window (within {WINDOW_SECONDS:.0f} s of best): laps {window.min()}–{window.max()}")
        ax.plot(curve["pit_lap"], curve["gap"], color="#E10600", marker="o", markersize=3)
        ax.set_xlabel("Pit lap")
        ax.set_ylabel("Predicted race time vs best pit lap (s)")
        ax.set_title(f"1-stop pit window, {pair} — {circuit}, {team}")
        ax.legend(loc="upper center")
        file_path = Path(output_dir) / "pit_window.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {file_path}")

    def best_strategies(self, model, total_laps: int, team: str, circuit: str, track_temperature: float, year: int,
                        pit_loss: float, max_stops: int, stint_limits: dict[str, int]) -> tuple[pl.DataFrame, list[Strategy]]:
        strategies = list(self.generate_strategies(total_laps, max_stops, max_stint=stint_limits))
        if not strategies:
            raise ValueError(f"No strategy fits {total_laps} laps with stints of {MIN_STINT} to {stint_limits} laps")
        return self.simulate_many(strategies, model, total_laps, team, circuit, track_temperature, year, pit_loss), strategies

    def run(self, circuit: str, team: str, year: int, strategy: Strategy | None = None, total_laps: int | None = None,
            track_temperature: float | None = None, pit_loss: float | None = None, max_stops: int = 2,
            session_key: int | None = None, driver_number: int | None = None, plot: bool = True) -> pl.DataFrame:
        context = self.race_context(circuit)
        total_laps = total_laps or context["total_laps"]
        track_temperature = track_temperature if track_temperature is not None else context["track_temperature"]
        pit_loss = pit_loss if pit_loss is not None else context["pit_loss"]
        print(f"{circuit}, {team}, {year}: {total_laps} laps, track {track_temperature:.1f} °C, pit loss {pit_loss:.2f} s")
        model = self.load_model()

        if strategy:
            base_pace = self.base_pace(session_key, driver_number) if session_key and driver_number else 0.0
            laps_df = self.lap_times([strategy], model, total_laps, team, circuit, track_temperature, year, base_pace)
            total = laps_df["lap_time"].sum() + (len(strategy) - 1) * pit_loss
            print(laps_df.drop("strategy_id"))
            print(f"{self.describe(strategy)} ({len(strategy) - 1} stops)")
            if base_pace:
                print(f"Predicted race time: {total:.2f} s (base pace {base_pace:.3f} s)")
            else:
                print(f"Predicted race time: {total:.2f} s relative to base pace (pass --session-key and --driver for an absolute time)")
            return laps_df

        limits = self.stint_limits(year)
        print(f"Stint limits (p{STINT_QUANTILE * 100:.0f} tyre life in training + 1): {limits}, minimum {MIN_STINT} laps")
        results, strategies = self.best_strategies(model, total_laps, team, circuit, track_temperature, year, pit_loss, max_stops, limits)
        print(f"{len(strategies)} strategies simulated")
        with pl.Config(tbl_rows=10, fmt_table_cell_list_len=3):
            print(results.head(10).drop("strategy_id"))

        best = results.row(0, named=True)
        other_plan = results.filter(pl.col("compounds") != best["compounds"]).row(0, named=True)
        print(f"Best strategy: {self.describe(strategies[best['strategy_id']])} (pit on lap {', '.join(map(str, best['pit_laps']))}).")
        print(f"It beats the next best ({self.describe(strategies[results['strategy_id'][1]])}) by {results['gap_to_best'][1]:.2f} s "
              f"and the best different plan ({self.describe(strategies[other_plan['strategy_id']])}) by {other_plan['gap_to_best']:.2f} s.")
        for stops in range(1, max_stops + 1):
            gap = results.filter(pl.col("n_stops") == stops)["gap_to_best"].min()
            if gap is not None:
                print(f"Best {stops}-stop: +{gap:.2f} s")
        if plot:
            self.plot_pit_window(results, circuit, team)
        return results

    def base_pace(self, session_key: int, driver_number: int) -> float:
        laps_df = self.load_laps().filter(
            pl.col("session_key") == session_key,
            pl.col("driver_number") == driver_number,
        )
        if laps_df.is_empty():
            raise ValueError(f"No clean laps for driver {driver_number} in session {session_key}")
        return laps_df.select(self.fuel_corrected().median()).item()
