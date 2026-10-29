from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ml.features import FUEL_SECONDS_PER_LAP
from ml.validate import MLValidate

GAP_BINS = [0.5, 1.0, 1.5, 2.0, 3.0]
FREE_AIR = "3.0+"
OVERTAKE_DELTA = 0.7
MIN_FOLLOW_GAP = 0.3
TOP_STRATEGIES = 20

class MLTraffic(MLValidate):
    def gap_ahead(self) -> pl.DataFrame:
        return (
            self.fetch_data("SELECT session_key, driver_number, lap_number, date_start FROM race_laps")
            .sort("session_key", "lap_number", "date_start")
            .with_columns(gap_ahead=pl.col("date_start").diff().over("session_key", "lap_number").dt.total_milliseconds() / 1000)
            .drop("date_start")
        )

    def dirty_air(self) -> pl.DataFrame:
        model = self.load_model()
        laps_df = self.load_laps()
        x, y = self.build_features(laps_df)
        residual = y.to_numpy() - model.predict(x.select(self.columns).to_numpy())
        binned = (
            laps_df.select("session_key", "driver_number", "lap_number")
            .with_columns(residual=residual)
            .join(self.gap_ahead(), on=["session_key", "driver_number", "lap_number"], how="inner")
            .filter(pl.col("gap_ahead").is_not_null())
            .with_columns(gap_bin=pl.col("gap_ahead").cut(GAP_BINS, labels=[
                "0.0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", FREE_AIR]))
            .group_by("gap_bin")
            .agg(n_laps=pl.len(), residual=pl.col("residual").median())
        )
        free_air = binned.filter(pl.col("gap_bin") == FREE_AIR)["residual"].item()
        return (
            binned.with_columns(penalty_s=(pl.col("residual") - free_air).clip(lower_bound=0))
            .with_columns(upper=pl.col("gap_bin").cast(pl.String).str.split("-").list.last().str.strip_suffix("+").cast(pl.Float64))
            .sort("upper")
        )

    def penalty(self, table: pl.DataFrame, gap: float) -> float:
        for upper, penalty in table.select("upper", "penalty_s").iter_rows():
            if gap < upper:
                return penalty
        return 0.0

    def field(self, session_key: int, model, total_laps: int, circuit: str, pit_loss: float) -> pl.DataFrame:
        results = self.fetch_data(f"SELECT * FROM results WHERE session_key = {session_key} ORDER BY position")
        stints = self.fetch_data(f"SELECT * FROM stints WHERE session_key = {session_key}")
        laps = self.fetch_data(f"""
            SELECT driver_number, lap_number, lap_duration, track_temperature, year,
                   MAX(lap_number) OVER () - lap_number AS laps_remaining
            FROM laps WHERE session_key = {session_key}
        """)
        track = laps["track_temperature"].median()
        start = (
            self.fetch_data(f"SELECT driver_number, date_start FROM race_laps WHERE session_key = {session_key} AND lap_number = 2")
            .with_columns(start_gap=(pl.col("date_start") - pl.col("date_start").min()).dt.total_milliseconds() / 1000)
        )
        cars = []
        for driver in results.filter(pl.col("laps_completed") == total_laps).iter_rows(named=True):
            driver_stints = stints.filter(pl.col("driver_number") == driver["driver_number"])
            strategy, _ = self.real_strategy(driver_stints, total_laps) if driver_stints.height else (None, "")
            own = laps.filter(pl.col("driver_number") == driver["driver_number"])
            gap = start.filter(pl.col("driver_number") == driver["driver_number"])
            if strategy is None or own.is_empty() or gap.is_empty():
                continue
            base_pace = (own["lap_duration"] - FUEL_SECONDS_PER_LAP * own["laps_remaining"]).median()
            times = self.lap_times([strategy], model, total_laps, driver["team_name"], circuit, track,
                                   driver["year"], base_pace)["lap_time"].to_numpy()
            stops = np.zeros(total_laps)
            stops[[lap - 1 for lap in self.pit_laps(strategy)]] = pit_loss
            cars.append({
                "driver_number": driver["driver_number"],
                "team_name": driver["team_name"],
                "position": driver["position"],
                "base_pace": base_pace,
                "strategy": strategy,
                "end_of_lap": gap["start_gap"].item() + np.cumsum(np.where(np.arange(total_laps) == 0, 0, times + stops)),
                "stops": stops,
            })
        return cars

    def drive(self, free_laps: np.ndarray, stops: np.ndarray, start_gap: float, rivals: list[dict],
              dirty_air: pl.DataFrame) -> tuple[float, float, int]:
        reach = dirty_air["upper"].filter(dirty_air["penalty_s"] > 0).max() or 0.0
        t, lost, laps_held = start_gap, 0.0, 0
        for lap in range(1, len(free_laps)):
            lap_time = free_laps[lap] + stops[lap]
            ahead = [r for r in rivals if r["end_of_lap"][lap - 1] < t]
            if ahead and stops[lap] == 0:
                rival = max(ahead, key=lambda r: r["end_of_lap"][lap - 1])
                car = rival["end_of_lap"]
                gap, their_lap = t - car[lap - 1], car[lap] - car[lap - 1]
                if gap < reach and rival["stops"][lap] == 0:
                    held = lap_time + self.penalty(dirty_air, gap)
                    if free_laps[lap] > their_lap - OVERTAKE_DELTA:
                        held = max(held, car[lap] + MIN_FOLLOW_GAP - t)
                    lost += held - lap_time
                    laps_held += held > lap_time
                    lap_time = held
            t += lap_time
        return t, lost, laps_held

    def plot(self, table: pl.DataFrame, circuit: str, team: str, output_dir: str = "plots"):
        top = table.head(10)
        best_free = table["free_time"].min()
        free_gap = (top["free_time"] - best_free).to_numpy()
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = top["strategy"].to_list()
        ax.barh(labels, free_gap, color="#2a78d6", edgecolor="white", linewidth=2, label="Gap to the best strategy without traffic")
        ax.barh(labels, top["traffic_loss_s"].to_numpy(), left=free_gap, color="#eb6834", edgecolor="white", linewidth=2,
                label="Time lost in traffic")
        ax.invert_yaxis()
        ax.set_xlabel("Seconds behind the best strategy without traffic")
        ax.set_title(f"Strategies ranked with traffic — {circuit}, {team}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        file_path = Path(output_dir) / "traffic.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {file_path}")

    def run(self, circuit: str, team: str, year: int, max_stops: int = 2, plot: bool = True) -> pl.DataFrame:
        context = self.race_context(circuit)
        total_laps, pit_loss, session_key = context["total_laps"], context["pit_loss"], context["session_key"]
        model = self.load_model()

        dirty_air = self.dirty_air()
        with pl.Config(float_precision=3):
            print(dirty_air.drop("upper"))
        cars = self.field(session_key, model, total_laps, circuit, pit_loss)
        ours = next((car for car in cars if car["team_name"] == team), None)
        if ours is None:
            raise ValueError(f"No simulable {team} car in the latest {circuit} race (session {session_key})")
        rivals = [car for car in cars if car is not ours]
        print(f"Reference race: session {session_key}, {len(cars)} cars simulated with their real strategies. "
              f"Our car replaces #{ours['driver_number']} (P{ours['position']}, base pace {ours['base_pace']:.3f} s).")

        limits = self.stint_limits(year)
        results, strategies = self.best_strategies(model, total_laps, team, circuit, context["track_temperature"],
                                                   year, pit_loss, max_stops, limits)
        best_per_plan = results.group_by("compounds", "n_stops").agg(pl.all().sort_by("total_time").first())
        candidates = pl.concat([results.head(TOP_STRATEGIES), best_per_plan.select(results.columns)]).unique("strategy_id")
        candidate_strategies = [strategies[i] for i in candidates["strategy_id"]] + [ours["strategy"]]

        start_gap = ours["end_of_lap"][0]
        rows = []
        for i, strategy in enumerate(candidate_strategies):
            free = self.lap_times([strategy], model, total_laps, team, circuit, context["track_temperature"], year,
                                  ours["base_pace"])["lap_time"].to_numpy()
            stops = np.zeros(total_laps)
            stops[[lap - 1 for lap in self.pit_laps(strategy)]] = pit_loss
            free_time = start_gap + free[1:].sum() + stops.sum()
            time, lost, laps_held = self.drive(free, stops, start_gap, rivals, dirty_air)
            rows.append({
                "strategy": self.describe(strategy) + (" (real)" if i == len(candidate_strategies) - 1 else ""),
                "free_time": free_time,
                "traffic_time": time,
                "traffic_loss_s": lost,
                "laps_held": laps_held,
                "position": 1 + sum(r["end_of_lap"][-1] < time for r in rivals),
            })
        table = pl.DataFrame(rows).sort("traffic_time").with_columns(
            free_rank=pl.col("free_time").rank("min").cast(pl.Int64),
            gap_to_best_s=pl.col("traffic_time") - pl.col("traffic_time").min(),
        )
        with pl.Config(tbl_rows=15, tbl_width_chars=200, float_precision=2):
            print(table.head(15))
        best_free = table.sort("free_time").row(0, named=True)
        best = table.row(0, named=True)
        if best["strategy"] == best_free["strategy"]:
            print(f"Traffic doesn't change the best strategy: {best['strategy']}, {best['traffic_loss_s']:.2f} s lost in traffic.")
        else:
            print(f"With traffic the best is {best['strategy']} (P{best['position']}), "
                  f"{best_free['traffic_time'] - best['traffic_time']:.2f} s ahead of the best without traffic, "
                  f"{best_free['strategy']}, which loses {best_free['traffic_loss_s']:.2f} s behind other cars.")
        if plot:
            self.plot(table, circuit, team)
        return table