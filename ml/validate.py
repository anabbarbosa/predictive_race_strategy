from pathlib import Path

import polars as pl
from scipy.stats import spearmanr

from ml.pit_loss import MLPitLoss
from ml.safety_car import COMPOUNDS, MLSafetyCar
from ml.simulate import WINDOW_SECONDS, MLSimulate, Strategy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TOP_N = 5
REPLAY_SIMS = 300
DRY = {"SOFT", "MEDIUM", "HARD"}


class MLValidate(MLSimulate):
    def real_strategy(self, stints: pl.DataFrame, laps_completed: int) -> tuple[Strategy | None, str]:
        stints = stints.sort("stint_number")
        if stints["lap_start"][0] != 1 or (stints["lap_start"][1:] != stints["lap_end"][:-1] + 1).any():
            return None, "gap in stint data"
        if not set(stints["compound"]) <= DRY:
            return None, "wet tyres"
        if stints["lap_end"][-1] < laps_completed:
            return None, "stints end before the race"
        lengths = (stints["lap_end"] - stints["lap_start"] + 1).to_list()
        lengths[-1] -= stints["lap_end"][-1] - laps_completed
        return list(zip(stints["compound"], lengths)), ""

    def load_race_data(self, test_year: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        results = self.fetch_data(f"SELECT * FROM results WHERE year = {test_year} AND race_time IS NOT NULL")
        stints = self.fetch_data(f"SELECT * FROM stints WHERE year = {test_year}")
        laps = self.fetch_data(f"""
            SELECT session_key, driver_number, lap_number, lap_duration, track_temperature
            FROM laps WHERE year = {test_year}
        """)
        # Base pace = each driver's median fuel-corrected lap, the reference lap_delta is relative to
        base_pace = laps.with_columns(corrected=self.fuel_corrected()).group_by("session_key", "driver_number").agg(
            base_pace=pl.col("corrected").median()
        )
        track = laps.group_by("session_key").agg(track_temperature=pl.col("track_temperature").median())
        wet = stints.filter(~pl.col("compound").is_in(list(DRY)))["session_key"].unique()
        finishers = (
            results.filter(pl.col("laps_completed") == pl.col("laps_completed").max().over("session_key"))
            .filter(pl.col("position") <= TOP_N)
            .join(base_pace, on=["session_key", "driver_number"], how="left")
            .join(track, on="session_key", how="left")
            # Worst race event first: a suspension adds half an hour, rain changes the whole race, a Safety Car
            # slows several laps, a VSC only a few
            .with_columns(group=pl.when("suspended").then(pl.lit("Suspended"))
                          .when(pl.col("session_key").is_in(wet) | pl.col("had_rain")).then(pl.lit("Wet"))
                          .when(pl.col("safety_car_laps") > 0).then(pl.lit("Safety Car"))
                          .otherwise(pl.lit("VSC only")))
            .sort("session_key", "position")
        )
        return finishers, stints, laps

    def predict_driver(self, driver: dict, strategy: Strategy, model, pit_loss: float) -> float:
        total_laps = driver["laps_completed"]
        first, last = driver["timed_from_lap"], total_laps - 1
        laps_df = self.lap_times([strategy], model, total_laps, driver["team_name"], driver["circuit_short_name"],
                                 driver["track_temperature"], driver["year"], driver["base_pace"])
        stops = sum(first <= lap <= last for lap in self.pit_laps(strategy))
        return laps_df.filter(pl.col("lap_number").is_between(first, last))["lap_time"].sum() + stops * pit_loss

    def accuracy(self, finishers: pl.DataFrame, stints: pl.DataFrame, model, pit_losses: dict[str, float]) -> pl.DataFrame:
        rows = []
        for driver in finishers.iter_rows(named=True):
            driver_stints = stints.filter(pl.col("session_key") == driver["session_key"],
                                          pl.col("driver_number") == driver["driver_number"])
            strategy, skipped = self.real_strategy(driver_stints, driver["laps_completed"]) if driver_stints.height else (None, "no stints")
            if strategy is None or driver["base_pace"] is None:
                print(f"Skipping {driver['circuit_short_name']} P{driver['position']} #{driver['driver_number']}: {skipped or 'no clean laps'}")
                continue
            predicted = self.predict_driver(driver, strategy, model, pit_losses[driver["circuit_short_name"]])
            rows.append({
                "session_key": driver["session_key"],
                "circuit_short_name": driver["circuit_short_name"],
                "group": driver["group"],
                "position": driver["position"],
                "driver_number": driver["driver_number"],
                "team_name": driver["team_name"],
                "strategy": self.describe(strategy),
                "used_tyres": int((driver_stints["tyre_age_at_start"] > 0).sum()),
                "actual": driver["race_time"],
                "predicted": predicted,
            })
        return pl.DataFrame(rows).with_columns(
            error=pl.col("predicted") - pl.col("actual"),
            error_pct=(pl.col("predicted") - pl.col("actual")) / pl.col("actual") * 100,
            # Each driver relative to the race average, so what every car lived through (SC laps, a suspension)
            # cancels out and only the strategy and pace differences the model is meant to capture remain
            gap_error=pl.when(pl.len().over("session_key") > 1).then(
                (pl.col("predicted") - pl.col("predicted").mean().over("session_key"))
                - (pl.col("actual") - pl.col("actual").mean().over("session_key"))
            ),
        )

    def summary(self, drivers: pl.DataFrame) -> pl.DataFrame:
        ranking = (
            drivers.group_by("session_key", "group")
            .agg(pl.col("actual"), pl.col("predicted"))
            .filter(pl.col("actual").list.len() >= 3)
            .with_columns(spearman=pl.struct("actual", "predicted").map_elements(
                lambda race: spearmanr(race["actual"], race["predicted"]).statistic, return_dtype=pl.Float64))
            .group_by("group").agg(spearman=pl.col("spearman").mean())
        )
        return (
            drivers.group_by("group")
            .agg(
                races=pl.col("session_key").n_unique(),
                drivers=pl.len(),
                mae_s=pl.col("error").abs().mean(),
                bias_s=pl.col("error").mean(),
                mae_pct=pl.col("error_pct").abs().mean(),
                gap_mae_s=pl.col("gap_error").abs().mean(),
            )
            .join(ranking, on="group", how="left")
            .sort(pl.col("group").replace_strict({"VSC only": 0, "Safety Car": 1, "Wet": 2, "Suspended": 3}))
        )

    def optimal_vs_winner(self, drivers: pl.DataFrame, finishers: pl.DataFrame, stints: pl.DataFrame, model,
                          pit_losses: dict[str, float], test_year: int) -> pl.DataFrame:
        limits = self.stint_limits(test_year)
        rows = []
        for winner in finishers.filter(pl.col("position") == 1).iter_rows(named=True):
            winner_stints = stints.filter(pl.col("session_key") == winner["session_key"],
                                          pl.col("driver_number") == winner["driver_number"])
            strategy, _ = self.real_strategy(winner_stints, winner["laps_completed"]) if winner_stints.height else (None, "")
            if strategy is None:
                continue
            total_laps, pit_loss = winner["laps_completed"], pit_losses[winner["circuit_short_name"]]
            args = (model, total_laps, winner["team_name"], winner["circuit_short_name"], winner["track_temperature"], test_year)
            results, strategies = self.best_strategies(*args, pit_loss, 2, limits)
            best = results.row(0, named=True)
            winner_total = self.simulate_race(strategy, *args, pit_loss)
            winner_compounds = "-".join(c[0] for c, _ in strategy)
            # Given the compounds the winner chose, the pit laps the simulator rates within WINDOW_SECONDS of the best
            same_plan = results.filter(pl.col("compounds") == winner_compounds, pl.col("n_stops") == len(strategy) - 1)
            window = same_plan.filter(pl.col("total_time") - pl.col("total_time").min() <= WINDOW_SECONDS)
            in_window = len(strategy) == 2 and window.height > 0 and (
                window["pit_laps"].list.first().min() <= self.pit_laps(strategy)[0] <= window["pit_laps"].list.first().max())
            rows.append({
                "circuit_short_name": winner["circuit_short_name"],
                "group": winner["group"],
                "winner_strategy": self.describe(strategy),
                "optimal_strategy": self.describe(strategies[best["strategy_id"]]),
                "same_stops": len(strategy) - 1 == best["n_stops"],
                "same_compounds": winner_compounds == best["compounds"],
                "optimal_pit_lap_same_compounds": same_plan["pit_laps"][0][0] if len(strategy) == 2 and same_plan.height else None,
                "winner_pit_in_window": in_window,
                "predicted_gain_s": winner_total - best["total_time"],
            })
        return pl.DataFrame(rows)

    def safety_car_replay(self, finishers: pl.DataFrame, stints: pl.DataFrame, model, pit_losses: dict[str, float],
                          test_year: int) -> pl.DataFrame:
        sc = MLSafetyCar()
        periods = sc.neutral_periods(before_year=test_year)
        ratios = dict(sc.neutral_pit_loss().select("sc_type", "ratio").iter_rows())
        real = self.fetch_data(f"SELECT * FROM safety_car_periods WHERE year = {test_year}")
        limits = self.stint_limits(test_year)
        rows = []
        for winner in finishers.filter(pl.col("position") == 1).iter_rows(named=True):
            winner_stints = stints.filter(pl.col("session_key") == winner["session_key"],
                                          pl.col("driver_number") == winner["driver_number"])
            strategy, _ = self.real_strategy(winner_stints, winner["laps_completed"]) if winner_stints.height else (None, "")
            if strategy is None:
                continue
            circuit, total_laps = winner["circuit_short_name"], winner["laps_completed"]
            pit_loss = pit_losses[circuit]
            args = (model, total_laps, winner["team_name"], circuit, winner["track_temperature"], test_year)
            results, strategies = self.best_strategies(*args, pit_loss, 2, limits)
            candidates = sc.candidates(results)
            grid = self.lap_delta_grid(*args)
            rates = sc.deployment_rates(periods, circuit, before_year=test_year)
            sims = sc.monte_carlo(candidates, strategies, grid, total_laps, pit_loss, ratios, periods, rates, limits,
                                  REPLAY_SIMS)
            summary = sc.summarise(sims, candidates, strategies)
            events = [(e["sc_type"], e["start_lap"], min(e["end_lap"], total_laps))
                      for e in real.filter(pl.col("session_key") == winner["session_key"]).sort("start_lap").iter_rows(named=True)]
            cum = sc.stint_table(grid)

            def replay(strategy_id: int) -> tuple[str, int]:
                _, pit_laps, compounds, _ = sc.play(strategies[strategy_id], events, cum, total_laps, pit_loss, ratios, limits)
                bounds = [0, *pit_laps, total_laps]
                return self.describe([(COMPOUNDS[c], b - a) for c, a, b in zip(compounds, bounds, bounds[1:])]), len(pit_laps)

            green_plan, green_stops = replay(results["strategy_id"][0])
            sc_plan, sc_stops = replay(summary["strategy_id"][0])
            rows.append({
                "circuit_short_name": circuit,
                "group": winner["group"],
                "neutralisations": ", ".join(f"{t} {a}-{b}" for t, a, b in events),
                "winner_strategy": self.describe(strategy),
                "green_plan_raced": green_plan,
                "sc_plan_raced": sc_plan,
                "winner_stops": len(strategy) - 1,
                "green_stops": green_stops,
                "sc_stops": sc_stops,
            })
        return pl.DataFrame(rows).with_columns(
            green_same_stops=pl.col("green_stops") == pl.col("winner_stops"),
            sc_same_stops=pl.col("sc_stops") == pl.col("winner_stops"),
        )

    def run(self, test_year: int = 2026):
        model = self.load_model()
        pit_table = MLPitLoss().pit_loss_table()
        overall = pit_table["pit_loss"].median()
        finishers, stints, _ = self.load_race_data(test_year)
        pit_losses = {c: overall for c in finishers["circuit_short_name"].unique()} | dict(
            pit_table.select("circuit_short_name", "pit_loss").iter_rows())

        drivers = self.accuracy(finishers, stints, model, pit_losses)
        summary = self.summary(drivers)
        optimal = self.optimal_vs_winner(drivers, finishers, stints, model, pit_losses, test_year)
        replay = self.safety_car_replay(finishers, stints, model, pit_losses, test_year)

        RESULTS_DIR.mkdir(exist_ok=True)
        for name, table in {"validation_drivers": drivers, "validation_summary": summary, "validation_optimal": optimal,
                            "validation_safety_car": replay}.items():
            table.write_csv(RESULTS_DIR / f"{name}.csv", float_precision=3)
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
            print(drivers.drop("session_key", "driver_number"))
            print(summary)
            print(optimal)
            print(replay)
        one_stop = optimal.filter(pl.col("optimal_pit_lap_same_compounds").is_not_null())
        print(f"Same number of stops as the winner in {optimal['same_stops'].sum()}/{optimal.height} races, "
              f"same compounds in {optimal['same_compounds'].sum()}/{optimal.height}. "
              f"1-stop winners pitted inside the simulator's window for their compounds in "
              f"{one_stop['winner_pit_in_window'].sum()}/{one_stop.height} races. "
              f"Median predicted gain from the optimal strategy: {optimal['predicted_gain_s'].median():.2f} s.")
        print(f"Raced through the real Safety Cars and VSCs, re-planning at each one: the green-flag best ends with the "
              f"winner's number of stops in {replay['green_same_stops'].sum()}/{replay.height} races, the plan picked "
              f"with Safety Cars in {replay['sc_same_stops'].sum()}/{replay.height}.")
        print(f"Saved validation tables to {RESULTS_DIR}")
        return drivers, summary, optimal
