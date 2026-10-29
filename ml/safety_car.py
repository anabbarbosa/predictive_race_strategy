from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ml.pit_loss import MLPitLoss
from ml.simulate import MIN_STINT, MLSimulate, Strategy

N_SIMS = 1000
TOP_STRATEGIES = 20
PRIOR_RACES = 3


class MLSafetyCar(MLSimulate):
    def neutral_periods(self) -> pl.DataFrame:
        return self.fetch_data("""
            SELECT p.*, p.end_lap - p.start_lap + 1 AS n_laps
            FROM safety_car_periods p
            WHERE p.session_key NOT IN (SELECT session_key FROM results WHERE suspended)
        """)

    def deployment_rates(self, periods: pl.DataFrame, circuit: str) -> dict[str, float]:
        races = self.fetch_data("SELECT DISTINCT session_key, circuit_short_name FROM results WHERE NOT suspended")
        rates = {}
        for sc_type in ("SC", "VSC"):
            counts = races.join(
                periods.filter(pl.col("sc_type") == sc_type).group_by("session_key").len(),
                on="session_key", how="left",
            ).with_columns(pl.col("len").fill_null(0))
            overall = counts["len"].mean()
            own = counts.filter(pl.col("circuit_short_name") == circuit)["len"]
            rates[sc_type] = (own.sum() + PRIOR_RACES * overall) / (own.len() + PRIOR_RACES)
        return rates

    def neutral_pit_loss(self) -> pl.DataFrame:
        periods = self.neutral_periods()
        stops = self.fetch_data("""
            SELECT session_key, driver_number, circuit_short_name, lap_number, in_lap_duration, out_lap_duration
            FROM pit_stops
            WHERE in_lap_duration IS NOT NULL AND out_lap_duration IS NOT NULL
              AND (stop_duration IS NULL OR stop_duration <= 5)
        """)
        laps = self.fetch_data("SELECT session_key, driver_number, lap_number, lap_duration FROM race_laps")
        # The field's lap time on each lap, leaving out the cars that pitted on it or came out of the pits
        stop_laps = stops.select("session_key", "driver_number", "lap_number")
        field = (
            laps.join(stop_laps, on=["session_key", "driver_number", "lap_number"], how="anti")
            .join(stop_laps.with_columns(pl.col("lap_number") + 1), on=["session_key", "driver_number", "lap_number"], how="anti")
            .group_by("session_key", "lap_number")
            .agg(field_lap=pl.col("lap_duration").median())
        )
        green = MLPitLoss().pit_loss_table().select("circuit_short_name", "pit_loss")
        overall = green["pit_loss"].median()
        neutral = (
            stops.join_where(
                periods.select("session_key", "sc_type", "start_lap", "end_lap"),
                pl.col("session_key") == pl.col("session_key_right"),
                pl.col("lap_number").is_between(pl.col("start_lap"), pl.col("end_lap") - 1),
            )
            .join(field.rename({"field_lap": "field_in"}), on=["session_key", "lap_number"])
            .join(field.with_columns(pl.col("lap_number") - 1).rename({"field_lap": "field_out"}), on=["session_key", "lap_number"])
            .join(green, on="circuit_short_name", how="left")
            .with_columns(
                neutral_loss=pl.col("in_lap_duration") + pl.col("out_lap_duration") - pl.col("field_in") - pl.col("field_out"),
                green_loss=pl.col("pit_loss").fill_null(overall),
            )
            .with_columns(ratio=pl.col("neutral_loss") / pl.col("green_loss"))
        )
        return neutral.group_by("sc_type").agg(
            n_stops=pl.len(),
            neutral_loss=pl.col("neutral_loss").median(),
            green_loss=pl.col("green_loss").median(),
            ratio=pl.col("ratio").median(),
        ).sort("sc_type")

    def sample_neutral_laps(self, rng: np.random.Generator, periods: pl.DataFrame, rates: dict[str, float],
                            total_laps: int) -> list[tuple[str, int, int]]:
        events = []
        for sc_type, rate in rates.items():
            history = periods.filter(pl.col("sc_type") == sc_type)
            for _ in range(rng.poisson(rate)):
                row = history.row(int(rng.integers(history.height)), named=True)
                start = max(1, round(row["start_lap"] / row["race_laps"] * total_laps))
                events.append((sc_type, start, min(total_laps, start + row["n_laps"] - 1)))
        return sorted(events, key=lambda event: event[1])

    def react(self, pit_laps: list[int], compounds: list[str], events: list[tuple[str, int, int]],
              total_laps: int, limits: dict[str, int]) -> tuple[list[int], list[str]]:
        pit_laps = list(pit_laps)
        for _, start, end in events:
            done = [lap for lap in pit_laps if lap < start]
            upcoming = [i for i, lap in enumerate(pit_laps) if lap >= start]
            if end <= start or not upcoming or pit_laps[upcoming[0]] <= end - 1:
                continue 
            i = upcoming[0]
            previous = done[-1] if done else 0
            following = pit_laps[i + 1] if i + 1 < len(pit_laps) else total_laps
            if start - previous >= MIN_STINT and following - start <= limits.get(compounds[i + 1], total_laps):
                pit_laps[i] = start
        under = [next((t for t, s, e in events if s <= lap <= e - 1), "") for lap in pit_laps]
        return pit_laps, under

    def candidates(self, results: pl.DataFrame) -> pl.DataFrame:
        best_per_plan = results.group_by("compounds", "n_stops").agg(pl.all().sort_by("total_time").first())
        return pl.concat([results.head(TOP_STRATEGIES), best_per_plan.select(results.columns)]).unique("strategy_id").sort("total_time")

    def monte_carlo(self, candidates: pl.DataFrame, strategies: list[Strategy], grid: np.ndarray, total_laps: int,
                    pit_loss: float, ratios: dict[str, float], periods: pl.DataFrame, rates: dict[str, float],
                    limits: dict[str, int], n_sims: int, seed: int = 42) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        races = [self.sample_neutral_laps(rng, periods, rates, total_laps) for _ in range(n_sims)]
        rows = []
        for strategy_id in candidates["strategy_id"]:
            strategy = strategies[strategy_id]
            compounds = [compound for compound, _ in strategy]
            for sim, events in enumerate(races):
                pit_laps, under = self.react(self.pit_laps(strategy), compounds, events, total_laps, limits)
                costs = [pit_loss * ratios.get(sc_type, 1.0) for sc_type in under]
                rows.append((strategy_id, sim, self.race_time(grid, compounds, pit_laps, total_laps, costs),
                             any(under), len(events) > 0))
        sims = pl.DataFrame(rows, schema=["strategy_id", "sim", "total_time", "cheap_stop", "neutralised"], orient="row")
        return sims.with_columns(gap=pl.col("total_time") - pl.col("total_time").min().over("sim"))

    def summarise(self, sims: pl.DataFrame, candidates: pl.DataFrame, strategies: list[Strategy]) -> pl.DataFrame:
        green_best = candidates["total_time"].min()
        best = pl.col("gap") == 0
        return (
            sims.with_columns(win=best.cast(pl.Float64) / best.sum().over("sim"))
            .group_by("strategy_id")
            .agg(
                expected_s=pl.col("total_time").mean(),
                mean_regret_s=pl.col("gap").mean(),
                p90_regret_s=pl.col("gap").quantile(0.9),
                win_pct=pl.col("win").mean() * 100,
                cheap_stop_pct=pl.col("cheap_stop").mean() * 100,
            )
            .join(candidates.select("strategy_id", "n_stops", "total_time"), on="strategy_id")
            .with_columns(
                strategy=pl.col("strategy_id").map_elements(lambda i: self.describe(strategies[i]), return_dtype=pl.String),
                green_gap_s=pl.col("total_time") - green_best,
                expected_gap_s=pl.col("expected_s") - pl.col("expected_s").min(),
            )
            .select("strategy", "n_stops", "green_gap_s", "expected_gap_s", "mean_regret_s", "p90_regret_s",
                    "win_pct", "cheap_stop_pct", "strategy_id")
            .sort("expected_gap_s")
        )

    def plot(self, sims: pl.DataFrame, summary: pl.DataFrame, circuit: str, team: str, output_dir: str = "plots"):
        top = summary.head(6)
        data = [sims.filter(pl.col("strategy_id") == i)["gap"].to_numpy() for i in top["strategy_id"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data, vert=False, tick_labels=top["strategy"].to_list(), whis=(5, 95), showfliers=False,
                   showmeans=True, medianprops={"color": "#2a78d6", "linewidth": 2},
                   meanprops={"marker": "^", "markerfacecolor": "#eb6834", "markeredgecolor": "#eb6834"})
        ax.invert_yaxis()
        ax.set_xlabel("Seconds lost to the best strategy of the same simulated race (whiskers 5-95%, blue = median, triangle = mean)")
        ax.set_title(f"Safety Car Monte Carlo, {sims['sim'].n_unique()} races — {circuit}, {team}")
        file_path = Path(output_dir) / "safety_car_mc.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {file_path}")

    def run(self, circuit: str, team: str, year: int, total_laps: int | None = None,
            track_temperature: float | None = None, pit_loss: float | None = None, max_stops: int = 2,
            n_sims: int = N_SIMS, plot: bool = True) -> pl.DataFrame:
        context = self.race_context(circuit)
        total_laps = total_laps or context["total_laps"]
        track_temperature = track_temperature if track_temperature is not None else context["track_temperature"]
        pit_loss = pit_loss if pit_loss is not None else context["pit_loss"]
        model = self.load_model()

        periods = self.neutral_periods()
        rates = self.deployment_rates(periods, circuit)
        neutral = self.neutral_pit_loss()
        ratios = dict(neutral.select("sc_type", "ratio").iter_rows())
        print(f"{circuit}, {team}, {year}: {total_laps} laps, pit loss {pit_loss:.2f} s")
        print(f"Expected deployments per race: SC {rates['SC']:.2f}, VSC {rates['VSC']:.2f}")
        print(neutral)

        limits = self.stint_limits(year)
        results, strategies = self.best_strategies(model, total_laps, team, circuit, track_temperature, year,
                                                   pit_loss, max_stops, limits)
        candidates = self.candidates(results)
        grid = self.lap_delta_grid(model, total_laps, team, circuit, track_temperature, year)
        check = candidates.row(0, named=True)
        lookup = self.race_time(grid, [c for c, _ in strategies[check["strategy_id"]]], check["pit_laps"], total_laps,
                                [pit_loss] * check["n_stops"])
        print(f"Grid lookup vs simulator for the best strategy: {lookup - check['total_time']:+.4f} s")

        sims = self.monte_carlo(candidates, strategies, grid, total_laps, pit_loss, ratios, periods, rates, limits, n_sims)
        summary = self.summarise(sims, candidates, strategies)
        self.sims = sims
        neutralised = sims.group_by("sim").agg(pl.col("neutralised").first())["neutralised"].mean()
        print(f"{len(candidates)} strategies x {n_sims} races, {neutralised:.0%} of them with at least one SC or VSC")
        with pl.Config(tbl_rows=15, tbl_width_chars=200, float_precision=2):
            print(summary.drop("strategy_id").head(15))
        best_green, best_expected = summary.sort("green_gap_s").row(0, named=True), summary.row(0, named=True)
        if best_green["strategy_id"] == best_expected["strategy_id"]:
            print(f"The green-flag best {best_green['strategy']} is also the best on average with Safety Cars.")
        else:
            print(f"With Safety Cars the best on average is {best_expected['strategy']}, "
                  f"{best_green['expected_gap_s']:.2f} s better than the green-flag best {best_green['strategy']}.")
        if plot:
            self.plot(sims, summary, circuit, team)
        return summary
