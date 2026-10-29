from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from ml.simulate import MIN_STINT, MLSimulate

MAX_LAPS_LATER = 10
DIVERGING = LinearSegmentedColormap.from_list("undercut", ["#1c5cab", "#f0efec", "#e34948"])


class MLUndercut(MLSimulate):
    def undercut_gain(self, grid: np.ndarray, old: str, new: str, pit_lap: int, laps_later: int) -> float:
        last = pit_lap + laps_later + 1  # B's out-lap
        a = self.stint_time(grid, old, 1, pit_lap) + self.stint_time(grid, new, pit_lap + 1, last)
        b = self.stint_time(grid, old, 1, pit_lap + laps_later) + self.stint_time(grid, new, last, last)
        return b - a

    def gain_table(self, grid: np.ndarray, old: str, new: str, total_laps: int, limits: dict[str, int]) -> pl.DataFrame:
        rows = [
            (pit_lap, laps_later, self.undercut_gain(grid, old, new, pit_lap, laps_later))
            for pit_lap in range(MIN_STINT, min(limits.get(old, total_laps), total_laps - MIN_STINT) + 1)
            for laps_later in range(1, MAX_LAPS_LATER + 1)
            # B's old tyres must last until it pits, and the race must still have laps after B's out-lap
            if pit_lap + laps_later <= limits.get(old, total_laps) and pit_lap + laps_later + 1 < total_laps
        ]
        return pl.DataFrame(rows, schema=["pit_lap", "laps_later", "gain_s"], orient="row")

    def plot(self, table: pl.DataFrame, old: str, new: str, circuit: str, team: str, output_dir: str = "plots"):
        pivot = table.pivot("pit_lap", index="laps_later", values="gain_s").sort("laps_later")
        values = pivot.drop("laps_later").to_numpy()
        pit_laps = [int(col) for col in pivot.columns[1:]]
        limit = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 0.01)

        fig, ax = plt.subplots(figsize=(12, 5))
        image = ax.imshow(values, aspect="auto", origin="lower", cmap=DIVERGING,
                          norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
                          extent=(pit_laps[0] - 0.5, pit_laps[-1] + 0.5, 0.5, MAX_LAPS_LATER + 0.5))
        bar = fig.colorbar(image, ax=ax)
        bar.set_label("Seconds car A gains (red: undercut works, blue: overcut)")
        ax.set_xlabel("Lap car A pits (in-lap)")
        ax.set_ylabel("Laps car B stays out after A")
        ax.set_yticks(range(1, MAX_LAPS_LATER + 1))
        ax.set_title(f"Undercut vs overcut, {old} -> {new} — {circuit}, {team}")
        file_path = Path(output_dir) / "undercut.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {file_path}")

    def run(self, circuit: str, team: str, year: int, old: str = "MEDIUM", new: str = "HARD",
            total_laps: int | None = None, track_temperature: float | None = None, plot: bool = True) -> pl.DataFrame:
        self.check_strategy([(old, 1), (new, 1)], 2)
        context = self.race_context(circuit)
        total_laps = total_laps or context["total_laps"]
        track_temperature = track_temperature if track_temperature is not None else context["track_temperature"]
        print(f"{circuit}, {team}, {year}: {total_laps} laps, both cars {old} -> {new}")
        model = self.load_model()
        limits = self.stint_limits(year)
        grid = self.lap_delta_grid(model, total_laps, team, circuit, track_temperature, year)
        table = self.gain_table(grid, old, new, total_laps, limits)

        one_lap = table.filter(pl.col("laps_later") == 1)
        best = one_lap.sort("gain_s", descending=True).row(0, named=True)
        worst = one_lap.sort("gain_s").row(0, named=True)
        print(f"Pitting one lap before the car in front gains {one_lap['gain_s'].median():.2f} s (median over pit laps), "
              f"from {worst['gain_s']:+.2f} s (lap {worst['pit_lap']}) to {best['gain_s']:+.2f} s (lap {best['pit_lap']}).")
        print(f"So the undercut works from up to ~{max(best['gain_s'], 0):.1f} s behind; "
              f"the overcut (staying out) pays off in {(table['gain_s'] < 0).mean():.0%} of the pit lap x wait combinations.")
        with pl.Config(tbl_rows=MAX_LAPS_LATER, float_precision=2):
            print(table.group_by("laps_later").agg(
                median_gain_s=pl.col("gain_s").median(), min_gain_s=pl.col("gain_s").min(), max_gain_s=pl.col("gain_s").max(),
            ).sort("laps_later"))
        if plot:
            self.plot(table, old, new, circuit, team)
        return table
