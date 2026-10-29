from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

from controller import ProjectUtils


class MLController(ProjectUtils):
    def fetch_data(self, query: str = "SELECT * FROM laps") -> pl.DataFrame:
        return pl.read_database(query, connection=self.engine, infer_schema_length=None)

    def lap_vs_tyre_life(self, in_df: pl.DataFrame, x_col: str, y_col: str, title: str):
        compound_colors = {
            "SOFT": "#FF0000",
            "MEDIUM": "#F504AD",
            "HARD": "#2600FF",
        }
        fig, ax = plt.subplots(figsize=(10, 6))

        for compound, group in in_df.group_by("compound"):
            name = compound[0]
            ax.scatter(
                group["tyre_life"],
                group["lap_duration"],
                label=name,
                color=compound_colors.get(name),
                s=10,
                alpha=0.8,
            )

        ax.set_xlabel("Tyre life (laps)")
        ax.set_ylabel("Lap duration (s)")
        ax.set_title("Lap duration vs tyre life by compound")
        ax.legend(title="Compound")

    def lap_per_driver(self, in_df: pl.DataFrame):
        mid_df = in_df.filter(
            pl.col("circuit_short_name") == "Melbourne",
            pl.col("year") == 2026,
        )
        drivers = mid_df.select("broadcast_name", "driver_number", "team_name").unique()
        dashed = set(
            drivers.filter(
                pl.col("driver_number").rank("ordinal").over("team_name") > 1
            )["broadcast_name"]
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        for (name, colour), group in mid_df.group_by("broadcast_name", "team_colour"):
            group = group.sort("lap_number")
            color = f"#{colour}"

            ax.plot(
                group["lap_number"],      # x
                group["lap_duration"],    # y
                label=name,
                color=color,
                linestyle="--" if name in dashed else "-",
                linewidth=1.2,
                alpha=1,
            )

            pits = group.filter(pl.col("stint_number") > 1).group_by("stint_number").agg(pl.all().first())
            pit_points = group.join(pits, on=["stint_number", "lap_number"])
            ax.scatter(pit_points["lap_number"], pit_points["lap_duration"],
                    color=color, marker="v", s=40, edgecolor="black", zorder=3)

        ax.set_xlabel("Lap number")
        ax.set_ylabel("Lap duration (s)")
        ax.set_title("Lap duration by lap number per driver (▼ = first lap after a pit stop)")
        ax.legend(title="Driver", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    def stint_by_compound(self, in_df: pl.DataFrame):
        compound_colors = {
            "SOFT": "#FF0000",
            "MEDIUM": "#F504AD",
            "HARD": "#2600FF",
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        bins = range(1, in_df["stint_length"].max() + 6, 5)  # same 5-lap bins for every compound
        order = ["SOFT", "MEDIUM", "HARD"]
        ax.hist(
            [in_df.filter(pl.col("compound") == c)["stint_length"] for c in order],
            label=order,
            color=[compound_colors[c] for c in order],
            bins=bins,
        )

        ax.set_xlabel("Stint length (laps)")
        ax.set_ylabel("Number of stints")
        ax.set_xticks(bins)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_title("Stint length by compound (all races)")
        ax.legend(title="Compound")

    def average_lap_by_team(self, in_df: pl.DataFrame):
        mid_df = in_df.filter(
            pl.col("circuit_short_name") == "Melbourne",
            pl.col("year") == 2026,
        )
        team_avg = (
            mid_df.group_by("team_name")
            .agg(
                average_laptime=pl.col("lap_duration").mean(),
                team_colour=pl.col("team_colour").last(),
            )
            .sort("average_laptime")
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            team_avg["team_name"],
            team_avg["average_laptime"],
            color=[f"#{c}" for c in team_avg["team_colour"]],
        )
        ax.set_xlabel("Average lap duration (s)")
        ax.set_title("Average lap time per team (Melbourne 2026)")
        ax.set_xlim(team_avg["average_laptime"].min() - 0.5, team_avg["average_laptime"].max() + 0.5)
        ax.invert_yaxis()

    def save_plot(self, laps_df: pl.DataFrame, stints_df: pl.DataFrame, output_dir: str = "plots"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plots = {
            "lap_vs_tyre_life": lambda: self.lap_vs_tyre_life(laps_df, "tyre_life", "lap_duration", "Lap duration vs tyre life"),
            "lap_per_driver": lambda: self.lap_per_driver(laps_df),
            "stint_by_compound": lambda: self.stint_by_compound(stints_df),
            "average_lap_by_team": lambda: self.average_lap_by_team(laps_df),
        }
        for name, draw in plots.items():
            draw()
            file_path = out / f"{name}.png"
            plt.savefig(file_path, bbox_inches="tight")
            plt.close()
            print(f"Saved {file_path}")
