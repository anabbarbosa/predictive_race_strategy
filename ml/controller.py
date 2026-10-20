import os
from pathlib import Path

import polars as pl 
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine

class MLController:
    def __init__(self):
        load_dotenv()
        url = (
            f"postgresql+psycopg://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
            f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
        )
        engine = create_engine(url)

    def fetch_data(self, query: str) -> pl.DataFrame:
        """Fetch data from the database using the provided SQL query."""
        in_df = pl.read_database("SELECT * FROM laps", connection=engine)

        with self.engine.connect() as connection:
            result = connection.execute(query)
            data = result.fetchall()
            columns = result.keys()
            return pl.DataFrame(data, schema=columns)

    def plot_data(self, df: pl.DataFrame, x_col: str, y_col: str, title: str):
        def lap_vs_tyre_life():
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

        def lap_per_driver():
            mid_df = in_df.filter(
                pl.col("circuit_short_name") == "Melbourne",
                pl.col("year") == 2026,
            )
            driver_colors = {
                "M VERSTAPPEN": "#3671C6", "I HADJAR": "#3671C6",
                "C LECLERC": "#E8002D", "L HAMILTON": "#E8002D",
                "G RUSSELL": "#27F4D2", "K ANTONELLI": "#27F4D2",
                "L NORRIS": "#FF8000", "O PIASTRI": "#FF8000",
                "L NORRIS": "#FF8000", "O PIASTRI": "#FF8000",
                "F ALONSO": "#229971", "L STROLL": "#229971",
                "P GASLY": "#FF87BC", "F COLAPINTO": "#FF87BC",
                "A ALBON": "#64C4FF", "C SAINZ": "#64C4FF",
                "L LAWSON": "#6692FF", "A LINDBLAD": "#6692FF",
                "E OCON": "#B6BABD", "O BEARMAN": "#B6BABD",
                "N HULKENBERG": "#52E252", "G BORTOLETO": "#52E252",
                "S PEREZ": "#1E1E1E", "V BOTTAS": "#1E1E1E",
            }

            dashed = {"I HADJAR", "L HAMILTON", "K ANTONELLI", "O PIASTRI", "L STROLL",
                    "F COLAPINTO", "C SAINZ", "A LINDBLAD", "O BEARMAN", "G BORTOLETO", "V BOTTAS"}

            fig, ax = plt.subplots(figsize=(12, 6))

            for driver, group in mid_df.group_by("broadcast_name"):
                name = driver[0]
                group = group.sort("lap_number")
                color = driver_colors.get(name)

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
            plt.show()

        def sting_by_compound():
            mid_df = in_df.filter(
                pl.col("circuit_short_name") == "Melbourne",
                pl.col("year") == 2026,
            )
            compound_colors = {
                "SOFT": "#FF0000",
                "MEDIUM": "#F504AD",
                "HARD": "#2600FF",
            }

            fig, ax = plt.subplots(figsize=(10, 6))

            for compound, group in mid_df.group_by("compound"):
                name = compound[0]
                group = group.sort("lap_number")
                ax.plot(
                    group["lap_number"],
                    group["lap_duration"],
                    label=name,
                    color=compound_colors.get(name),
                    linewidth=1.5,
                    alpha=0.8,
                )

            ax.set_xlabel("Lap number")
            ax.set_ylabel("Lap duration (s)")
            ax.set_title("Lap duration by lap number per compound")
            ax.legend(title="Compound")

        def average_lap_by_team():
            mid_df = in_df.filter(
                pl.col("circuit_short_name") == "Melbourne",
                pl.col("year") == 2026,
            )
            team_colors = {
                "RED BULL RACING": "#3671C6",
                "FERRARI": "#E8002D",
                "MERCEDES": "#27F4D2",
                "MCLAREN": "#FF8000",
                "ALPINE": "#229971",
                "ALFA ROMEO": "#FF87BC",
                "ASTON MARTIN": "#64C4FF",
                "HAAS F1 TEAM": "#6692FF",
                "ALPHATAURI": "#B6BABD",
                "WILLIAMS": "#52E252",
            }

            fig, ax = plt.subplots(figsize=(12, 6))

            for team, group in mid_df.group_by("team"):
                name = team[0]
                group = group.sort("lap_number")
                ax.plot(
                    group["lap_number"],
                    group["lap_duration"],
                    label=name,
                    color=team_colors.get(name),
                    linewidth=1.5,
                    alpha=0.8,
                )

            ax.set_xlabel("Lap number")
            ax.set_ylabel("Lap duration (s)")
            ax.set_title("Average lap duration by lap number per team")
            ax.legend(title="Team", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

            
    def save_plot(self, df: pl.DataFrame, x_col: str, y_col: str, title: str, file_path: str):
        """Save the plot to a file."""
        plt.figure(figsize=(10, 6))
        plt.plot(df[x_col], df[y_col], marker='o')
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid()
        plt.savefig(file_path)