from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from ml.controller import MLController

MIN_STOPS = 5


class MLPitLoss(MLController):
    def pit_loss_table(self) -> pl.DataFrame:
        stops = self.fetch_data("SELECT * FROM pit_stops WHERE is_representative")
        overall = stops["pit_loss"].median()
        return (
            stops.group_by("circuit_short_name")
            .agg(
                n_stops=pl.len(),
                lane_duration=pl.col("lane_duration").median(),
                stop_duration=pl.col("stop_duration").median(),
                pit_loss_b=pl.col("pit_loss").median(),
            )
            .with_columns(
                a_minus_b=pl.col("lane_duration") - pl.col("pit_loss_b"),
                pit_loss=pl.when(pl.col("n_stops") >= MIN_STOPS).then(pl.col("pit_loss_b")).otherwise(pl.lit(overall)),
            )
            .sort("pit_loss")
        )

    def pit_loss(self, circuit: str) -> float:
        table = self.pit_loss_table()
        row = table.filter(pl.col("circuit_short_name") == circuit)
        return row["pit_loss"].item() if row.height else table["pit_loss"].median()

    def plot(self, table: pl.DataFrame, output_dir: str = "plots"):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(table["lane_duration"], table["circuit_short_name"], label="A: lane_duration (pit entry to exit)", color="#9AA0A6")
        ax.scatter(table["pit_loss_b"], table["circuit_short_name"], label="B: race time lost (used as PIT_LOSS)", color="#E10600")
        ax.hlines(table["circuit_short_name"], table["lane_duration"], table["pit_loss_b"], color="#9AA0A6", linewidth=1)
        ax.set_xlabel("Seconds (median over representative stops)")
        ax.set_title("Pit-stop loss per circuit: method A vs method B")
        ax.legend(loc="lower right")
        file_path = Path(output_dir) / "pit_loss.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {file_path}")

    def run(self) -> pl.DataFrame:
        stops = self.fetch_data("SELECT is_representative, stop_duration FROM pit_stops")
        print(f"{stops.height} pit stops, {stops['is_representative'].sum()} representative "
              f"(green flag, dry, stop_duration <= 5 s, at least 2 clean reference laps)")
        table = self.pit_loss_table()
        with pl.Config(tbl_rows=-1):
            print(table)
        print(f"Median over all circuits: A {table['lane_duration'].median():.2f} s, B {table['pit_loss_b'].median():.2f} s, "
              f"stationary {table['stop_duration'].median():.2f} s")
        self.plot(table)
        return table
