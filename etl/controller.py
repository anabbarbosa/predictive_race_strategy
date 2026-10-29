import time
from pathlib import Path
import requests
import polars as pl
from sqlalchemy import inspect, text
from controller import UtilsABC

DATA_DIR = Path(__file__).resolve().parent / "data"

class ETLController(UtilsABC):
    def __init__(self):
        super().__init__()
        self.session_keys = []

    SESSION_ENDPOINTS = [
        ("drivers", "drivers", "drivers"),
        ("laps", "laps", "laps"),
        ("stints", "stints", "stints"),
        ("pits", "pit", "pits"),
        ("weather", "weather", "weather"),
        ("race control", "race_control", "race_control"),
    ]
    MAX_RETRIES = 5

    def extract(self):
        self.fetch_session_data()
        self.session_list()
        for label, endpoint, folder in self.SESSION_ENDPOINTS:
            self.fetch_per_session(label, endpoint, folder)
        print("Data extraction completed.")

    def get_with_retry(self, url):
        for attempt in range(self.MAX_RETRIES + 1):
            response = requests.get(url)
            if response.status_code != 429 or attempt == self.MAX_RETRIES:
                return response
            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
            print(f"Rate limited (429), retrying in {wait}s...")
            time.sleep(wait)

    def fetch_session_data(self):
        print("Fetching sessions data from 2024 to 2026...")
        url = f"https://api.openf1.org/v1/sessions?year>=2024&year<=2026&csv=true"
        response = self.get_with_retry(url)
        if response.status_code == 200:
            with open(f"{DATA_DIR}/raw/sessions/sessions.csv", "wb") as f:
                f.write(response.content)
            print(f"Data for sessions from 2024 to 2026 saved successfully.")
        else:
            print(f"Failed to fetch data for sessions from 2024 to 2026. Status code: {response.status_code}")

    def session_list(self):
        print("Filtering sessions for races that were not cancelled...")
        df = pl.read_csv(f"{DATA_DIR}/raw/sessions/sessions.csv")
        df = df.filter((pl.col('session_name') == 'Race') & ~pl.col('is_cancelled'))
        self.session_keys = df['session_key'].to_list()
        return self.session_keys

    def fetch_per_session(self, label, endpoint, folder):
        print(f"Fetching {label} data...")
        for session_key in self.session_keys:
            url = f"https://api.openf1.org/v1/{endpoint}?session_key={session_key}&csv=true"
            response = self.get_with_retry(url)
            if response.status_code == 200:
                with open(f"{DATA_DIR}/raw/{folder}/{folder}_{session_key}.csv", "wb") as f:
                    f.write(response.content)
                print(f"Data for {label} on session {session_key} saved successfully.")
            else:
                print(f"Failed to fetch data for session {session_key}. Status code: {response.status_code}")
            time.sleep(1)

    def transform(self):
        self.read_raw_data()
        mid_df = self.build_lap_table()
        mid_df = self.add_tyre_life(mid_df)
        mid_df = self.add_lap_flags(mid_df)
        mid_df = self.add_safety_car_flags(mid_df)
        final_df = self.clean_laps(mid_df)
        self.write_output(final_df, "laps")
        self.write_output(self.build_stint_table(), "stints")
        self.write_output(self.build_pit_stop_table(mid_df), "pit_stops")
        self.write_output(self.build_result_table(mid_df), "results")
        self.write_output(self.build_safety_car_periods(), "safety_car_periods")
        self.write_output(self.build_race_lap_table(), "race_laps")

    def read_raw_data(self):
        print("Reading raw data from CSV files...")
        self.laps_df = pl.read_csv(f"{DATA_DIR}/raw/laps/laps_[0-9]*.csv").with_columns(
            pl.col("date_start").str.to_datetime(time_zone="UTC")
        )
        self.stints_df = pl.read_csv(f"{DATA_DIR}/raw/stints/stints_[0-9]*.csv")
        self.session_df = pl.read_csv(f"{DATA_DIR}/raw/sessions/sessions.csv")
        self.drivers_df = pl.read_csv(f"{DATA_DIR}/raw/drivers/drivers_[0-9]*.csv")
        self.race_controls_df = pl.read_csv(f"{DATA_DIR}/raw/race_control/race_control_[0-9]*.csv")
        self.pits_df = pl.read_csv(f"{DATA_DIR}/raw/pits/pits_[0-9]*.csv", infer_schema_length=None).with_columns(
            pl.col("lane_duration", "stop_duration").cast(pl.Float64)
        )
        self.weather_df = pl.read_csv(f"{DATA_DIR}/raw/weather/weather_[0-9]*.csv").with_columns(
            pl.col("date").str.to_datetime(time_zone="UTC")
        )

    def build_lap_table(self):
        print("Building lap table by joining laps, stints, drivers, sessions, and weather data...")
        mid_df = (
            self.laps_df.join_where(
                self.stints_df,
                pl.col("session_key") == pl.col("session_key_right"),
                pl.col("driver_number") == pl.col("driver_number_right"),
                pl.col("lap_number").is_between(pl.col("lap_start"), pl.col("lap_end")),
            )
            .drop("session_key_right", "driver_number_right", "meeting_key_right")
            .join(self.drivers_df, on=["session_key", "driver_number"], how="left")
            .join(self.session_df.drop("meeting_key").rename({"date_start": "session_start_date", "date_end": "session_end_date"}), on="session_key", how="left")
            .sort("date_start")
            .join_asof(self.weather_df.drop("meeting_key").sort("date"), left_on="date_start", right_on="date", by="session_key", strategy="backward")
            .drop("meeting_key_right", "country_code_right")
        )
        return mid_df

    def add_tyre_life(self, mid_df):
        mid_df = mid_df.with_columns(
            tyre_life=pl.col("tyre_age_at_start") + (pl.col("lap_number") - pl.col("lap_start"))
        )
        return mid_df

    def add_lap_flags(self, mid_df):
        mid_df = mid_df.sort("session_key", "driver_number", "lap_number").with_columns(
            is_out_lap=pl.col("is_pit_out_lap").fill_null(False),
            is_in_lap=pl.col("is_pit_out_lap")
                .shift(-1)
                .over("session_key", "driver_number")
                .fill_null(False),
            no_lap_time=pl.col("lap_duration").is_null()
        )
        return mid_df

    def build_safety_car_periods(self):
        race_laps = self.laps_df.group_by("session_key").agg(race_laps=pl.col("lap_number").max())
        return (
            self.race_controls_df.filter(pl.col("category") == "SafetyCar")
            .sort("session_key", "date")
            .with_columns(
                sc_type=pl.when(pl.col("message").str.starts_with("VSC"))
                    .then(pl.lit("VSC"))
                    .otherwise(pl.lit("SC")),
                end_lap=pl.col("lap_number").shift(-1).over("session_key"),
            )
            .filter(pl.col("message").str.ends_with("DEPLOYED"))
            .join(race_laps, on="session_key", how="left")
            .with_columns(end_lap=pl.min_horizontal(pl.col("end_lap").fill_null(999), "race_laps"))
            .join(self.session_df.select("session_key", "circuit_short_name", "year"), on="session_key", how="left")
            .select("session_key", "circuit_short_name", "year", "race_laps", "sc_type",
                    pl.col("lap_number").alias("start_lap"), "end_lap")
        )

    def build_race_lap_table(self):
      return (
            self.laps_df.filter(pl.col("date_start").is_not_null())
            .select("session_key", "driver_number", "lap_number", "date_start", "lap_duration", "is_pit_out_lap")
            .join(self.session_df.select("session_key", "circuit_short_name", "year"), on="session_key", how="left")
            .sort("session_key", "driver_number", "lap_number")
        )

    def add_safety_car_flags(self, mid_df):
        sc_laps = (
            self.build_safety_car_periods()
            .select(
                "session_key", "sc_type",
                lap_number=pl.int_ranges("start_lap", pl.col("end_lap") + 1),
            )
            .explode("lap_number")
            .unique(["session_key", "lap_number"])
        )

        mid_df = (
            mid_df.join(sc_laps, on=["session_key", "lap_number"], how="left")
            .with_columns(
                is_safety_car=(pl.col("sc_type") == "SC").fill_null(False),
                is_vsc=(pl.col("sc_type") == "VSC").fill_null(False),
            )
            .drop("sc_type")
        )
        return mid_df

    def clean_laps(self, mid_df):
        final_df = mid_df.filter(
            ~pl.col("no_lap_time"),
            pl.col("lap_number") != 1,
            ~pl.col("is_out_lap"),
            ~pl.col("is_in_lap"),
            ~pl.col("is_safety_car"),
            ~pl.col("is_vsc"),
            ~pl.col("compound").is_in(["WET", "INTERMEDIATE"]),
            pl.col("rainfall") == 0,
        )
        final_df = final_df.filter(
            pl.col("lap_duration") <= 1.07 * pl.col("lap_duration").median().over("session_key", "driver_number")
        )
        print(f"Filtered {mid_df.height - final_df.height} laps based on cleaning criteria.")
        print(final_df.head(20))
        print(f"{mid_df.height} laps processed, {final_df.height} laps after cleaning.")
        return final_df

    def build_stint_table(self):
        print("Building stint table from stints, drivers and sessions data...")
        sessions = self.session_df.select("session_key", "circuit_short_name", "year")
        drivers = self.drivers_df.select("session_key", "driver_number", "broadcast_name", "team_name", "team_colour")
        stints_df = (
            self.stints_df.filter(pl.col("compound").is_not_null())
            .join(drivers, on=["session_key", "driver_number"], how="left")
            .join(sessions, on="session_key", how="left")
            .with_columns(stint_length=pl.col("lap_end") - pl.col("lap_start") + 1)
            .sort("session_key", "driver_number", "stint_number")
        )
        print(f"{stints_df.height} stints built.")
        return stints_df

    def build_pit_stop_table(self, mid_df):
        print("Building pit stop table...")
        keys = ["session_key", "driver_number"]
        laps = mid_df.select(
            *keys, "lap_number", "lap_duration", "is_safety_car", "is_vsc", "rainfall", "compound",
            "circuit_short_name", "year",
        )
        clean = mid_df.filter(
            ~pl.col("no_lap_time"),
            pl.col("lap_number") != 1,
            ~pl.col("is_out_lap"),
            ~pl.col("is_in_lap"),
            ~pl.col("is_safety_car"),
            ~pl.col("is_vsc"),
            pl.col("rainfall") == 0,
        ).select(*keys, pl.col("lap_number").alias("ref_lap"), pl.col("lap_duration").alias("ref_duration"))

        stops = self.pits_df.select(*keys, "lap_number", "lane_duration", "stop_duration")
        reference = (
            stops.join_where(
                clean,
                pl.col("session_key") == pl.col("session_key_right"),
                pl.col("driver_number") == pl.col("driver_number_right"),
                pl.col("ref_lap").is_between(pl.col("lap_number") - 3, pl.col("lap_number") + 4),
            )
            .group_by(*keys, "lap_number")
            .agg(ref_duration=pl.col("ref_duration").median(), n_ref_laps=pl.len())
        )
        in_lap = laps.rename({"lap_duration": "in_lap_duration", "is_safety_car": "in_sc", "is_vsc": "in_vsc"})
        out_lap = laps.select(
            *keys,
            (pl.col("lap_number") - 1).alias("lap_number"),
            pl.col("lap_duration").alias("out_lap_duration"),
            pl.col("compound").alias("new_compound"),
            (pl.col("is_safety_car") | pl.col("is_vsc")).alias("out_neutralised"),
        )
        pit_stops = (
            stops.join(in_lap, on=[*keys, "lap_number"], how="left")
            .join(out_lap, on=[*keys, "lap_number"], how="left")
            .join(reference, on=[*keys, "lap_number"], how="left")
            .with_columns(
                pit_loss=pl.col("in_lap_duration") + pl.col("out_lap_duration") - 2 * pl.col("ref_duration"),
            )
            .with_columns(
                is_representative=(
                    pl.col("pit_loss").is_not_null()
                    & (pl.col("n_ref_laps") >= 2)
                    & ~pl.col("in_sc") & ~pl.col("in_vsc") & ~pl.col("out_neutralised")
                    & (pl.col("rainfall") == 0)
                    & ~pl.col("new_compound").is_in(["WET", "INTERMEDIATE"])
                    & (pl.col("stop_duration").is_null() | (pl.col("stop_duration") <= 5))
                    & (pl.col("lane_duration") < 60)
                ).fill_null(False)
            )
            .select(
                *keys, "circuit_short_name", "year", "lap_number", "lane_duration", "stop_duration",
                "in_lap_duration", "out_lap_duration", "ref_duration", "n_ref_laps", "pit_loss", "is_representative",
            )
            .sort(*keys, "lap_number")
        )
        print(f"{pit_stops.height} pit stops built, {pit_stops['is_representative'].sum()} representative.")
        return pit_stops

    def build_result_table(self, mid_df):
        print("Building result table...")
        keys = ["session_key", "driver_number"]
        neutralised = mid_df.group_by("session_key").agg(
            safety_car_laps=pl.col("lap_number").filter(pl.col("is_safety_car")).n_unique(),
            vsc_laps=pl.col("lap_number").filter(pl.col("is_vsc")).n_unique(),
            had_rain=(pl.col("rainfall") > 0).any(),
        )
        timed = pl.col("lap_number") > 1
        suspended = (
            self.laps_df.filter(pl.col("date_start").is_not_null())
            .sort(*keys, "lap_number")
            .group_by(*keys)
            .agg(stopped=(pl.col("date_start").diff().dt.total_seconds() > 300).any())
            .group_by("session_key")
            .agg(suspended=pl.col("stopped").mean() > 0.5)
        )
        results = (
            self.laps_df.filter(pl.col("date_start").is_not_null())
            .group_by(*keys)
            .agg(
                laps_completed=pl.col("lap_number").max(),
                timed_from_lap=pl.col("lap_number").filter(timed).min(),
                first_timed_start=pl.col("date_start").filter(timed).sort_by(pl.col("lap_number").filter(timed)).first(),
                last_lap_start=pl.col("date_start").sort_by("lap_number").last(),
            )
            .sort("session_key", "laps_completed", "last_lap_start", descending=[False, True, False])
            .with_columns(
                race_time=(pl.col("last_lap_start") - pl.col("first_timed_start")).dt.total_milliseconds() / 1000,
                position=pl.int_range(1, pl.len() + 1).over("session_key"),
            )
            .join(self.drivers_df.select(*keys, "team_name"), on=keys, how="left")
            .join(self.session_df.select("session_key", "circuit_short_name", "year"), on="session_key", how="left")
            .join(neutralised, on="session_key", how="left")
            .join(suspended, on="session_key", how="left")
            .drop("first_timed_start", "last_lap_start")
            .sort("session_key", "position")
        )
        print(f"{results.height} driver results built.")
        return results

    def write_output(self, final_df, name):
        (DATA_DIR / "output_data").mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(f"{DATA_DIR}/output_data/{name}.parquet")

    def load(self):
        print("Loading data into the database...")
        with self.engine.begin() as conn:
            for name in ("laps", "stints", "pit_stops", "results", "safety_car_periods", "race_laps"):
                df = pl.read_parquet(f"{DATA_DIR}/output_data/{name}.parquet")
                session_keys = df["session_key"].unique().to_list()
                if inspect(conn).has_table(name):
                    conn.execute(text(f"DELETE FROM {name} WHERE session_key = ANY(:keys)"), {"keys": session_keys})
                df.write_database(name, connection=conn, if_table_exists="append")
                print(f"{df.height} {name} from {len(session_keys)} sessions loaded into the database.")
            conn.execute(text("""
                CREATE OR REPLACE VIEW team_race_pace AS
                SELECT session_key, circuit_short_name, year, team_name,
                       AVG(lap_duration) AS avg_lap, COUNT(*) AS n_laps
                FROM laps
                GROUP BY session_key, circuit_short_name, year, team_name
            """))
        print("View team_race_pace refreshed.")
