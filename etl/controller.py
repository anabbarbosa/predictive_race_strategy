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

    def extract(self):
        self.fetch_session_data()
        self.session_list()
        self.fetch_drivers_data()
        self.fetch_laps_data()
        self.fetch_stints_data()
        self.fetch_pits_data()
        self.fetch_weather_data()
        self.fetch_race_control_data()

    def fetch_session_data(self):
        print("Fetching sessions data from 2026...")
        url = f"https://api.openf1.org/v1/sessions?year=2026&csv=true"
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"{DATA_DIR}/raw/sessions/session_2026.csv", "wb") as f:
                f.write(response.content)
            print(f"Data for sessions from 2026 saved successfully.")
        else:
            print(f"Failed to fetch data for sessions from 2026. Status code: {response.status_code}")

    def session_list(self):
        print("Filtering sessions for races that were not cancelled...")
        df = pl.read_csv(f"{DATA_DIR}/raw/sessions/session_2026.csv")
        df = df.filter((pl.col('session_name') == 'Race') & ~pl.col('is_cancelled'))
        self.session_keys = df['session_key'].to_list()
        return self.session_keys

    def fetch_drivers_data(self):
        print("Fetching drivers data...")
        for session_keys in self.session_keys:
            url = f"https://api.openf1.org/v1/drivers?session_key={session_keys}&csv=true"
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"{DATA_DIR}/raw/drivers/drivers_{session_keys}.csv", "wb") as f:
                    f.write(response.content)
                print(f"Data for drivers on session {session_keys} saved successfully.")
            else:
                print(f"Failed to fetch data for session {session_keys}. Status code: {response.status_code}")  
            time.sleep(1)

    def fetch_laps_data(self):
        print("Fetching laps data...")
        for session_key in self.session_keys:
            url = f"https://api.openf1.org/v1/laps?session_key={session_key}&csv=true"
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"{DATA_DIR}/raw/laps/laps_{session_key}.csv", "wb") as f:
                    f.write(response.content)
                print(f"Data for laps on session {session_key} saved successfully.")
            else:
                print(f"Failed to fetch data for session {session_key}. Status code: {response.status_code}")
            time.sleep(1)

    def fetch_stints_data(self):
        print("Fetching stints data...")
        for session_key in self.session_keys:
            url = f"https://api.openf1.org/v1/stints?session_key={session_key}&csv=true"
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"{DATA_DIR}/raw/stints/stints_{session_key}.csv", "wb") as f:
                    f.write(response.content)
                print(f"Data for stints on session {session_key} saved successfully.")
            else:
                print(f"Failed to fetch data for session {session_key}. Status code: {response.status_code}")
            time.sleep(1)

    def fetch_pits_data(self):
        print("Fetching pits data...")
        for session_key in self.session_keys:
            url = f"https://api.openf1.org/v1/pit?session_key={session_key}&csv=true"
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"{DATA_DIR}/raw/pits/pits_{session_key}.csv", "wb") as f:
                    f.write(response.content)
                print(f"Data for pits on session {session_key} saved successfully.")
            else:
                print(f"Failed to fetch data for session {session_key}. Status code: {response.status_code}")
            time.sleep(1)

    def fetch_weather_data(self):
        print("Fetching weather data...")
        for session_key in self.session_keys:
            url = f"https://api.openf1.org/v1/weather?session_key={session_key}&csv=true"
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"{DATA_DIR}/raw/weather/weather_{session_key}.csv", "wb") as f:
                    f.write(response.content)
                print(f"Data for weather on session {session_key} saved successfully.")
            else:
                print(f"Failed to fetch data for session {session_key}. Status code: {response.status_code}")
            time.sleep(1)

    def fetch_race_control_data(self):
        print("Fetching race control data...")
        for session_key in self.session_keys:
            url = f"https://api.openf1.org/v1/race_control?session_key={session_key}&csv=true"
            response = requests.get(url)
            if response.status_code == 200:
                with open(f"{DATA_DIR}/raw/race_control/race_control_{session_key}.csv", "wb") as f:
                    f.write(response.content)
                print(f"Data for race control on session {session_key} saved successfully.")
            else:
                print(f"Failed to fetch data for session {session_key}. Status code: {response.status_code}")
            time.sleep(1)
        
        print("Data extraction completed.")

    def transform(self):
        self.read_raw_data()
        mid_df = self.build_lap_table()
        mid_df = self.add_tyre_life(mid_df)
        mid_df = self.add_lap_flags(mid_df)
        mid_df = self.add_safety_car_flags(mid_df)
        final_df = self.clean_laps(mid_df)
        self.write_output(final_df, "laps")
        self.write_output(self.build_stint_table(), "stints")

    def read_raw_data(self):
        print("Reading raw data from CSV files...")
        self.laps_df = pl.read_csv(f"{DATA_DIR}/raw/laps/laps_[0-9]*.csv").with_columns(
            pl.col("date_start").str.to_datetime(time_zone="UTC")
        )
        self.stints_df = pl.read_csv(f"{DATA_DIR}/raw/stints/stints_[0-9]*.csv")
        self.session_df = pl.read_csv(f"{DATA_DIR}/raw/sessions/session_[0-9]*.csv")
        self.drivers_df = pl.read_csv(f"{DATA_DIR}/raw/drivers/drivers_[0-9]*.csv")
        self.race_controls_df = pl.read_csv(f"{DATA_DIR}/raw/race_control/race_control_[0-9]*.csv")
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

    def add_safety_car_flags(self, mid_df):
        sc_laps = (
            self.race_controls_df.filter(pl.col("category") == "SafetyCar")
            .sort("session_key", "date")
            .with_columns(
                sc_type=pl.when(pl.col("message").str.starts_with("VSC"))
                    .then(pl.lit("VSC"))
                    .otherwise(pl.lit("SC")),
                end_lap=pl.col("lap_number").shift(-1).over("session_key"),
            )
            .filter(pl.col("message").str.ends_with("DEPLOYED"))
            .with_columns(pl.col("end_lap").fill_null(999))  # corrida terminou sob SC
            .select(
                "session_key", "sc_type",
                lap_number=pl.int_ranges("lap_number", pl.col("end_lap") + 1),
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

    def write_output(self, final_df, name):
        (DATA_DIR / "output_data").mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(f"{DATA_DIR}/output_data/{name}.parquet")

    def load(self):
        print("Loading data into the database...")
        with self.engine.begin() as conn:
            for name in ("laps", "stints"):
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
