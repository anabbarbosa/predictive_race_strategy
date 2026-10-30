import json

import numpy as np
import polars as pl

from ml.features import COMPOUND_ORDER, MLfeatures
from ml.tuning import TRAINED_DIR

DEGRADATION_FILE = TRAINED_DIR / "degradation.json"
MIN_STINT_LAPS = 60


class DegradationModel:

    def __init__(self, params: dict):
        self.params = params
        self.columns = params["columns"]
        self.compound_names = {code: name for name, code in COMPOUND_ORDER.items()}

    def weekend_deg(self, circuit: str) -> dict[str, float]:
        weekends = [w for w in self.params["weekends"] if w["circuit_short_name"] == circuit]
        if not weekends:
            return dict(self.params["compound_deg"])
        latest = max(weekends, key=lambda w: (w["year"], w["meeting_key"]))
        return {w["compound"]: w["deg"] for w in weekends if w["meeting_key"] == latest["meeting_key"]}

    def predict_circuit(self, x: np.ndarray, circuit: str) -> np.ndarray:
        deg = self.weekend_deg(circuit)
        compound = x[:, self.columns.index("compound")].astype(int)
        names = [self.compound_names[c] for c in range(len(COMPOUND_ORDER))]
        offset = np.array([self.params["offset"][n] for n in names])[compound]
        slope = np.array([deg[n] for n in names])[compound]
        curve = np.array([self.params["curvature"][n] for n in names])[compound]
        tyre_life = x[:, self.columns.index("tyre_life")]
        with np.errstate(divide="ignore"):
            vertex = np.where(curve < 0, -slope / (2 * curve), np.inf)
        tyre_life = np.minimum(tyre_life, vertex)
        wear = slope * tyre_life + curve * tyre_life ** 2
        return offset + wear + self.params["evolution"] * x[:, self.columns.index("lap_number")]


class MLDegradation(MLfeatures):
    def race_slopes(self, laps_df: pl.DataFrame) -> tuple[dict[str, float], dict[str, float], float, pl.DataFrame]:
        race = ["session_key", "driver_number"]
        df = laps_df.with_columns(
            corrected=self.fuel_corrected(),
            rc=pl.concat_str(pl.col("session_key").cast(pl.String), pl.col("compound"), separator="_"),
        )
        slope_keys = df.group_by("rc").len().filter(pl.col("len") >= MIN_STINT_LAPS)["rc"].sort().to_list()
        sessions = df["session_key"].unique().sort().to_list()
        columns = {
            "is_MEDIUM": (pl.col("compound") == "MEDIUM").cast(pl.Float64),
            "is_HARD": (pl.col("compound") == "HARD").cast(pl.Float64),
            **{f"curve_{c}": pl.when(pl.col("compound") == c).then(pl.col("tyre_life") ** 2).otherwise(0.0)
               for c in COMPOUND_ORDER},
            **{f"deg_{k}": pl.when(pl.col("rc") == k).then(pl.col("tyre_life")).otherwise(0.0) for k in slope_keys},
            **{f"evo_{s}": pl.when(pl.col("session_key") == s).then(pl.col("lap_number")).otherwise(0.0) for s in sessions},
        }
        # stints too short to get their own slope are dropped rather than given a wrong one
        df = df.filter(pl.col("rc").is_in(slope_keys)).with_columns(**columns)
        names = [*columns, "corrected"]
        demeaned = df.select([(pl.col(c) - pl.col(c).mean().over(race)).alias(c) for c in names])
        beta, *_ = np.linalg.lstsq(demeaned.select(list(columns)).to_numpy(), demeaned["corrected"].to_numpy(), rcond=None)
        coef = dict(zip(columns, beta))
        slopes = (
            df.group_by("rc", "meeting_key", "compound").agg(n_laps=pl.len())
            .with_columns(race_deg=pl.col("rc").map_elements(lambda k: coef[f"deg_{k}"], return_dtype=pl.Float64))
        )
        offsets = {"SOFT": 0.0, "MEDIUM": float(coef["is_MEDIUM"]), "HARD": float(coef["is_HARD"])}
        curvature = {c: float(coef[f"curve_{c}"]) for c in COMPOUND_ORDER}
        evolution = float(np.median([coef[f"evo_{s}"] for s in sessions]))
        return offsets, curvature, evolution, slopes

    def calibrate(self, slopes: pl.DataFrame, practice: pl.DataFrame) -> tuple[dict[str, float], float]:
        df = slopes.join(practice, on=["meeting_key", "compound"], how="inner").with_columns(
            fp_signal=pl.col("fp_deg") - pl.col("compound_deg"))
        x = np.column_stack([(df["compound"] == c).cast(pl.Float64).to_numpy() for c in COMPOUND_ORDER]
                            + [df["fp_signal"].to_numpy()])
        w = np.sqrt(df["n_laps"].to_numpy())
        beta, *_ = np.linalg.lstsq(x * w[:, None], df["race_deg"].to_numpy() * w, rcond=None)
        return {c: float(b) for c, b in zip(COMPOUND_ORDER, beta[:3])}, float(beta[3])

    def fit(self, test_year: int = 2026) -> dict:
        laps_df = self.load_laps().join(
            self.fetch_data("SELECT DISTINCT session_key, meeting_key FROM laps"), on="session_key", how="left")
        practice = self.practice_features(self.load_practice(), test_year)
        offsets, curvature, evolution, slopes = self.race_slopes(laps_df.filter(pl.col("year") < test_year))
        alpha, beta = self.calibrate(slopes, practice)
        weekends = practice.with_columns(
            deg=(pl.col("compound").replace_strict(alpha, return_dtype=pl.Float64)
                 + beta * (pl.col("fp_deg") - pl.col("compound_deg"))).clip(lower_bound=0.0)
        )
        return {
            "columns": self.load_feature_columns(),
            "offset": offsets,
            "curvature": curvature,
            "evolution": evolution,
            "compound_deg": alpha,
            "fp_beta": beta,
            "weekends": weekends.select("meeting_key", "circuit_short_name", "year", "compound", "deg", "n_runs").to_dicts(),
        }

    def save(self, params: dict):
        TRAINED_DIR.mkdir(parents=True, exist_ok=True)
        DEGRADATION_FILE.write_text(json.dumps(params, indent=2))
        print(f"Saved {DEGRADATION_FILE}")

    def load(self) -> DegradationModel:
        return DegradationModel(json.loads(DEGRADATION_FILE.read_text()))

    def run(self, test_year: int = 2026) -> dict:
        params = self.fit(test_year)
        print(f"Compound offsets vs SOFT (s): { {k: round(v, 3) for k, v in params['offset'].items()} }")
        print(f"Curvature (s per lap²): { {k: round(v, 5) for k, v in params['curvature'].items()} }")
        print(f"Track evolution: {params['evolution']:+.4f} s per lap")
        print(f"Typical race degradation (s per lap of tyre life): { {k: round(v, 4) for k, v in params['compound_deg'].items()} }")
        print(f"Practice long runs: every 0.01 s/lap above the compound's typical FP slope adds {params['fp_beta'] * 0.01:+.4f} s/lap in the race")
        self.save(params)
        return params
