import polars as pl
import streamlit as st

from ml.safety_car import MLSafetyCar
from ml.simulate import WINDOW_SECONDS, MLSimulate
from ml.traffic import MLTraffic
from ml.undercut import DIVERGING, MLUndercut

YEAR = 2026


@st.cache_resource
def simulator() -> MLSimulate:
    return MLSimulate()


@st.cache_data
def options() -> tuple[list[str], list[str]]:
    sim = simulator()
    circuits = sim.fetch_data("SELECT DISTINCT circuit_short_name FROM results ORDER BY 1")["circuit_short_name"].to_list()
    teams = sorted(col.removeprefix("team_name_") for col in sim.columns if col.startswith("team_name_"))
    return circuits, teams


@st.cache_data
def context(circuit: str) -> dict:
    return simulator().race_context(circuit)


@st.cache_data
def best_strategies(circuit: str, team: str, total_laps: int, track_temperature: float, pit_loss: float,
                    max_stops: int) -> pl.DataFrame:
    return MLSimulate().run(circuit, team, YEAR, total_laps=total_laps, track_temperature=track_temperature,
                            pit_loss=pit_loss, max_stops=max_stops, plot=False)


@st.cache_data
def safety_car(circuit: str, team: str, total_laps: int, track_temperature: float, pit_loss: float, max_stops: int,
               n_sims: int) -> pl.DataFrame:
    return MLSafetyCar().run(circuit, team, YEAR, total_laps, track_temperature, pit_loss, max_stops, n_sims, plot=False)


@st.cache_data
def undercut(circuit: str, team: str, old: str, new: str, total_laps: int, track_temperature: float) -> pl.DataFrame:
    return MLUndercut().run(circuit, team, YEAR, old, new, total_laps, track_temperature, plot=False)


@st.cache_data
def traffic(circuit: str, team: str, max_stops: int) -> pl.DataFrame:
    return MLTraffic().run(circuit, team, YEAR, max_stops, plot=False)


st.set_page_config(page_title="Race strategy", layout="wide")
st.title("Predictive race strategy")

circuits, teams = options()
with st.sidebar:
    circuit = st.selectbox("Circuit", circuits, index=circuits.index("Shanghai") if "Shanghai" in circuits else 0)
    team = st.selectbox("Team", teams, index=teams.index("Mercedes") if "Mercedes" in teams else 0)
    race = context(circuit)
    total_laps = st.number_input("Race laps", 20, 90, int(race["total_laps"]))
    track_temperature = st.slider("Track temperature (°C)", 15.0, 60.0, float(round(race["track_temperature"], 1)), 0.5)
    pit_loss = st.slider("Pit loss (s)", 15.0, 35.0, float(round(race["pit_loss"], 1)), 0.1)
    max_stops = st.radio("Most pit stops", [1, 2], index=1, horizontal=True)
    st.caption(f"Defaults from the latest race at {circuit} and its pit-loss estimate. Model: tuned monotone XGBoost.")

best_tab, sc_tab, undercut_tab, traffic_tab = st.tabs(["Best strategy", "Safety Car", "Undercut", "Traffic"])

with best_tab:
    with st.spinner("Simulating every 1- and 2-stop strategy..."):
        results = best_strategies(circuit, team, total_laps, track_temperature, pit_loss, max_stops)
    best = results.row(0, named=True)
    best_by_stops = results.group_by("n_stops").agg(pl.col("gap_to_best").min()).sort("n_stops")
    cols = st.columns(1 + best_by_stops.height)
    cols[0].metric("Best strategy", f"{best['compounds']}, pit lap {', '.join(map(str, best['pit_laps']))}")
    for col, (stops, gap) in zip(cols[1:], best_by_stops.iter_rows()):
        col.metric(f"Best {stops}-stop", f"+{gap:.2f} s")

    pair = results.filter(pl.col("n_stops") == 1)["compounds"][0] if (results["n_stops"] == 1).any() else None
    if pair:
        curve = (
            results.filter(pl.col("compounds") == pair)
            .select(pit_lap=pl.col("pit_laps").list.first(), seconds_vs_best=pl.col("total_time") - pl.col("total_time").min())
            .sort("pit_lap")
        )
        window = curve.filter(pl.col("seconds_vs_best") <= WINDOW_SECONDS)["pit_lap"]
        st.subheader(f"1-stop pit window, {pair}: laps {window.min()}–{window.max()}")
        st.line_chart(curve, x="pit_lap", y="seconds_vs_best", x_label="Pit lap", y_label="Race time vs best pit lap (s)")
    st.subheader("Top 20 strategies")
    st.dataframe(results.head(20).drop("strategy_id"), hide_index=True)

with sc_tab:
    st.write("Each strategy runs through the same random races with Safety Cars and VSCs drawn from the circuit's "
             "history. A stop under a neutralisation is cheaper, and the car pits early when one comes.")
    n_sims = st.select_slider("Random races", [200, 500, 1000, 2000], value=1000)
    if st.button("Run Monte Carlo"):
        with st.spinner("Simulating random races..."):
            summary = safety_car(circuit, team, total_laps, track_temperature, pit_loss, max_stops, n_sims)
        st.dataframe(summary.drop("strategy_id"), hide_index=True)
        st.caption("expected_gap_s: average race time vs the best average. mean/p90_regret_s: time lost against the "
                   "best strategy of each random race. win_pct: share of races where it is the best.")

with undercut_tab:
    compounds = ["SOFT", "MEDIUM", "HARD"]
    left, right = st.columns(2)
    old = left.selectbox("Both cars start on", compounds, index=1)
    new = right.selectbox("Both cars switch to", compounds, index=2)
    table = undercut(circuit, team, old, new, total_laps, track_temperature)
    pivot = table.pivot("laps_later", index="pit_lap", values="gain_s").sort("pit_lap").to_pandas().set_index("pit_lap")
    limit = max(abs(table["gain_s"].min()), abs(table["gain_s"].max()), 0.01)
    st.write("Seconds car A gains by pitting before car B, by A's pit lap (rows) and the laps B stays out (columns). "
             "The undercut works when the gain is bigger than the gap between them before the stop.")
    st.dataframe(pivot.style.background_gradient(cmap=DIVERGING, vmin=-limit, vmax=limit, axis=None).format("{:.2f}"))

with traffic_tab:
    st.write(f"Our {team} car races the field of the latest race at {circuit}, every rival on its real strategy. "
             "Close behind a car it loses time in dirty air and only passes if it is clearly faster.")
    if st.button("Run traffic simulation"):
        with st.spinner("Racing the field..."):
            try:
                ranked = traffic(circuit, team, max_stops)
            except ValueError as e:
                st.error(str(e))
            else:
                st.dataframe(ranked, hide_index=True)
