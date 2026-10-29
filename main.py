import click

from etl.controller import ETLController
from ml.analytics import save_results
from ml.baseline import MLbaseline
from ml.controller import MLController
from ml.explain import MLExplain
from ml.export_web import MLExportWeb
from ml.features import MLfeatures
from ml.models import MLModels
from ml.pit_loss import MLPitLoss
from ml.safety_car import N_SIMS, MLSafetyCar
from ml.simulate import MLSimulate
from ml.traffic import MLTraffic
from ml.tuning import MLTuning
from ml.undercut import MLUndercut
from ml.validate import MLValidate


def parse_strategy(ctx, param, value):
    """'MEDIUM:18,HARD:39' -> [("MEDIUM", 18), ("HARD", 39)]"""
    if value is None:
        return None
    try:
        return [(compound.strip().upper(), int(length)) for compound, length in
                (stint.split(":") for stint in value.split(","))]
    except ValueError:
        raise click.BadParameter("use COMPOUND:LAPS separated by commas, e.g. MEDIUM:18,HARD:39")


def parse_compounds(ctx, param, value):
    """'MEDIUM,HARD' -> ("MEDIUM", "HARD")"""
    compounds = tuple(compound.strip().upper() for compound in value.split(","))
    if len(compounds) != 2:
        raise click.BadParameter("use OLD,NEW, e.g. MEDIUM,HARD")
    return compounds


@click.command()
@click.option("--download", type=click.BOOL, default=False, help="Download raw data from the OpenF1 API into etl/data/raw/.")
@click.option("--process", type=click.BOOL, default=False, help="Parse raw data and write the lap, stint, pit stop and result tables to etl/data/output_data/.")
@click.option("--merge", type=click.BOOL, default=False, help="Read the parquet and merge it into Postgres.")
@click.option("--plot", type=click.BOOL, default=False, help="Save the plots to plots/ and their results as a new run in the analytics_* tables.")
@click.option("--features", type=click.BOOL, default=False, help="Build the model features (X, y) from the laps table, save the column order to ml/columns/ and split train/test by season.")
@click.option("--baseline", type=click.BOOL, default=False, help="Evaluate the baseline models (global mean, mean per compound) on the test season.")
@click.option("--models", type=click.BOOL, default=False, help="Train Random Forest and XGBoost, compare them with the baselines and save the evaluation plots to plots/.")
@click.option("--tune", type=click.BOOL, default=False, help="Tune Random Forest and XGBoost with race-grouped CV, plot feature importance and degradation curves, and save the models to ml/trained/.")
@click.option("--pitstops", type=click.BOOL, default=False, help="Compare the two pit loss estimates per circuit and save plots/pit_loss.png.")
@click.option("--simulate", type=click.BOOL, default=False, help="Search every 1- and 2-stop strategy for a circuit and team (or predict one --strategy) with the tuned monotone XGBoost from ml/trained/.")
@click.option("--validate", type=click.BOOL, default=False, help="Simulate the real strategies of the top finishers in the test season, compare with their race times and with the optimal strategy, and save the tables to results/.")
@click.option("--shap", type=click.BOOL, default=False, help="Explain the monotone XGBoost with SHAP values on the test season and save plots/shap_*.png.")
@click.option("--safety-car", type=click.BOOL, default=False, help="Run the best strategies for a circuit and team through random Safety Car and VSC races (Monte Carlo) and save plots/safety_car_mc.png.")
@click.option("--undercut", type=click.BOOL, default=False, help="Seconds gained by pitting before the car in front, for every pit lap and wait, and save plots/undercut.png.")
@click.option("--traffic", type=click.BOOL, default=False, help="Rank the best strategies again racing the field of the latest race at the circuit, with dirty air and overtaking, and save plots/traffic.png.")
@click.option("--web", type=click.BOOL, default=False, help="Export the model comparisons and simulation results to web/data.js for the web page (web/index.html).")
@click.option("--circuit", help="Circuit short name as in the laps table, e.g. Spielberg (with --simulate, --safety-car, --undercut, --traffic).")
@click.option("--team", help="Team name as in the laps table, e.g. McLaren (with --simulate, --safety-car, --undercut, --traffic).")
@click.option("--strategy", callback=parse_strategy, help="Predict only this strategy, as COMPOUND:LAPS, e.g. MEDIUM:18,HARD:39 (with --simulate).")
@click.option("--max-stops", default=2, show_default=True, help="Most pit stops in the strategy search (with --simulate).")
@click.option("--total-laps", type=int, default=None, help="Race distance in laps [default: latest race at the circuit] (with --simulate).")
@click.option("--track-temp", type=float, default=None, help="Track temperature in °C [default: median of the latest race at the circuit] (with --simulate).")
@click.option("--pit-loss", type=float, default=None, help="Seconds lost per pit stop [default: the circuit's from --pitstops] (with --simulate).")
@click.option("--session-key", type=int, default=None, help="Race to take the base pace from, for an absolute race time (with --simulate --strategy and --driver).")
@click.option("--driver", type=int, default=None, help="Driver number to take the base pace from (with --simulate --strategy and --session-key).")
@click.option("--n-sims", default=N_SIMS, show_default=True, help="Random races per strategy (with --safety-car).")
@click.option("--compounds", default="MEDIUM,HARD", show_default=True, callback=parse_compounds, help="Compound before and after the stop of both cars (with --undercut).")
@click.option("--n-iter", default=20, show_default=True, help="Hyperparameter combinations tried per model when tuning.")
@click.option("--test-year", default=2026, show_default=True, help="Season used as the test set; earlier seasons are the training set.")
@click.option("--label", default=None, help="Label for the analytics run, e.g. 'before 1-stop strategy'.")
def main(download, process, merge, plot, features, baseline, models, tune, pitstops, simulate, validate, shap, safety_car,
         undercut, traffic, web, circuit, team, strategy, max_stops, total_laps, track_temp, pit_loss, session_key, driver,
         n_sims, compounds, n_iter, test_year, label):
    race_steps = {"--simulate": simulate, "--safety-car": safety_car, "--undercut": undercut, "--traffic": traffic}
    if not (download or process or merge or plot or features or baseline or models or tune or pitstops or validate
            or shap or web or any(race_steps.values())):
        raise click.UsageError("Choose at least one step: --download 1, --process 1, --merge 1, --plot 1, --features 1, "
                               "--baseline 1, --models 1, --tune 1, --pitstops 1, --simulate 1, --validate 1, --shap 1, "
                               "--safety-car 1, --undercut 1, --traffic 1 or --web 1.")
    if any(race_steps.values()):
        # click has no "required only when another flag is on", so check it here before any step runs
        missing = [name for name, value in [("--circuit", circuit), ("--team", team)] if value is None]
        if missing:
            steps = " / ".join(f"{name} 1" for name, on in race_steps.items() if on)
            raise click.UsageError(f"{steps} also needs {', '.join(missing)}.")
        if (session_key is None) != (driver is None):
            raise click.UsageError("--session-key and --driver go together.")

    etl = ETLController()
    if download:
        etl.extract()
    if process:
        etl.transform()
    if merge:
        etl.load()
    if download and process and merge:
        click.echo("ETL process completed successfully.")

    if plot:
        eda = MLController()
        laps_df = eda.fetch_data("SELECT * FROM laps")
        stints_df = eda.fetch_data("SELECT * FROM stints")
        eda.save_plot(laps_df, stints_df)
        save_results(eda.engine, laps_df, stints_df, label)

    if features:
        x_train, x_test, y_train, y_test = MLfeatures().run(test_year)
        click.echo(f"Features built: train {x_train.shape}, test {x_test.shape}.")

    if baseline:
        MLbaseline().run(test_year)

    if models:
        MLModels().run(test_year)

    if tune:
        MLTuning().run(test_year, n_iter)

    if pitstops:
        MLPitLoss().run()

    if simulate:
        try:
            MLSimulate().run(circuit, team, test_year, strategy, total_laps, track_temp, pit_loss, max_stops,
                             session_key, driver)
        except ValueError as e:
            raise click.UsageError(str(e))

    if validate:
        MLValidate().run(test_year)

    if shap:
        MLExplain().run(test_year)

    try:
        if safety_car:
            MLSafetyCar().run(circuit, team, test_year, total_laps, track_temp, pit_loss, max_stops, n_sims)
        if undercut:
            MLUndercut().run(circuit, team, test_year, *compounds, total_laps, track_temp)
        if traffic:
            MLTraffic().run(circuit, team, test_year, max_stops)
    except ValueError as e:
        raise click.UsageError(str(e))

    if web:
        MLExportWeb().run(test_year)


if __name__ == "__main__":
    main()
