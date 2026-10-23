import click

from etl.controller import ETLController
from ml.analytics import save_results
from ml.baseline import MLbaseline
from ml.controller import MLController
from ml.features import MLfeatures
from ml.models import MLModels


@click.command()
@click.option("--download", type=click.BOOL, default=False, help="Download raw data from the OpenF1 API into etl/data/raw/.")
@click.option("--process", type=click.BOOL, default=False, help="Parse raw data and write the lap and stint tables to etl/data/output_data/.")
@click.option("--merge", type=click.BOOL, default=False, help="Read the parquet and merge it into Postgres.")
@click.option("--plot", type=click.BOOL, default=False, help="Save the plots to plots/ and their results as a new run in the analytics_* tables.")
@click.option("--features", type=click.BOOL, default=False, help="Build the model features (X, y) from the laps table, save the column order to ml/columns/ and split train/test by season.")
@click.option("--baseline", type=click.BOOL, default=False, help="Evaluate the baseline models (global mean, mean per compound) on the test season.")
@click.option("--models", type=click.BOOL, default=False, help="Train Random Forest and XGBoost, compare them with the baselines and save the evaluation plots to plots/.")
@click.option("--test-year", default=2026, show_default=True, help="Season used as the test set; earlier seasons are the training set.")
@click.option("--label", default=None, help="Label for the analytics run, e.g. 'before 1-stop strategy'.")
def main(download, process, merge, plot, features, baseline, models, test_year, label):
    if not (download or process or merge or plot or features or baseline or models):
        raise click.UsageError("Choose at least one step: --download 1, --process 1, --merge 1, --plot 1, --features 1, --baseline 1 or --models 1.")

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


if __name__ == "__main__":
    main()
