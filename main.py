import click

from etl.controller import ETLController
from ml.analytics import save_results
from ml.controller import MLController


@click.command()
@click.option("--download", type=click.BOOL, default=False, help="Download raw data from the OpenF1 API into etl/data/raw/.")
@click.option("--process", type=click.BOOL, default=False, help="Parse raw data and write the lap and stint tables to etl/data/output_data/.")
@click.option("--merge", type=click.BOOL, default=False, help="Read the parquet and merge it into Postgres.")
@click.option("--plot", type=click.BOOL, default=False, help="Save the plots to plots/ and their results as a new run in the analytics_* tables.")
@click.option("--label", default=None, help="Label for the analytics run, e.g. 'before 1-stop strategy'.")
def main(download, process, merge, plot, label):
    if not (download or process or merge or plot):
        raise click.UsageError("Choose at least one step: --download 1, --process 1, --merge 1 or --plot 1.")

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


if __name__ == "__main__":
    main()
