import click

from etl.controller import ETLController


@click.command()
@click.option("--download", type=click.BOOL, default=False, help="Download raw data from the OpenF1 API into data/raw/.")
@click.option("--process", type=click.BOOL, default=False, help="Parse raw data and write the lap table to data/output_data/.")
@click.option("--merge", type=click.BOOL, default=False, help="Read the parquet and merge it into Postgres.")
def main(download, process, merge):
    if not (download or process or merge):
        raise click.UsageError("Choose at least one step: --download 1, --process 1 or --merge 1.")

    etl = ETLController()
    if download:
        etl.extract()
    if process:
        etl.transform()
    if merge:
        etl.load()


if __name__ == "__main__":
    main()
