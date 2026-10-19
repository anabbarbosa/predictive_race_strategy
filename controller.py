import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import polars as pl

load_dotenv()

class ProjectUtils():

    def __init__(self):
        PGUSER = os.getenv("PGUSER", "postgres")
        PGPASSWORD = os.getenv("PGPASSWORD", "change-me")
        PGHOST = os.getenv("PGHOST", "localhost")
        PGPORT = os.getenv("PGPORT", "5432")
        PGDATABASE = os.getenv("PGDATABASE", "postgres")

        DATABASE_URL = f"postgresql+psycopg://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
        self.engine = create_engine(DATABASE_URL, echo=False, future=True)


class UtilsABC(ProjectUtils):
    @property
    def vendor(self):
        """
        get the vendor for which data is to be processed.

        :return: the vendor/source id for which data is to be processed.
        :rtype: str

        """
        raise NotImplementedError


    def extract(self):
        """
        handle data extraction for the project.

        :raise: `CvStatsError` if an anticipated error is caught.
        :return: None
        :rtype: None

        """
        raise NotImplementedError

    def transform(self):
        """
        handle data transformation for the project.

        :raise: `CvStatsError` if an anticipated error is caught.
        :return: None
        :rtype: None

        """
        raise NotImplementedError

    def load(self):
        """
        handle data loading for the project.

        :raise: `CvStatsError` if an anticipated error is caught.
        :return: None
        :rtype: None

        """
        raise NotImplementedError