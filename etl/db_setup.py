import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def create_database():
    db = os.environ["PGDATABASE"]
    # CREATE DATABASE não roda dentro de transação, por isso autocommit
    with psycopg.connect(
        host=os.environ["PGHOST"],
        port=os.environ["PGPORT"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        dbname="postgres",
        autocommit=True,
    ) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,)).fetchone()
        if exists:
            print(f"Database {db} already exists.")
        else:
            conn.execute(f'CREATE DATABASE "{db}"')
            print(f"Database {db} created.")


if __name__ == "__main__":
    create_database()
