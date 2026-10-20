FROM python:3.12-slim

WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements-etl.txt

COPY controller.py main.py ./
COPY etl/ etl/
COPY data/output_data/ data/output_data/
RUN mkdir -p data/raw/sessions data/raw/drivers data/raw/laps data/raw/stints \
    data/raw/pits data/raw/weather data/raw/race_control

ENTRYPOINT ["sh", "-c", "python etl/db_setup.py && exec python main.py \"$@\"", "--"]
CMD ["--merge", "1"]
