FROM python:3.12-slim

WORKDIR /app

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY controller.py main.py ./
COPY etl/ etl/
COPY ml/ ml/
RUN mkdir -p etl/data/raw/sessions etl/data/raw/drivers etl/data/raw/laps etl/data/raw/stints \
    etl/data/raw/pits etl/data/raw/weather etl/data/raw/race_control

ENTRYPOINT ["sh", "-c", "python etl/db_setup.py && exec python main.py \"$@\"", "--"]
CMD ["--merge", "1"]
