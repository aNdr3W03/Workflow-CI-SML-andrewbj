FROM python:3.13.4-slim

RUN apt-get update -y && apt-get install -y gcc && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY MLProject/ .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "mlflow", "run", "." ]