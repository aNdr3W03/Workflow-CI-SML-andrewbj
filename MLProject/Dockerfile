FROM python:3.13.4-slim

WORKDIR /app

COPY MLProject/ /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update -y && apt-get install -y gcc && apt-get install -y git

ENTRYPOINT [ "mlflow", "run", "." ]