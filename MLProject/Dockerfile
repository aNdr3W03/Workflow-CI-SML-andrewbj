FROM python:3.13.4-slim

RUN apt-get update -y && apt-get install -y gcc && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN conda env create -f conda.yaml && \
    conda clean -a -y && \
    pip install --no-cache-dir mlflow

SHELL [ "conda", "run", "-n", "diabet-pred-env", "/bin/bash", "-c" ]

EXPOSE 5005

CMD [ "conda", "run", "-n", "diabet-pred-env", "python", "modelling.py" ]