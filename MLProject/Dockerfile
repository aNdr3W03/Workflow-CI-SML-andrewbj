# Change from python:3.13.4-slum to miniconda3 base image
FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda env create -f conda.yaml && \
    conda clean -a -y

SHELL [ "conda", "run", "-n", "diabet-pred-env", "/bin/bash", "-c" ]

EXPOSE 5005

CMD [ "conda", "run", "-n", "diabet-pred-env", "python", "modelling.py", "-t" ]