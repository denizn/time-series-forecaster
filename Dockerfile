FROM python:3.12.2-slim-bookworm

WORKDIR /app

RUN mkdir -p /reports/figures
RUN mkdir -p /models/components
RUN mkdir -p /models/results
RUN mkdir -p /models/saved_models

COPY requirements.txt requirements.txt

COPY data data

COPY src src

COPY run.py run.py

RUN pip3 install -r requirements.txt

RUN python3 run.py