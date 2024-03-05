FROM python:3.12.2-slim-bookworm

WORKDIR /

RUN mkdir -p /models/graphs
RUN mkdir -p /models/components
RUN mkdir -p /models/results
RUN mkdir -p /models/saved_models

COPY requirements.txt requirements.txt

COPY data data

COPY reports reports

COPY src src

COPY conf conf

COPY run.py run.py

RUN pip3 install -r requirements.txt

RUN python3 run.py