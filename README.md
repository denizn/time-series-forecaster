# time-series-forecaster
A library to showcase time series analysis and forecasting

This project and repository aims to showcase a sample time series analysis and forecasting project.
From a dataset point of view, we will be using Rossmann Retail store dataset from below competition https://www.kaggle.com/competitions/rossmann-store-sales/data.

There are three main parts of the code base:
1. data/make_dataset:
2. feature/run_pipeline
3. models/mass_forecaster

#### Part 1: Make dataset

In this part we load the datasets train.csv and test.csv and make some basic transformations necessary for the prophet module

#### Part 2: Run pipeline

Here, on top of seasonality and trend, we would like to create features based on whether a store has been in a promotion on the specific date
and whether it has been affected by a School Holiday on that specific day.
For this, we use a Column Transformer, One Hot Encoder and a pipeline to handle these transformations. Because the expected results are categorical and known (binary), we are able to do the transformations at once and did not need to handle this as part of the modelling pipeline. Therefore we only use fit_transform methodology within the pipeline.

#### Part 3: Mass Forecaster

Mass Forecaster tries to find the best combination of parameters for the facebook prophet algorithm, and tries to optimize these by applying these onto multiple subsets of the data via backtesting. This is performed for a forecasting horizon of 42 days, which is what the Rossmann needed for the competition.

    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive','multiplicative']

![image](https://github.com/denizn/time-series-forecaster/assets/35758436/5d10f29f-f2b7-404f-8647-e1dc2ecca6f5)
(Animation from: https://skforecast.org/0.11.0/user_guides/backtesting)

# Sample history and 42-day forecast can be seen below:
![newplot](https://github.com/denizn/time-series-forecaster/assets/35758436/8d4b592a-a12e-473a-8fd4-2146b0a555bb)
(Plotly animated html files are available under: reports/figure)

# Backtesting cross validation results
Cross validation results can be under model/results folder in below form:
<img width="500" alt="Screenshot 2024-02-26 at 7 40 12 pm" src="https://github.com/denizn/time-series-forecaster/assets/35758436/05f9781a-f974-48f4-8dac-55fe9351ade6">

