# time-series-forecaster
A library to showcase time series analysis and forecasting

This project and repository aims to showcase a sample time series analysis and forecasting project.
From a dataset point of view, we will be using Rossmann Retail store dataset from below competition https://www.kaggle.com/competitions/rossmann-store-sales/data.

#### Environment details
Library has been built using poetry package manager on python 3.12.
Poetry.lock and pyproject.toml files include the necessary details on the dev environment.
Poetry can be installed using pipx, and it installs the necessary requirements to run the project.

#### Installation steps
On macOS:
brew install pipx -> pipx ensurepath -> pipx install poetry -> poetry install

On Linux:
Ubuntu 23.04 or above
sudo apt update -> sudo apt install pipx -> pipx ensurepath -> pipx install poetry -> poetry install

Ubuntu 22.04 or below
python3 -m pip install --user pipx -> python3 -m pipx ensurepath -> pipx install poetry -> poetry install

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

Mass Forecaster attempts to find the best combination of parameters for the facebook prophet algorithm, and the optimal model for each store by applying backtesting forecasts onto multiple subsets of the data. This is performed for all Rossmann stores in the test dataset at a forecasting horizon of 42 days, which is what the Rossmann needs for their business and for this competition.

The parameter grid (or the search space for the models) are as per below:

    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive','multiplicative']

#### How to:
A simple python3 run.py command triggers the entire pipeline where forecasts for all stores get generated.
- python3 run.py

After the pipeline is run, a single prediction can be obtained using the below command, or results can be viewed under models/results folder.
- python3 ./src/models/predict_model.py --store_number 3

#### Configurations:
Configuration of the run.py can be set in the conf/config.yaml file.
Paths for data, models and reports can be set here.
Since there are 856 stores currently, to limit runtime I've also added a max store count parameter. Sample config file can be seen below.

<img width="578" alt="Screenshot 2024-03-05 at 4 20 38 am" src="https://github.com/denizn/time-series-forecaster/assets/35758436/111b4d23-ea56-42a6-bf81-10a4db409115">

The methodology used to determine the most optimal forecast is "backtesting cross-validation with refit", and below representation
in my opinion does a great job of visualizing the overall process.

![image](https://github.com/denizn/time-series-forecaster/assets/35758436/5d10f29f-f2b7-404f-8647-e1dc2ecca6f5)
(Animation from: https://skforecast.org/0.11.0/user_guides/backtesting)

# Sample history and 42-day forecast can be seen below:

![newplot](https://github.com/denizn/time-series-forecaster/assets/35758436/8d4b592a-a12e-473a-8fd4-2146b0a555bb)
(Plotly animated html files are available under: models/graphs)

# Components (trend, yearly and weekly seasonality)

In terms of explainability, it's possible to separate components of the forecast such as trend, weekly and yearly seasonality, which is a great benefit of the prophet algorithm. See figure below:
<img width="837" alt="Screenshot 2024-02-27 at 3 31 27 am" src="https://github.com/denizn/time-series-forecaster/assets/35758436/13255060-82c4-4ec0-9d3d-256edddaec12">

# Backtesting cross validation results

Cross validation results can be seen for each store under model/results folder in below form:

<img width="500" alt="Screenshot 2024-02-26 at 7 40 12 pm" src="https://github.com/denizn/time-series-forecaster/assets/35758436/05f9781a-f974-48f4-8dac-55fe9351ade6">

