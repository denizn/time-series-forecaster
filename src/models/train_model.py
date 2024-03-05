#!/usr/bin/env python

"""Module that splits train dataset into multiple time series folds, runs prophet algorithm
using different hyperparameters and returns forecasts using the best performing model"""

import pickle
import logging
import itertools
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly

from config.config import get_config

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled=True

def time_series_cv(df_train, df_test, param_grid, include_promo, include_holiday):
    '''
    Performs a grid search hyperparameter optimization over a matrix of parameters
    and backtesting cross-validation with refitting for a single store
    Finally, returns the model that performs the best historical rmse.
    '''

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    # Use cross validation to evaluate all parameters
    for params in all_params:

        m = Prophet(**params)

        if include_promo:
            m.add_regressor('cat__Promo_1.0')

        if include_holiday:
            m.add_regressor('cat__SchoolHoliday_1.0')

        m.fit(df_train.reset_index())  # Fit model with initial params

        df_cv = cross_validation(m, initial='730 days', period='90 days', horizon = '42 days', disable_tqdm=True)
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses

    # Refit best model
    best_params = all_params[tuning_results['rmse'].argmin()]
    print(f'Best params are {best_params}')
    print(f'Best rmse is : {tuning_results['rmse'].min():.2f}')
    m_best = Prophet(**best_params).fit(df_train.reset_index())
    yhat_train = m_best.predict(df_train.reset_index())
    yhat_test = m_best.predict(df_test.reset_index())

    return yhat_train, yhat_test, m_best, tuning_results

def mass_forecaster(conf):
    '''
    Mass forecaster will run the time_series_cv
    across every store in test dataset if no max_store_count is defined
    In cases where max_store_count is defined, it will run for the first
    n stores.
    '''

    df_train = pd.read_parquet(conf.DATA_PATH / "processed" / "df_train.parquet")
    df_test = pd.read_parquet(conf.DATA_PATH / "processed" / "df_test_X.parquet")

    stores = df_test.index.levels[0]

    forecasts = []
    tuning_results = []

    for store in stores[0:conf.MAX_STORE_COUNT] if conf.MAX_STORE_COUNT else stores:
        print(f'Starting forecasting procedure for Store:{store}')
        yhat_train, yhat_test, m_best, tuning_result = time_series_cv(df_train.loc[store], df_test.loc[store], conf.PARAM_GRID, conf.INCLUDE_PROMO, conf.INCLUDE_HOLIDAY)
        forecast = pd.concat([yhat_train, yhat_test],axis=0)

        # Save Best Model
        with (conf.MODEL_PATH / "saved_models" / f"{store}.pkl").open("wb") as handle:
            pickle.dump(m_best, handle)
        
        forecast.insert(0,'store',store)
        forecasts.append(forecast)

        tuning_result.insert(0,'store',store)
        tuning_results.append(tuning_result)

        # Save Figures
        fig = plot_plotly(m_best,forecast)
        fig.write_html(file=conf.MODEL_PATH / "graphs" / f"{str(store)}.html")

        # Save components
        fig = plot_components_plotly(m_best,forecast)
        fig.write_html(file=conf.MODEL_PATH / "components" / f"{str(store)}.html")

    # Save bulks forecasts and model tuning results
    pd.concat(forecasts).to_csv(conf.MODEL_PATH / "results" / "forecasts.csv")
    pd.concat(tuning_results).to_csv(conf.MODEL_PATH / "results" / "tuning_results.csv")

if __name__ == '__main__':

    mass_forecaster(get_config())
