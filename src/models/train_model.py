import os
import logging
import itertools
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled=True

data_folder = '../../data'

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive','multiplicative']
}

def time_series_cv(df_train, df_test, param_grid:str):
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
        m.add_regressor('cat__Promo_1.0')
        # m.add_regressor('cat__SchoolHoliday_1.0')
        m.fit(df_train.reset_index())  # Fit model with given params

        df_cv = cross_validation(m, initial='730 days', period='90 days', horizon = '42 days')
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses

    # Refit best model
    best_params = all_params[tuning_results['rmse'].argmin()]
    print(f'Best params are {best_params}')
    print(f'Best rmse is : {tuning_results['rmse'].min()}')
    m_best = Prophet(**best_params).fit(df_train.reset_index())
    yhat_train = m_best.predict(df_train.reset_index())
    yhat_test = m_best.predict(df_test.reset_index())

    return yhat_train, yhat_test, m_best, tuning_results

def mass_forecaster(param_grid, data_folder='../../data', store_count=2):

    df_train = pd.read_parquet(data_folder+'/processed/df_train.parquet')
    df_test = pd.read_parquet(data_folder+'/processed/df_test_X.parquet')

    stores = df_test.index.levels[0]

    print(stores[0:store_count])

    forecasts = []
    tuning_results = []

    for store in stores[0:store_count]:
        yhat_train, yhat_test, m_best, tuning_result = time_series_cv(df_train.loc[store], df_test.loc[store], param_grid)
        forecast = pd.concat([yhat_train, yhat_test],axis=0)
        
        forecast.insert(0,'store',store)
        forecasts.append(forecast)

        tuning_result.insert(0,'store',store)
        tuning_results.append(tuning_result)

        fig = plot_plotly(m_best,forecast)
        fig.write_html(file='../../reports/figures/'+str(store)+'.html')

    # Save bulks forecasts and model tuning results
    pd.concat(forecasts).to_csv('../../models/results/forecasts.csv')
    pd.concat(tuning_results).to_csv('../../models/results/tunings_results.csv')

if __name__ == '__main__':

    mass_forecaster(param_grid=param_grid)