from src.data.make_dataset import make_dataset
from src.features.build_features import run_pipeline
from src.models.train_model import mass_forecaster


DATA_FOLDER = './data'

# Parameter grid to be used for optimization of models
# These parameters will be tested for multiple sections of the time series with refitting
# The specific cross validation technique (backfitting) will respect temporal nature of time series
# i.e. evaluation dates are later than training sets in the cross validation splits

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive','multiplicative']
}

if __name__ == '__main__':

    # Prepare dataset
    make_dataset(DATA_FOLDER)

    # Run Column Transformer and OneHotEncoder
    run_pipeline(DATA_FOLDER)

    # Run backtesting grid search cross validation across the parameter grid above
    mass_forecaster(param_grid, DATA_FOLDER)