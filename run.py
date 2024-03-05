#!/usr/bin/env python

"""Module that orchestrates the three parts of the mass forecasting engine"""

from argparse import ArgumentParser
from src.data.make_dataset import make_dataset
from src.features.build_features import run_pipeline
from src.models.train_model import mass_forecaster
from src.config.config import get_config


if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Time Series Forecaster",
        description="This program creates forecasts for each store in Rossmann Store Sales dataset",
    )
    parser.add_argument(
        "--config_file", "-c", required=False, default="./conf/config.yaml"
    )
    args = parser.parse_args()

    conf = get_config(args.config_file)

    # Prepare dataset
    make_dataset(conf.DATA_PATH)

    # Run Column Transformer and OneHotEncoder
    run_pipeline(conf.DATA_PATH)

    # Run backtesting grid search cross validation across the parameter grid above
    mass_forecaster(conf)
