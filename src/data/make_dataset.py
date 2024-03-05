#!/usr/bin/env python

"""Module that loads raw train and test Rossmann dataset, filters columns and saves interim datasets"""

from pathlib import Path
import pandas as pd


def make_dataset(data_folder) -> tuple[pd.DataFrame, : pd.DataFrame]:
    """
    Make dataset takes in a data folder and looks for a train.csv and a test.csv file and preprocesses it
    """

    # Set data types for pandas to load csv
    dtype = {
        "Store": "Int64",
        "DayOfWeek": "Int64",
        "Date": "str",
        "Sales": "Int64",
        "Customers": "Int64",
        "Open": "Int64",
        "Promo": "Int64",
        "StateHoliday": "str",
        "SchoolHoliday": "Int64",
    }

    # Train dataset is determined by Rossmann as Sales data from Jan 2013 and July 2015
    data_train = pd.read_csv(
        data_folder / "raw" / "train.csv", dtype=dtype, parse_dates=["Date"]
    )

    # Test dataset is determined by Rossman as Sales data from August 2015 and September 2015
    data_test = pd.read_csv(
        data_folder / "raw" / "test.csv", dtype=dtype, parse_dates=["Date"]
    )

    sample_submission = pd.read_csv(data_folder / "raw" / "sample_submission.csv")
    data_store = pd.read_csv(data_folder / "raw" / "store.csv")

    # Dropping fields DayOfWeek (redundant), Customers (not needed), 'Open' filter is not needed as well since we only train and predict open days

    # Filter necessary fields

    data_train = data_train[data_train["Open"] == 1][
        ["Store", "Date", "Promo", "SchoolHoliday", "Sales"]
    ]
    data_test = data_test[data_test["Open"] == 1][
        ["Store", "Date", "Promo", "SchoolHoliday"]
    ]

    data_train.rename({"Sales": "y", "Date": "ds"}, axis=1, inplace=True)
    data_test.rename({"Sales": "y", "Date": "ds"}, axis=1, inplace=True)

    data_train.set_index(["Store", "ds"], inplace=True)
    data_test.set_index(["Store", "ds"], inplace=True)

    data_train.index.levels[1].freq = "D"
    data_test.index.levels[1].freq = "D"

    data_train.sort_index(level=[0, 1], inplace=True)
    data_test.sort_index(level=[0, 1], inplace=True)

    data_train.to_parquet(data_folder / "interim" / "df_train.parquet")
    data_test.to_parquet(data_folder / "interim" / "df_test.parquet")

    print(f"Train and test datasets saved!")

    return data_train, data_test


if __name__ == "__main__":

    DATA_FOLDER = Path(".") / "data"

    data_train, data_test = make_dataset(DATA_FOLDER)
