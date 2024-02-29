from pathlib import Path
import pickle
import argparse
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--store_number", type=int)
args = parser.parse_args()

MODEL_PATH = Path("./models")
DATA_PATH = Path("./data")


def get_prediction(store_number, only_forecast=True):
    """
    This function gets predictions for the requested store number
    :param store_number: store number to predict, this should exist in test dataset
    :only_forecast: binary variable, to determine whether extra columns should be returned. Defaults to True
    :returns: pd.DataFrame
    """

    df_test_large = pd.read_parquet(DATA_PATH / "processed" / "df_test_X.parquet")

    if store_number not in (df_test_large.index.levels[0]):
        raise Exception("Store number not found in test dataset")

    df_test = df_test_large.loc[store_number].reset_index()

    with open(MODEL_PATH / "saved_models" / f"{store_number}.pkl", "rb") as handle:
        model = pickle.load(handle)
        prediction = model.predict(df_test)

    fig = plot_components_plotly(model, model.predict(df_test))
    fig.write_html(MODEL_PATH / "components" / f"{store_number}.html")

    return prediction.set_index("ds")["yhat"] if only_forecast else prediction


if __name__ == "__main__":
    yhat = get_prediction(store_number=args.store_number if args.store_number else 1)
    print(yhat)
