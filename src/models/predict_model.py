import pickle, sys, argparse
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--store_number", type=int)
args = parser.parse_args()


def get_prediction(store_number, only_forecast=True):
    """
    This function gets predictions for the requested store number
    returns: pd.DataFrame
    """

    df_test_large = pd.read_parquet("data/processed/df_test_X.parquet")

    if store_number not in (df_test_large.index.levels[0]):
        raise Exception("Store number not found in test dataset")

    df_test = df_test_large.loc[store_number].reset_index()

    with open("models/saved_models/1.pkl", "rb") as handle:
        model = pickle.load(handle)
        prediction = model.predict(df_test)

    fig = plot_components_plotly(model, model.predict(df_test))
    fig.write_html(f"models/components/{store_number}.html")

    return prediction.set_index("ds")["yhat"] if only_forecast else prediction


if __name__ == "__main__":
    yhat = get_prediction(
        store_number=args.store_number if args.store_number else 1
    ).set_index("ds")["yhat"]
    print(yhat)
