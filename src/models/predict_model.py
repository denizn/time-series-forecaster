import pickle, sys, argparse
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly

parser = argparse.ArgumentParser()
parser.add_argument('store_number', type=int)
args = parser.parse_args()

def get_prediction(store_number=1):
    '''
    This function gets predictions for the requested store number
    returns: pd.DataFrame
    '''
    
    df_test = pd.read_parquet('../../data/processed/df_test_X.parquet').loc[store_number].reset_index()

    with open('../../models/saved_models/1.pkl','rb') as handle:
        model = pickle.load(handle)
        prediction = model.predict(df_test)

    plot_components_plotly(model, model.predict(df_test))

    return prediction

if __name__ == '__main__':
    yhat = get_prediction(store_number=args.store_number if args.store_number else None).set_index('ds')['yhat']
    print(yhat)