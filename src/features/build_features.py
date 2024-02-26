from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn import set_config

set_config(transform_output="pandas")

## Feature pipeline

def run_pipeline(data_folder='../../data'):

    numeric_features = []
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_features = ["Promo","SchoolHoliday"]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop='if_binary', sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(steps=[
                        ('preprocessor',preprocessor)
                        ]
    )

    df_train = pd.read_parquet(data_folder+'/interim/df_train.parquet')
    df_test = pd.read_parquet(data_folder+'/interim/df_test.parquet')

    df_train_X = pipeline.fit_transform(X=df_train.drop('y',axis=1))
    df_train_y = df_train[['y']]
    df_test_X = pipeline.transform(X=df_test)

    df_train_X = pd.DataFrame(df_train_X, columns=pipeline.get_feature_names_out())
    df_test_X = pd.DataFrame(df_test_X, columns=pipeline.get_feature_names_out())

    df_train = pd.concat([df_train_X,df_train_y],axis=1)

    df_train.to_parquet(data_folder+'/processed/df_train.parquet')
    df_test_X.to_parquet(data_folder+'/processed/df_test_X.parquet')
    
    print(f'Feature datasets have been saved!')

    return df_train, df_test_X

if __name__ == '__main__':
    run_pipeline()