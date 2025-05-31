import mlflow
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import pickle
import xgboost as xgb
from prefect import flow, task

@task
def load_data():
    train_df = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")
    val_df = pd.read_parquet('data/yellow_tripdata_2023-02.parquet')
    return train_df, val_df

@task
def preprocessing(train_df, val_df):
    train_df['duration'] = (train_df['tpep_dropoff_datetime'] - train_df['tpep_pickup_datetime']).dt.total_seconds() / 60
    val_df['duration'] = (val_df['tpep_dropoff_datetime'] - val_df['tpep_pickup_datetime']).dt.total_seconds() / 60

    train_df = train_df.query('duration >= 1 and duration <= 60')
    val_df = val_df.query('duration >= 1 and duration <= 60')

    categorical = ['PULocationID', 'DOLocationID']
    train_df[categorical] = train_df[categorical].astype(str)
    val_df[categorical] = val_df[categorical].astype(str)

    train_dict = train_df[categorical].to_dict(orient='records')
    val_dict = val_df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)

    target = 'duration'
    y_train = train_df[target].values
    y_val = val_df[target].values

    return X_train, y_train, X_val, y_val

@task
def train_model(X_train, y_train, X_val, y_val):
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    params = {
        'learning_rate': 0.1233407742765768,
        'max_depth': 30,
        'min_child_weight': 13.092247284357976,
        'objective': 'reg:squarederror',
        'reg_alpha': 0.3399656070101837,
        'reg_lambda': 0.35806973799616537,
        'seed': 42
    }
    mlflow.log_params(params)

    booster = xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=300,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50
    )
    mlflow.xgboost.log_model(booster, artifact_path='models_mlflow')
    return booster

@task
def evaluate_model(booster, X_val, y_val):
    valid = xgb.DMatrix(X_val, label=y_val)
    y_pred = booster.predict(valid)
    rmse = root_mean_squared_error(y_val, y_pred)
    return rmse

@flow
def ml_pipeline():
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('test-experiment')
    
    with mlflow.start_run(): 
        train_df, val_df = load_data()
        X_train, y_train, X_val, y_val = preprocessing(train_df, val_df)
        model = train_model(X_train, y_train, X_val, y_val)
        rmse = evaluate_model(model, X_val, y_val)
        mlflow.log_metric("rmse", rmse)

if __name__ == "__main__":
    ml_pipeline()