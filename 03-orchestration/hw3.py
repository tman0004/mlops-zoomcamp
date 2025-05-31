import mlflow
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import pickle
from sklearn.linear_model import LinearRegression
from prefect import flow, task

@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def preprocessing(train_df):
    categorical = ['PULocationID', 'DOLocationID']
    train_dict = train_df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dict)

    target = 'duration'
    y_train = train_df[target].values

    return X_train, y_train

@task
def train_model(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

@flow
def ml_pipeline():
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('hw3-experiment')
    
    with mlflow.start_run(): 
        train_df = read_dataframe('data/yellow_tripdata_2023-03.parquet')
        X_train, y_train = preprocessing(train_df)
        model = train_model(X_train, y_train)
        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

if __name__ == "__main__":
    ml_pipeline()