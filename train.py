import logging
import os
import subprocess
from datetime import datetime

import dvc.api
import hydra
import joblib
import mlflow
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_git_commit_id():
    try:
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        return commit_id
    except subprocess.CalledProcessError:
        logging.warning("Failed to retrieve Git commit ID.")
        return "unknown"


def load_data_from_dvc(file_path):
    logging.info("Loading data...")
    with dvc.api.open(path=file_path, mode="r") as fd:
        data = pd.read_csv(fd, sep=";")
    return data


def preprocess_data(data, target_column):
    logging.info("Preprocessing data...")
    X = data.drop(target_column, axis=1)

    y = data[target_column]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, y, scaler


def train_random_forest_model(X, y, cfg):
    logging.info("Setting up RandomForest model with hyperparameter tuning...")
    commit_id = get_git_commit_id()
    conf = cfg.model
    rf = RandomForestRegressor(random_state=conf.random_state)

    log_dir = os.path.join(
        cfg.tensorboard.dir, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    writer = tf.summary.create_file_writer(log_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=conf.test_size, random_state=conf.random_state
    )
    param_distributions = {
        "max_depth": list(conf.max_depth),
        "min_samples_split": list(conf.min_samples_split),
        "min_samples_leaf": list(conf.min_samples_leaf),
        "n_estimators": list(conf.n_estimators),
    }
    model = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=conf.n_iter,
        cv=conf.cv,
        random_state=conf.random_state,
    )
    mlflow.set_tracking_uri(uri=cfg.mlflow.uri)

    # Create a new MLflow Experiment
    mlflow.set_experiment("random_forest_experiment")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.log_params(model.best_params_)
        mlflow.log_metric("best_score", model.best_score_)
        signature = mlflow.models.signature.infer_signature(
            X_train, model.predict(X_train)
        )
        mlflow.sklearn.log_model(
            model.best_estimator_,
            "random_forest_model",
            signature=signature,
            input_example=X_train,
        )
        mlflow.log_param("git_commit_id", commit_id)

    logging.info(f"Best parameters: {model.best_params_}")
    hyperparams = model.best_params_

    model = RandomForestRegressor(**hyperparams)
    step = hyperparams["n_estimators"] // 10
    print(range(1, hyperparams["n_estimators"] + 1, step))
    index = 0
    for n_estimators in range(1, hyperparams["n_estimators"] + 1, step):
        model.set_params(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        with writer.as_default():
            tf.summary.scalar("Mean Squared Error", mse, step=index)
            tf.summary.scalar("Mean Absolure Error", mae, step=index)
            tf.summary.scalar("R2 Score", r2, step=index)
            writer.flush()
        index += 1

    return model


def save_model(model, scaler, cfg):
    if not os.path.exists(cfg.model.dir):
        os.makedirs(cfg.model.dir)
    model_path = os.path.join(cfg.model.dir, cfg.model.file_name)
    scaler_path = os.path.join(cfg.model.dir, cfg.scaler.file_name)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    df = load_data_from_dvc(cfg.data.train_path)
    X, y, scaler = preprocess_data(df, cfg.data.target_column)
    model = train_random_forest_model(X, y, cfg)
    model_path, scaler_path = save_model(model, scaler, cfg)
    logging.info(f"Model and scaler saved. Paths: {model_path}, {scaler_path}")


if __name__ == "__main__":
    main()
