import logging
import os
from datetime import datetime

import dvc.api
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data_from_dvc(path):
    logging.info("Loading inference data from DVC...")
    with dvc.api.open(path=path, mode="r") as fd:
        df = pd.read_csv(fd, sep=";")
    return df


def preprocess_data(data, scaler, target_column):
    X = data.drop(target_column, axis=1)
    X_scaled = scaler.transform(X)
    return X_scaled


def load_latest_model_and_scaler(
    model_dir,
    model_name,
    scaler_name,
):
    logging.info("Loading latest model and scaler...")
    model = joblib.load(f"{model_dir}/{model_name}")
    scaler = joblib.load(f"{model_dir}/{scaler_name}")
    return model, scaler


def make_predictions(model, X):
    return np.rint(model.predict(X))


def save_predictions(df, cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_dir = cfg.model.inference_dir
    if not os.path.exists(inf_dir):
        os.makedirs(inf_dir)
    path = os.path.join(inf_dir, f"predictions_{timestamp}.csv")
    df.to_csv(path, index=False)


def main(cfg: DictConfig):
    df = load_data_from_dvc(cfg.data.inference_path)
    model, scaler = load_latest_model_and_scaler(
        cfg.model.dir, cfg.model.file_name, cfg.scaler.file_name
    )
    X = preprocess_data(df, scaler, cfg.data.target_column)
    predictions = make_predictions(model, X)
    df["predictions"] = predictions
    save_predictions(df, cfg)
    logging.info("Inference completed.")


if __name__ == "__main__":
    main()
