import joblib
import pandas as pd


def load_model(filename="wine_quality_model.pkl"):
    # Load the trained model from a file
    model = joblib.load(filename)
    return model


def load_new_data():
    url = "some/path"
    data = pd.read_csv(url, delimiter=";")
    return data


def preprocess_data(data):
    # Basic preprocessing for inference data
    # Assuming the target column 'quality' is not in the inference dataset
    X = data
    return X


def make_predictions(model, X):
    # Make predictions using the loaded model
    predictions = model.predict(X)
    return predictions


def save_predictions(predictions, filename="predictions.csv"):
    # Save the predictions to a CSV file
    pd.DataFrame(predictions, columns=["Predicted_Quality"]).to_csv(
        filename, index=False
    )


def main():
    model = load_model()
    new_data = load_new_data()
    X = preprocess_data(new_data)
    predictions = make_predictions(model, X)
    save_predictions(predictions)


if __name__ == "__main__":
    main()
