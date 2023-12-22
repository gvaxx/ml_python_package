import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data():
    url = "path"
    data = pd.read_csv(url, delimiter=";")
    return data


def preprocess_data(data):
    X = data.drop("quality", axis=1)
    y = data["quality"]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")

    return model


def save_model(model, filename="wine_quality_model.pkl"):
    # Save the trained model to a file
    joblib.dump(model, filename)


def main():
    data = load_data()
    X, y, scaler = preprocess_data(data)
    model = train_model(X, y)
    save_model(model)
    save_model(scaler, filename="scaler.pkl")


if __name__ == "__main__":
    main()
