import joblib
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def load_data():
    df = pd.read_csv("data/network_traffic.csv")
    return df


def prepare_data(df):
    df = df.sort_values(["tower_id", "hour"])

    # Shift congestion to next hour
    df["next_congestion"] = df.groupby("tower_id")["congestion"].shift(-1)

    # Drop last hour rows (no next value)
    df = df.dropna()

    X = df[[
        "active_users",
        "bandwidth_usage",
        "latency",
        "packet_loss",
        "signal_strength"
    ]]

    y = df["next_congestion"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("ROC-AUC Score:", roc_auc_score(y_test, probabilities))


if __name__ == "__main__":
    df = load_data()

    print("Dataset shape:", df.shape)
    print("Congestion Distribution:\n", df["congestion"].value_counts())

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = train_model(X_train, y_train)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/congestion_model.pkl")
    print("Model saved successfully.")

    evaluate_model(model, X_test, y_test)