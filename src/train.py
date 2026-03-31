import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os
import pickle


def train():
    # ── 1. Load Data ──────────────────────────────
    df = pd.read_csv('data/iris.csv')

    X = df.drop(columns=['target', 'target_name'])
    y = df['target']

    # ── 2. Split Data ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # ── 3. Scale Data ─────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 4. Define Parameters ──────────────────────
    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "random_state": 42
    }

    # ── 5. Start MLflow Run ───────────────────────
    mlflow.set_experiment("iris-classification")

    with mlflow.start_run():

        # train model
        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)

        # predictions
        y_pred = model.predict(X_test_scaled)

        # metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }

        # ── 6. Log to MLflow ──────────────────────
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # print results
        print("✅ Training Complete!")
        print(f"\nParameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print(f"\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # ── 7. Save Model Locally ─────────────────
        os.makedirs('models', exist_ok=True)

        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)

        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        print(f"\n✅ Model saved to models/")


if __name__ == "__main__":
    train()