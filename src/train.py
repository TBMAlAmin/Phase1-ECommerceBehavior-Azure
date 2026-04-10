import argparse
import os
import time
import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)

    X = df.drop(columns=["user_session", "label"])
    y = df["label"]

    categorical_features = ["first_event_type"]
    numeric_features = [
        "first_hour",
        "first_dayofweek",
        "first_price",
        "brand_missing",
        "category_missing",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
    ])

    print("Training model...")
    model.fit(X, y)

    print("Evaluating on training data...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_prob)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    runtime = time.time() - start_time
    mlflow.log_metric("training_time_seconds", runtime)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    print("Done.")
    print(f"precision={precision:.4f}")
    print(f"recall={recall:.4f}")
    print(f"f1={f1:.4f}")
    print(f"roc_auc={roc_auc:.4f}")
    print(f"model_path={model_path}")


if __name__ == "__main__":
    main()
