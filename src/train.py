import argparse
import os
import time
import joblib
import mlflow
import azureml.mlflow  # noqa: F401
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--C", type=float, default=3.73)
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(".parquet")]
        if not files:
            raise FileNotFoundError(f"No parquet file found in {path}")
        path = os.path.join(path, files[0])
    return pd.read_parquet(path)


def prepare(df: pd.DataFrame):
    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)

    exclude = {"asin", "reviewerID", "overall", "label"}
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    # reduce memory: skip TF-IDF for first successful run
    feature_cols = [c for c in numeric_cols if not c.startswith("tfidf_")]
    if not feature_cols:
        raise RuntimeError("No usable feature columns found")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = df["label"]
    return X, y


def log_metrics(name, y_true, preds, probs):
    mlflow.log_metric(f"{name}_accuracy", accuracy_score(y_true, preds))
    mlflow.log_metric(f"{name}_auc", roc_auc_score(y_true, probs))
    mlflow.log_metric(f"{name}_precision", precision_score(y_true, preds, zero_division=0))
    mlflow.log_metric(f"{name}_recall", recall_score(y_true, preds, zero_division=0))
    mlflow.log_metric(f"{name}_f1", f1_score(y_true, preds, zero_division=0))


def main():
    args = parse_args()
    start = time.time()
    mlflow.start_run()

    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)

    X_train, y_train = prepare(train_df)
    X_val, y_val = prepare(val_df)
    X_test, y_test = prepare(test_df)

    model = LogisticRegression(C=args.C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", args.C)
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("feature_selection", "all_numeric_except_tfidf")

    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        log_metrics(name, y, preds, probs)

    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    mlflow.log_metric("training_runtime_seconds", time.time() - start)

    mlflow.end_run()


if __name__ == "__main__":
    main()