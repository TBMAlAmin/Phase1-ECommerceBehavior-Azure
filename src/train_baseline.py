import argparse
import os
import time
import joblib
import mlflow
import azureml.mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data",   type=str, required=True)
    parser.add_argument("--test_data",  type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    return parser.parse_args()


def load_data(path):
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".csv"):
                return pd.read_csv(os.path.join(path, f))
    return pd.read_csv(path)


def prepare(df):
    df = df.copy()
    le = LabelEncoder()
    df["first_event_type"] = le.fit_transform(df["first_event_type"].astype(str))

    feature_cols = [
        "first_event_type",
        "first_hour",
        "first_dayofweek",
        "first_price",
        "brand_missing",
        "category_missing"
    ]

    X = df[feature_cols].fillna(0)
    y = df["label"]
    return X, y


def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    mlflow.log_metric(f"{split_name}_accuracy",  accuracy_score(y, preds))
    mlflow.log_metric(f"{split_name}_precision", precision_score(y, preds, zero_division=0))
    mlflow.log_metric(f"{split_name}_recall",    recall_score(y, preds, zero_division=0))
    mlflow.log_metric(f"{split_name}_f1",        f1_score(y, preds, zero_division=0))
    mlflow.log_metric(f"{split_name}_auc",       roc_auc_score(y, probs))

    print(f"{split_name} accuracy: {accuracy_score(y, preds):.4f}")
    print(f"{split_name} f1:       {f1_score(y, preds, zero_division=0):.4f}")


def main():
    args = parse_args()
    start_time = time.time()
    mlflow.start_run()

    print("Loading data...")
    train_df = load_data(args.train_data)
    val_df   = load_data(args.val_data)
    test_df  = load_data(args.test_data)

    print("Preparing features...")
    X_train, y_train = prepare(train_df)
    X_val,   y_val   = prepare(val_df)
    X_test,  y_test  = prepare(test_df)

    print("Training baseline Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    mlflow.log_param("model_type", "LogisticRegression_baseline")
    mlflow.log_param("max_iter",   1000)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val,   y_val,   "val")
    evaluate(model, X_test,  y_test,  "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    mlflow.log_metric("training_runtime_seconds", time.time() - start_time)
    mlflow.end_run()
    print("Done!")


if __name__ == "__main__":
    main()