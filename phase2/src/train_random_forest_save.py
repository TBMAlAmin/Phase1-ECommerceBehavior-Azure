import argparse
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="phase2/data/session_dataset_safe.csv",
        help="Path to leakage-safe session dataset CSV",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="phase2/models",
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="phase2/metrics",
        help="Directory to save evaluation metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    print(f"Loading dataset from: {args.input_file}")
    df = pd.read_csv(args.input_file)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, digits=4),
        "input_file": args.input_file,
        "random_state": 42,
        "model_type": "RandomForestClassifier",
        "feature_columns": [
            "first_event_type",
            "first_hour",
            "first_dayofweek",
            "first_price",
            "brand_missing",
            "category_missing",
        ],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.model_dir, f"rf_model_{timestamp}.joblib")
    latest_model_path = os.path.join(args.model_dir, "rf_model_latest.joblib")
    metrics_json_path = os.path.join(args.metrics_dir, "random_forest_metrics.json")
    metrics_txt_path = os.path.join(args.metrics_dir, "random_forest_metrics.txt")

    print(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)
    joblib.dump(model, latest_model_path)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("Random Forest Evaluation\n")
        f.write("========================\n")
        f.write(f"ROC-AUC: {metrics['test_roc_auc']:.4f}\n")
        f.write(f"Precision: {metrics['test_precision']:.4f}\n")
        f.write(f"Recall: {metrics['test_recall']:.4f}\n")
        f.write(f"F1: {metrics['test_f1']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics["confusion_matrix"]) + "\n\n")
        f.write("Classification Report:\n")
        f.write(metrics["classification_report"] + "\n")

    print("Done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
