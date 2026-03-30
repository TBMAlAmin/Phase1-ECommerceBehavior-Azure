import json
import os
import joblib
import numpy as np
import pandas as pd

model = None


def init():
    global model

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "model_output", "model.pkl")

    model = joblib.load(model_path)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    exclude = {"asin", "reviewerID", "overall", "label"}

    numeric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    preferred_small_features = {
        "review_length",
        "word_count",
        "char_count",
        "sentiment",
        "sentiment_score",
        "polarity",
        "subjectivity",
    }

    compact_cols = []
    for c in numeric_cols:
        if c.startswith("sbert_") or c in preferred_small_features:
            compact_cols.append(c)

    if not compact_cols:
        compact_cols = [c for c in numeric_cols if not c.startswith("tfidf_")]

    X = df[compact_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return X


def run(raw_data):
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        if isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)

        X = _prepare_features(df)

        predictions = model.predict(X).tolist()

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1].tolist()
        else:
            probabilities = None

        return {
            "predictions": predictions,
            "probabilities": probabilities
        }

    except Exception as e:
        return {"error": str(e)}
