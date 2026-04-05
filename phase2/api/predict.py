import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("phase2/models/rf_model_latest.joblib")

model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "first_event_type",
    "first_hour",
    "first_dayofweek",
    "first_price",
    "brand_missing",
    "category_missing",
]

def predict_session(data: dict) -> dict:
    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

    proba = float(model.predict_proba(df)[0][1])
    pred = int(model.predict(df)[0])

    return {
        "prediction": pred,
        "purchase_probability": round(proba, 6),
        "input_features": data,
    }
