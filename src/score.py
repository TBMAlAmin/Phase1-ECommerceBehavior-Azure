import os
import json
import joblib
import numpy as np
import pandas as pd


model = None


def init():
    global model
    # look for model.pkl inside the model directory
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    print(f"Model dir: {model_dir}")
    print(f"Files in model dir: {os.listdir(model_dir)}")
    
    # search for model.pkl recursively
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file == "model.pkl":
                model_path = os.path.join(root, file)
                print(f"Loading model from: {model_path}")
                model = joblib.load(model_path)
                print("Model loaded successfully")
                return
    
    raise FileNotFoundError("model.pkl not found in AZUREML_MODEL_DIR")


def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data["data"])

        # encode first_event_type same as training
        event_type_map = {"view": 0, "cart": 1, "purchase": 2}
        df["first_event_type"] = df["first_event_type"].map(event_type_map).fillna(0)

        feature_cols = [
            "first_event_type",
            "first_hour",
            "first_dayofweek",
            "first_price",
            "brand_missing",
            "category_missing"
        ]

        X = df[feature_cols].fillna(0)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        return json.dumps({
            "predictions": preds.tolist(),
            "probabilities": probs.tolist()
        })

    except Exception as e:
        return json.dumps({"error": str(e)})