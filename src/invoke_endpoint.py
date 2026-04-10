import json
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# endpoint details
ENDPOINT_URL = "https://session-endpoint.qatarcentral.inference.ml.azure.com/score"
API_KEY = "3SwAnrwYsLJm32U1dkFwSzir15mvSEP2uCjXYdd9FrPAeNH5TtSKJQQJ99CDAAAAAAAAAAAAINFRAZMLlF7T"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def main():
    # load deploy dataset
    print("Loading deploy dataset...")
    df = pd.read_csv("data/deploy.csv")

    # prepare features same as training
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
    y_true = df["label"]

    # send to endpoint
    print("Sending request to endpoint...")
    payload = {"data": X.to_dict(orient="records")}
    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        data=json.dumps(payload)
    )

    print("Response status:", response.status_code)
    
    # handle response being a string or dict
    result = response.json()
    if isinstance(result, str):
        result = json.loads(result)

    if "error" in result:
        print("Error:", result["error"])
        return

    # evaluate predictions
    preds = result["predictions"]
    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds, zero_division=0)

    print(f"\nDeployment Accuracy: {acc:.4f}")
    print(f"Deployment F1:       {f1:.4f}")
    print(f"Total predictions:   {len(preds)}")


if __name__ == "__main__":
    main()