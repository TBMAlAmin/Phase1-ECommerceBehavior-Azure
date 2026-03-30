import argparse
import json
import os
import pandas as pd
import requests
from sklearn.metrics import accuracy_score


def resolve_parquet_path(path: str) -> str:
    if os.path.isdir(path):
        parquet_files = [f for f in os.listdir(path) if f.endswith(".parquet")]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet file found in directory: {path}")
        return os.path.join(path, parquet_files[0])
    return path


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing from deployment dataset.")
    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--scoring_uri", type=str, required=True)
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=5)
    args = parser.parse_args()

    data_path = resolve_parquet_path(args.data)
    df = pd.read_parquet(data_path)
    df = create_labels(df)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.key}"
    }

    all_preds = []
    all_true = []

    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size].copy()
        y_true = batch["label"].tolist()

        payload = {"data": batch.to_dict(orient="records")}

        response = requests.post(
            args.scoring_uri,
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )

        print(f"Batch {start}:{start + len(batch)} status code:", response.status_code)

        if response.status_code != 200:
            print("Response body:", response.text)
            raise RuntimeError(f"Endpoint invocation failed on batch starting at row {start}")

        result = response.json()
        preds = result["predictions"]

        all_preds.extend(preds)
        all_true.extend(y_true)

    acc = accuracy_score(all_true, all_preds)

    print("Total predictions:", len(all_preds))
    print("Sample predictions:", all_preds[:10])
    print("Deployment accuracy:", acc)


if __name__ == "__main__":
    main()
