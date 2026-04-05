import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

INPUT_FILE = "phase2/data/session_dataset_safe.csv"
MODEL_DIR = "phase2/models"

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["user_session", "label"])
y = df["label"]

categorical_features = ["first_event_type"]
numeric_features = ["first_hour", "first_dayofweek", "first_price", "brand_missing", "category_missing"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        )
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
    ))
])

print("Training model...")
model.fit(X_train, y_train)

# create versioned filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"{MODEL_DIR}/rf_model_{timestamp}.joblib"

print(f"Saving model to {model_path}...")
joblib.dump(model, model_path)

# also save "latest"
joblib.dump(model, f"{MODEL_DIR}/rf_model_latest.joblib")

print("Done!")
