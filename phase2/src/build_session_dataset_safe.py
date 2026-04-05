import pandas as pd

INPUT_FILE = "phase2/data/processed_2019_oct_sample_200k.csv"
OUTPUT_FILE = "phase2/data/session_dataset_safe.csv"

print("Loading data...")
df = pd.read_csv(INPUT_FILE)

print("Parsing time...")
df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

print("Sorting...")
df = df.sort_values(["user_session", "event_time"])

print("Building leakage-safe session features...")

sessions = []

for session_id, group in df.groupby("user_session"):
    group = group.sort_values("event_time")
    events = list(group["event_type"])

    first_row = group.iloc[0]
    first_time = first_row["event_time"]

    session = {}
    session["user_session"] = session_id

    # early-session / first-event features only
    session["first_event_type"] = str(first_row["event_type"])
    session["first_hour"] = int(first_time.hour) if pd.notna(first_time) else -1
    session["first_dayofweek"] = int(first_time.dayofweek) if pd.notna(first_time) else -1
    session["first_price"] = float(first_row["price"]) if pd.notna(first_row["price"]) else -1.0
    session["brand_missing"] = int(pd.isna(first_row["brand"]) or str(first_row["brand"]).strip() == "")
    session["category_missing"] = int(pd.isna(first_row["category_code"]) or str(first_row["category_code"]).strip() == "")

    # label: strict sequential funnel view -> cart -> purchase
    label = 0
    if "view" in events and "cart" in events and "purchase" in events:
        v = events.index("view")
        c = events.index("cart")
        p = events.index("purchase")
        if v < c < p:
            label = 1

    session["label"] = label
    sessions.append(session)

session_df = pd.DataFrame(sessions)

print("Saving...")
session_df.to_csv(OUTPUT_FILE, index=False)

print("Done!")
print(session_df.head())
print("\nLabel distribution:")
print(session_df["label"].value_counts())
