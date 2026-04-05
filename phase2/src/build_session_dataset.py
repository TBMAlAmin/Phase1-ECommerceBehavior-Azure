import pandas as pd

INPUT_FILE = "phase2/data/processed_2019_oct_sample_200k.csv"
OUTPUT_FILE = "phase2/data/session_dataset.csv"

print("Loading data...")
df = pd.read_csv(INPUT_FILE)

print("Parsing time...")
df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

print("Sorting...")
df = df.sort_values(["user_session", "event_time"])

print("Building session features...")

sessions = []

for session_id, group in df.groupby("user_session"):
    events = list(group["event_type"])

    session = {}

    session["user_session"] = session_id
    session["n_events"] = len(group)

    session["n_views"] = events.count("view")
    session["n_carts"] = events.count("cart")
    session["n_purchases"] = events.count("purchase")

    session["unique_products"] = group["product_id"].nunique()

    session["session_duration"] = (
        group["event_time"].max() - group["event_time"].min()
    ).total_seconds()

    # label: sequential funnel
    if "view" in events and "cart" in events and "purchase" in events:
        v = events.index("view")
        c = events.index("cart") if "cart" in events else -1
        p = events.index("purchase") if "purchase" in events else -1

        if v < c < p:
            session["label"] = 1
        else:
            session["label"] = 0
    else:
        session["label"] = 0

    sessions.append(session)

session_df = pd.DataFrame(sessions)

print("Saving...")
session_df.to_csv(OUTPUT_FILE, index=False)

print("Done!")
print(session_df.head())
print("\nLabel distribution:")
print(session_df["label"].value_counts())
