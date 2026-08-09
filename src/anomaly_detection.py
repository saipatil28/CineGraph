"""
anomaly_detection.py
-----------------------
Two complementary approaches to finding unusual movies:

1. SLEEPER HITS / FLOPS - uses the pre-release revenue model saved by
   regression.py to predict expected revenue, then flags movies whose
   actual revenue was WAY higher (sleeper hit) or WAY lower (flop) than
   predicted. This directly follows up on the fuzzy logic module's note
   that "predicted vs actual gaps are candidate sleeper hits."

2. GENERAL OUTLIERS - uses Isolation Forest (a genuinely different,
   unsupervised anomaly detection technique) over budget/revenue/
   runtime/popularity/vote features to find movies that are just
   statistically unusual overall, independent of any prediction.

Uses movies_clean.pkl and revenue_model.pkl (both produced by earlier
scripts - run data_prep.py and regression.py first).

Run from project root:
    python src/anomaly_detection.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import IsolationForest

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_data():
    df = pd.read_pickle(os.path.join(PROCESSED_DIR, "movies_clean.pkl"))

    model_path = os.path.join(PROCESSED_DIR, "revenue_model.pkl")
    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    return df, model_bundle["model"], model_bundle["feature_columns"]


def align_features(df, feature_columns):
    """Rebuild the same one-hot genre feature matrix used at training
    time, then reindex to match the saved model's exact column order -
    any genre seen at training but missing here gets filled with 0, and
    any new genre not seen at training is safely dropped."""

    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(df["genre_names"]),
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index,
    )

    df["num_production_companies"] = df["production_company_names"].apply(len)
    base_cols = ["budget", "runtime", "release_year", "num_production_companies"]

    X = pd.concat([df[base_cols], genre_dummies], axis=1)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def find_sleeper_hits_and_flops(df, model, feature_columns, top_n=10):
    df_valid = df.dropna(subset=["budget", "revenue", "runtime"]).copy()

    X = align_features(df_valid, feature_columns)
    df_valid["predicted_revenue"] = model.predict(X)

    # Use a ratio in log-space rather than raw dollar difference - a
    # $50M miss means something very different for a $10M-budget indie
    # vs a $200M blockbuster, and log-ratio treats both fairly.
    df_valid["log_ratio"] = np.log1p(df_valid["revenue"]) - np.log1p(df_valid["predicted_revenue"])

    sleeper_hits = df_valid.sort_values("log_ratio", ascending=False).head(top_n)
    flops = df_valid.sort_values("log_ratio", ascending=True).head(top_n)

    print(f"\n--- Top {top_n} SLEEPER HITS (actual revenue far exceeded prediction) ---")
    for _, row in sleeper_hits.iterrows():
        print(
            f"\n  {row['title']} ({int(row['release_year'])})"
            f"\n    Predicted: ${row['predicted_revenue']:,.0f}  |  Actual: ${row['revenue']:,.0f}"
            f"\n    Budget: ${row['budget']:,.0f}  |  Genres: {', '.join(row['genre_names'])}"
        )

    print(f"\n--- Top {top_n} FLOPS (actual revenue far below prediction) ---")
    for _, row in flops.iterrows():
        print(
            f"\n  {row['title']} ({int(row['release_year'])})"
            f"\n    Predicted: ${row['predicted_revenue']:,.0f}  |  Actual: ${row['revenue']:,.0f}"
            f"\n    Budget: ${row['budget']:,.0f}  |  Genres: {', '.join(row['genre_names'])}"
        )

    return sleeper_hits, flops


def find_general_outliers(df, top_n=10, contamination=0.03):
    """Isolation Forest: an unsupervised method that isolates outliers
    by how few random splits it takes to separate them from the rest of
    the data - unusual points get isolated faster, giving them a more
    negative anomaly score. Unlike the sleeper-hit analysis, this makes
    no prediction and needs no target - it just flags what's
    statistically unusual across several features at once."""

    df_valid = df.dropna(subset=["budget", "revenue", "runtime", "popularity", "vote_count"]).copy()

    features = ["budget", "revenue", "runtime", "popularity", "vote_count", "vote_average"]
    X = df_valid[features]

    iso = IsolationForest(contamination=contamination, random_state=42)
    df_valid["anomaly_score"] = iso.fit_predict(X)       # -1 = outlier, 1 = normal
    df_valid["anomaly_degree"] = iso.decision_function(X)  # lower = more unusual

    outliers = df_valid[df_valid["anomaly_score"] == -1].sort_values("anomaly_degree")

    print(f"\n--- General statistical outliers (Isolation Forest, {len(outliers)} found) ---")
    print(f"Top {top_n} most unusual movies overall (budget/revenue/ratings combined):\n")

    for _, row in outliers.head(top_n).iterrows():
        print(
            f"  {row['title']} ({int(row['release_year'])})"
            f"\n    Budget: ${row['budget']:,.0f} | Revenue: ${row['revenue']:,.0f} | "
            f"Runtime: {row['runtime']:.0f}min | Vote avg: {row['vote_average']:.1f} "
            f"({row['vote_count']:.0f} votes)\n"
        )

    return outliers


def main():
    df, model, feature_columns = load_data()
    print(f"Loaded {len(df)} movies and pre-trained revenue model "
          f"({len(feature_columns)} features)")

    find_sleeper_hits_and_flops(df, model, feature_columns)
    find_general_outliers(df)


if __name__ == "__main__":
    main()
