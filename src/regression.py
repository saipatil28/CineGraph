"""
regression.py
--------------
Predicts a movie's revenue from budget, runtime, popularity, vote_count,
genre, and release_year using Linear Regression and Random Forest, then
compares them. Uses movies_clean.pkl from data_prep.py.

Also saves the trained pre-release Random Forest model + its feature
columns to data/processed/, so later modules (like the fuzzy logic
success advisor) can reuse it without retraining.

Run from project root:
    python src/regression.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_data():
    path = os.path.join(PROCESSED_DIR, "movies_clean.pkl")
    df = pd.read_pickle(path)
    return df


def prepare_features(df, mode="hindsight"):
    """
    mode="hindsight"    -> uses vote_count + popularity too (known only
                            AFTER release - inflates accuracy, less useful
                            for real prediction, but shows what's possible
                            with full information).
    mode="pre_release"  -> only uses features known BEFORE a movie comes
                            out: budget, runtime, genre, release_year,
                            production company count. This is the honest,
                            harder, more genuinely useful prediction task.
    """
    # Only keep rows where we actually have real budget/revenue
    # (recall: 0s were converted to NaN in data_prep.py)
    df = df.dropna(subset=["budget", "revenue", "runtime"]).copy()

    # One-hot encode genres (a movie can have multiple genres, so we
    # use MultiLabelBinarizer instead of pd.get_dummies)
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(df["genre_names"]),
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index,
    )

    if mode == "hindsight":
        feature_cols = ["budget", "runtime", "popularity", "vote_count", "release_year"]
    elif mode == "pre_release":
        # Number of production companies attached is a reasonable proxy
        # for studio backing/track record, and IS known before release.
        df["num_production_companies"] = df["production_company_names"].apply(len)
        feature_cols = ["budget", "runtime", "release_year", "num_production_companies"]
    else:
        raise ValueError("mode must be 'hindsight' or 'pre_release'")

    X = pd.concat([df[feature_cols], genre_dummies], axis=1)
    y = df["revenue"]

    return X, y, df


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {"model": model, "mae": mae, "r2": r2}

        print(f"\n--- {name} ---")
        print(f"MAE : ${mae:,.0f}")
        print(f"R^2 : {r2:.3f}")

    return results, X_test, y_test


def show_feature_importance(rf_model, X):
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    print("\n--- Top 10 features driving revenue prediction (Random Forest) ---")
    print(importances.head(10))


def run_mode(df, mode, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    X, y, df_used = prepare_features(df, mode=mode)
    print(f"Using {len(X)} rows after dropping missing budget/revenue/runtime")
    print(f"Features used: {list(X.columns[:6])}{' ...' if len(X.columns) > 6 else ''}")

    results, X_test, y_test = train_and_evaluate(X, y)

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\nBest model ({label}): {best_name} (R^2 = {results[best_name]['r2']:.3f})")

    if "Random Forest" in results:
        show_feature_importance(results["Random Forest"]["model"], X)

    return results


def main():
    df = load_data()
    print(f"Loaded {len(df)} total rows")

    hindsight_results = run_mode(
        df, mode="hindsight",
        label="MODEL 1: Hindsight features (includes vote_count, popularity)"
    )
    pre_release_results = run_mode(
        df, mode="pre_release",
        label="MODEL 2: Pre-release only (budget, runtime, genre, release_year)"
    )

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    h_r2 = hindsight_results["Random Forest"]["r2"]
    p_r2 = pre_release_results["Random Forest"]["r2"]
    print(f"Hindsight model R^2   : {h_r2:.3f}  (includes post-release signals)")
    print(f"Pre-release model R^2 : {p_r2:.3f}  (honest 'before release' prediction)")
    print(
        "\nNote: the hindsight model's higher score is expected and somewhat "
        "circular - vote_count is only known after a movie is out. The "
        "pre-release model is the more honest answer to 'can we predict a "
        "movie's success before it launches.'"
    )

    # --- Save the pre-release Random Forest model for reuse in later modules ---
    rf_model = pre_release_results["Random Forest"]["model"]
    X_pre, y_pre, _ = prepare_features(df, mode="pre_release")

    model_path = os.path.join(PROCESSED_DIR, "revenue_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": rf_model, "feature_columns": list(X_pre.columns)}, f)

    print(f"\nSaved pre-release revenue model -> {model_path}")
    print("(will be reused by the fuzzy logic success-advisor module)")


if __name__ == "__main__":
    main()
