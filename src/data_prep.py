"""
data_prep.py
-------------
Merges tmdb_5000_movies.csv + tmdb_5000_credits.csv into one clean dataset,
parses JSON-string columns (genres, keywords, cast, crew, production
companies) into usable Python objects, fixes the known budget/revenue
zero-value issue, and saves the result to data/processed/movies_clean.csv
and data/processed/movies_clean.pkl (pickle keeps list/dict columns intact,
CSV is for quick viewing).

Run from project root:
    python src/data_prep.py
"""

import os
import ast
import pandas as pd
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def safe_literal_eval(x):
    """Parse a JSON-like string column into a real Python list/dict.
    Returns an empty list if parsing fails or value is missing."""
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []


def extract_names(list_of_dicts, key="name", limit=None):
    """Pull out the 'name' field from a list of dicts like
    [{"id": 28, "name": "Action"}, ...] -> ["Action", ...]"""
    if not isinstance(list_of_dicts, list):
        return []
    names = [d.get(key) for d in list_of_dicts if isinstance(d, dict) and key in d]
    return names[:limit] if limit else names


def extract_director(crew_list):
    """Find the director's name from the crew list."""
    if not isinstance(crew_list, list):
        return None
    for member in crew_list:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name")
    return None


def load_and_merge():
    movies = pd.read_csv(os.path.join(RAW_DIR, "tmdb_5000_movies.csv"))
    credits = pd.read_csv(os.path.join(RAW_DIR, "tmdb_5000_credits.csv"))

    # credits has its own 'title' column too - drop it to avoid a duplicate
    # after merge, and rename movie_id -> id to match movies' key
    credits = credits.rename(columns={"movie_id": "id"}).drop(columns=["title"])

    merged = movies.merge(credits, on="id", how="inner")
    print(f"Merged shape: {merged.shape}")
    return merged


def clean(df):
    df = df.copy()

    # --- Parse JSON-string columns into real Python objects ---
    df["genres_parsed"] = df["genres"].apply(safe_literal_eval)
    df["keywords_parsed"] = df["keywords"].apply(safe_literal_eval)
    df["production_companies_parsed"] = df["production_companies"].apply(safe_literal_eval)
    df["cast_parsed"] = df["cast"].apply(safe_literal_eval)
    df["crew_parsed"] = df["crew"].apply(safe_literal_eval)

    # --- Flatten into simple, usable columns ---
    df["genre_names"] = df["genres_parsed"].apply(lambda x: extract_names(x))
    df["keyword_names"] = df["keywords_parsed"].apply(lambda x: extract_names(x))
    df["top_cast"] = df["cast_parsed"].apply(lambda x: extract_names(x, limit=5))
    df["director"] = df["crew_parsed"].apply(extract_director)
    df["production_company_names"] = df["production_companies_parsed"].apply(
        lambda x: extract_names(x)
    )

    # --- Fix the known zero-value issue: 0 budget/revenue = missing, not real ---
    MIN_PLAUSIBLE_DOLLAR_VALUE = 1000
    df["budget"] = df["budget"].where(df["budget"] >= MIN_PLAUSIBLE_DOLLAR_VALUE, np.nan)
    df["revenue"] = df["revenue"].where(df["revenue"] >= MIN_PLAUSIBLE_DOLLAR_VALUE, np.nan)

    # --- Derived columns useful for later modules ---
    df["profit"] = df["revenue"] - df["budget"]
    df["roi"] = df["profit"] / df["budget"]  # return on investment
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    # --- Drop rows with no title or no release year (unusable) ---
    df = df.dropna(subset=["title", "release_year"])

    return df


def main():
    merged = load_and_merge()
    cleaned = clean(merged)

    keep_cols = [
        "id", "title", "release_date", "release_year", "runtime",
        "budget", "revenue", "profit", "roi",
        "popularity", "vote_average", "vote_count",
        "genre_names", "keyword_names", "top_cast", "director",
        "production_company_names", "original_language",
    ]
    final = cleaned[keep_cols]

    csv_path = os.path.join(PROCESSED_DIR, "movies_clean.csv")
    pkl_path = os.path.join(PROCESSED_DIR, "movies_clean.pkl")

    final.to_csv(csv_path, index=False)
    final.to_pickle(pkl_path)

    print(f"\nFinal cleaned shape: {final.shape}")
    print(f"Saved CSV  -> {csv_path}")
    print(f"Saved PKL  -> {pkl_path}  (use this one in later scripts - it")
    print("              keeps genre_names/top_cast as real Python lists,")
    print("              not stringified text like the CSV will.)")

    print("\n--- Missing data after cleaning ---")
    print(final.isnull().sum())

    print("\n--- Sample ---")
    print(final[["title", "budget", "revenue", "genre_names", "director"]].head())


if __name__ == "__main__":
    main()