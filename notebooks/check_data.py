import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

movies = pd.read_csv(os.path.join(RAW_DIR, "tmdb_5000_movies.csv"))
credits = pd.read_csv(os.path.join(RAW_DIR, "tmdb_5000_credits.csv"))

print("movies shape:", movies.shape)
print("credits shape:", credits.shape)

print("\n--- movies columns ---")
print(movies.columns.tolist())

print("\n--- credits columns ---")
print(credits.columns.tolist())

print("\n--- sample rows ---")
print(movies[["title", "budget", "revenue", "genres"]].head())

print("\n--- budget zero-value check (known data quality issue) ---")
print("Rows with budget == 0:", (movies["budget"] == 0).sum())
print("Rows with revenue == 0:", (movies["revenue"] == 0).sum())

print("\n--- budget stats ---")
print(movies["budget"].describe())