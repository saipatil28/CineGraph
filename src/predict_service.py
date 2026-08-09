"""
predict_service.py
---------------------
A thin CLI wrapper around fuzzy_logic.py's prediction logic, meant to be
called as a subprocess from the Node backend. Takes budget/genre/studios
as command-line args, prints a single JSON line to stdout, and exits.

Usage:
    python src/predict_service.py <budget> <genre> <num_studios>
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

import fuzzy_logic


def main():
    if len(sys.argv) != 4:
        print(json.dumps({"error": "usage: predict_service.py <budget> <genre> <num_studios>"}))
        sys.exit(1)

    try:
        budget = float(sys.argv[1])
        genre = sys.argv[2]
        num_studios = float(sys.argv[3])
    except ValueError:
        print(json.dumps({"error": "budget and num_studios must be numeric"}))
        sys.exit(1)

    try:
        df = fuzzy_logic.load_data()
        df_valid = df.dropna(subset=["budget"])

        budget_min = df_valid["budget"].quantile(0.01)
        budget_max = df_valid["budget"].quantile(0.99)

        genre_scores = fuzzy_logic.compute_genre_popularity_scores(df)
        genre_pop_value = genre_scores.get(genre, 50)

        system, bounds = fuzzy_logic.build_fuzzy_system(budget_min, budget_max)
        score, category = fuzzy_logic.predict_success(system, budget, genre_pop_value, num_studios, bounds)

        available_genres = sorted(genre_scores.keys())

        print(json.dumps({
            "score": round(float(score), 1),
            "category": category,
            "genre_popularity": round(float(genre_pop_value), 1),
            "genre_recognized": genre in genre_scores,
            "budget_range": {"min": float(budget_min), "max": float(budget_max)},
            "available_genres": available_genres,
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()