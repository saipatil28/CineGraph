"""
export_results.py
--------------------
Runs (or reuses) every analysis module and exports the interesting
results into a single JSON file the frontend dashboard reads from.
This keeps the frontend "dumb" (just renders data) and avoids
duplicating any analysis logic - it imports and calls the real
functions from each module directly.

Run from project root, AFTER running data_prep.py and regression.py
at least once (their outputs are required inputs here):
    python src/export_results.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import regression
import fuzzy_logic
import association_rules
import graph_analysis
import anomaly_detection

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FRONTEND_DATA_DIR = os.path.join(BASE_DIR, "frontend", "data")
os.makedirs(FRONTEND_DATA_DIR, exist_ok=True)


def to_native(obj):
    """Recursively convert numpy/pandas types to plain Python types so
    json.dump doesn't choke on int64/float64/Timestamp/etc."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    if pd.isna(obj) if not isinstance(obj, (list, dict)) else False:
        return None
    return obj


def export_regression():
    df = regression.load_data()
    X_h, y_h, _ = regression.prepare_features(df, mode="hindsight")
    X_p, y_p, _ = regression.prepare_features(df, mode="pre_release")

    results_h, _, _ = regression.train_and_evaluate(X_h, y_h)
    results_p, _, _ = regression.train_and_evaluate(X_p, y_p)

    rf_h = results_h["Random Forest"]["model"]
    rf_p = results_p["Random Forest"]["model"]

    top_features_hindsight = (
        pd.Series(rf_h.feature_importances_, index=X_h.columns)
        .sort_values(ascending=False).head(8)
    )
    top_features_pre = (
        pd.Series(rf_p.feature_importances_, index=X_p.columns)
        .sort_values(ascending=False).head(8)
    )

    return {
        "hindsight": {
            "r2": results_h["Random Forest"]["r2"],
            "mae": results_h["Random Forest"]["mae"],
            "top_features": [{"name": k, "importance": v} for k, v in top_features_hindsight.items()],
        },
        "pre_release": {
            "r2": results_p["Random Forest"]["r2"],
            "mae": results_p["Random Forest"]["mae"],
            "top_features": [{"name": k, "importance": v} for k, v in top_features_pre.items()],
        },
    }


def export_fuzzy():
    df = fuzzy_logic.load_data()
    df_valid = df.dropna(subset=["budget"])

    budget_min = df_valid["budget"].quantile(0.01)
    budget_max = df_valid["budget"].quantile(0.99)

    genre_scores = fuzzy_logic.compute_genre_popularity_scores(df)
    system, bounds = fuzzy_logic.build_fuzzy_system(budget_min, budget_max)

    examples = [
        {"label": "Big-budget Adventure, many studios", "budget": budget_max * 0.8, "genre": "Adventure", "studios": 8},
        {"label": "Low-budget Horror, few studios", "budget": budget_min + (budget_max - budget_min) * 0.05, "genre": "Horror", "studios": 1},
        {"label": "Mid-budget Comedy, some studios", "budget": budget_min + (budget_max - budget_min) * 0.4, "genre": "Comedy", "studios": 3},
    ]
    example_results = []
    for ex in examples:
        genre_pop_val = genre_scores.get(ex["genre"], 50)
        score, category = fuzzy_logic.predict_success(system, ex["budget"], genre_pop_val, ex["studios"], bounds)
        example_results.append({**ex, "score": round(score, 1), "category": category})

    genre_ranking = sorted(genre_scores.items(), key=lambda x: -x[1])

    return {
        "budget_min": budget_min,
        "budget_max": budget_max,
        "genre_popularity": [{"genre": g, "score": round(s, 1)} for g, s in genre_ranking],
        "examples": example_results,
    }


def export_association_rules():
    df = association_rules.load_data()
    df = association_rules.add_success_label(df)
    baskets = association_rules.build_baskets(df)

    genre_rules = association_rules.mine_rules(baskets, min_support=association_rules.MIN_SUPPORT)
    person_rules_raw = association_rules.mine_rules(baskets, min_support=association_rules.PERSON_MIN_SUPPORT)

    def rules_to_list(rules_df, top_n=10):
        if rules_df is None or rules_df.empty:
            return []
        out = []
        for _, r in rules_df.sort_values("lift", ascending=False).head(top_n).iterrows():
            out.append({
                "antecedents": sorted(str(a) for a in r["antecedents"]),
                "consequents": sorted(str(c) for c in r["consequents"]),
                "support": round(r["support"], 3),
                "confidence": round(r["confidence"], 3),
                "lift": round(r["lift"], 2),
            })
        return out

    # show_success_rules/show_person_rules print AND return - suppress
    # their prints since this script only cares about the returned data
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        success_rules_df = association_rules.show_success_rules(genre_rules) if genre_rules is not None and not genre_rules.empty else None
        person_rules_df = association_rules.show_person_rules(person_rules_raw) if person_rules_raw is not None and not person_rules_raw.empty else None

    return {
        "genre_rules": rules_to_list(success_rules_df),
        "person_rules": rules_to_list(person_rules_df),
    }


def export_graph():
    df = graph_analysis.load_data()
    films = graph_analysis.get_people_per_film(df)
    frequent_people = graph_analysis.filter_frequent_people(films, graph_analysis.MIN_APPEARANCES)
    G = graph_analysis.build_graph(films, frequent_people)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        degree, weighted_degree, betweenness = graph_analysis.show_centrality(G)
        communities = graph_analysis.show_communities(G)

    top_betweenness = sorted(betweenness.items(), key=lambda x: -x[1])[:12]
    top_weighted = sorted(weighted_degree.items(), key=lambda x: -x[1])[:12]

    community_list = []
    for c in sorted(communities, key=len, reverse=True)[:6]:
        members = sorted(c)
        community_list.append({
            "size": len(members),
            "sample_members": members[:10],
        })

    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "top_bridges": [{"person": p.replace("Actor: ", "").replace("Director: ", ""), "score": round(s, 4)} for p, s in top_betweenness],
        "top_collaborators": [{"person": p.replace("Actor: ", "").replace("Director: ", ""), "collaborations": s} for p, s in top_weighted],
        "communities": community_list,
    }


def export_anomalies():
    df, model, feature_columns = anomaly_detection.load_data()

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sleeper_hits, flops = anomaly_detection.find_sleeper_hits_and_flops(df, model, feature_columns, top_n=8)
        outliers = anomaly_detection.find_general_outliers(df, top_n=8)

    def movie_row(row):
        return {
            "title": row["title"],
            "year": int(row["release_year"]),
            "budget": row["budget"],
            "revenue": row["revenue"],
            "predicted_revenue": row.get("predicted_revenue"),
            "genres": row["genre_names"],
        }

    return {
        "sleeper_hits": [movie_row(r) for _, r in sleeper_hits.iterrows()],
        "flops": [movie_row(r) for _, r in flops.iterrows()],
        "outliers": [
            {
                "title": r["title"], "year": int(r["release_year"]),
                "budget": r["budget"], "revenue": r["revenue"],
                "vote_average": r["vote_average"],
            }
            for _, r in outliers.iterrows()
        ],
    }


def export_overview(df):
    return {
        "total_movies": len(df),
        "date_range": [int(df["release_year"].min()), int(df["release_year"].max())],
        "total_revenue_tracked": df["revenue"].sum(skipna=True),
    }


def main():
    print("Exporting regression results...")
    regression_data = export_regression()

    print("Exporting fuzzy logic results...")
    fuzzy_data = export_fuzzy()

    print("Exporting association rules...")
    assoc_data = export_association_rules()

    print("Exporting graph analysis...")
    graph_data = export_graph()

    print("Exporting anomaly detection...")
    anomaly_data = export_anomalies()

    df = pd.read_pickle(os.path.join(PROCESSED_DIR, "movies_clean.pkl"))
    overview_data = export_overview(df)

    output = {
        "overview": overview_data,
        "regression": regression_data,
        "fuzzy": fuzzy_data,
        "association_rules": assoc_data,
        "graph": graph_data,
        "anomalies": anomaly_data,
    }

    output = to_native(output)

    out_path = os.path.join(FRONTEND_DATA_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported all results -> {out_path}")


if __name__ == "__main__":
    main()
