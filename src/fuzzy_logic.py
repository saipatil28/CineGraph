"""
fuzzy_logic.py
---------------
A fuzzy-logic "success likelihood" advisor for movies. Takes budget level,
genre popularity, and studio backing as fuzzy inputs and produces a
success-likelihood score (0-100) plus a human-readable category
(Low / Medium / High / Blockbuster).

This mimics how a studio executive reasons about a project informally -
"decent budget, hot genre, strong backing... probably does well" - rather
than a rigid if/else. A movie can partially belong to more than one
category at once, and the rules blend accordingly.

Uses movies_clean.pkl from data_prep.py to calibrate realistic budget
ranges and genre popularity scores from your actual dataset.

Run from project root:
    python src/fuzzy_logic.py
"""

import os
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_data():
    path = os.path.join(PROCESSED_DIR, "movies_clean.pkl")
    return pd.read_pickle(path)


def compute_genre_popularity_scores(df):
    """
    Build a 0-100 popularity score per genre based on average revenue
    within that genre, relative to all genres. Movies can have multiple
    genres, so we explode the genre_names list first.
    """
    exploded = df.dropna(subset=["revenue"]).explode("genre_names")
    genre_avg_revenue = exploded.groupby("genre_names")["revenue"].mean()

    # scale to 0-100
    min_rev, max_rev = genre_avg_revenue.min(), genre_avg_revenue.max()
    genre_scores = ((genre_avg_revenue - min_rev) / (max_rev - min_rev)) * 100

    return genre_scores.to_dict()


def build_fuzzy_system(budget_min, budget_max):
    """Defines the fuzzy variables, membership functions, and rules."""

    # --- Antecedents (inputs) ---
    budget = ctrl.Antecedent(np.linspace(budget_min, budget_max, 200), "budget")
    genre_pop = ctrl.Antecedent(np.linspace(0, 100, 100), "genre_pop")
    studio_backing = ctrl.Antecedent(np.linspace(0, 10, 50), "studio_backing")

    # --- Consequent (output) ---
    success = ctrl.Consequent(np.linspace(0, 100, 100), "success")

    # Membership functions - triangular/trapezoidal, calibrated to
    # the actual budget range in the dataset
    budget["low"] = fuzz.trimf(budget.universe, [budget_min, budget_min, (budget_min + budget_max) / 3])
    budget["medium"] = fuzz.trimf(
        budget.universe,
        [budget_min, (budget_min + budget_max) / 2, budget_max],
    )
    budget["high"] = fuzz.trimf(budget.universe, [(budget_min + budget_max) / 1.5, budget_max, budget_max])

    genre_pop["low"] = fuzz.trimf(genre_pop.universe, [0, 0, 40])
    genre_pop["medium"] = fuzz.trimf(genre_pop.universe, [20, 50, 80])
    genre_pop["high"] = fuzz.trimf(genre_pop.universe, [60, 100, 100])

    studio_backing["few"] = fuzz.trimf(studio_backing.universe, [0, 0, 3])
    studio_backing["some"] = fuzz.trimf(studio_backing.universe, [1, 3, 6])
    studio_backing["many"] = fuzz.trimf(studio_backing.universe, [4, 10, 10])

    success["low"] = fuzz.trimf(success.universe, [0, 0, 35])
    success["medium"] = fuzz.trimf(success.universe, [20, 45, 70])
    success["high"] = fuzz.trimf(success.universe, [50, 75, 95])
    success["blockbuster"] = fuzz.trimf(success.universe, [80, 100, 100])

    # --- Rules (the human-readable "expert system" part) ---
    # Base grid covers every budget x genre_pop combination (9 rules),
    # so no input combination is ever left with zero firing rules.
    # Studio backing then refines/boosts specific high-potential cases.
    rules = [
        # --- Base grid: budget x genre_pop (guarantees full coverage) ---
        ctrl.Rule(budget["low"] & genre_pop["low"], success["low"]),
        ctrl.Rule(budget["low"] & genre_pop["medium"], success["low"]),
        ctrl.Rule(budget["low"] & genre_pop["high"], success["medium"]),
        ctrl.Rule(budget["medium"] & genre_pop["low"], success["low"]),
        ctrl.Rule(budget["medium"] & genre_pop["medium"], success["medium"]),
        ctrl.Rule(budget["medium"] & genre_pop["high"], success["high"]),
        ctrl.Rule(budget["high"] & genre_pop["low"], success["medium"]),
        ctrl.Rule(budget["high"] & genre_pop["medium"], success["high"]),
        ctrl.Rule(budget["high"] & genre_pop["high"], success["high"]),

        # --- Refinements: strong studio backing pushes toward blockbuster ---
        ctrl.Rule(budget["high"] & genre_pop["high"] & studio_backing["many"], success["blockbuster"]),
        ctrl.Rule(budget["medium"] & genre_pop["high"] & studio_backing["many"], success["blockbuster"]),
        ctrl.Rule(budget["low"] & genre_pop["high"] & studio_backing["many"], success["high"]),

        # --- Refinements: weak studio backing pulls toward lower success ---
        ctrl.Rule(budget["low"] & genre_pop["low"] & studio_backing["few"], success["low"]),
        ctrl.Rule(budget["high"] & genre_pop["low"] & studio_backing["few"], success["low"]),
    ]

    system = ctrl.ControlSystem(rules)
    return system, (budget_min, budget_max)


def categorize(score):
    if score < 30:
        return "Low"
    elif score < 55:
        return "Medium"
    elif score < 80:
        return "High"
    else:
        return "Blockbuster"


def predict_success(system, budget_value, genre_pop_value, studio_backing_value, budget_bounds):
    """budget_bounds = (min, max) used to build the fuzzy system - inputs
    outside this range are clipped, since skfuzzy can't evaluate membership
    for values outside the defined universe."""
    bmin, bmax = budget_bounds
    budget_value = np.clip(budget_value, bmin, bmax)
    genre_pop_value = np.clip(genre_pop_value, 0, 100)
    studio_backing_value = np.clip(studio_backing_value, 0, 10)

    sim = ctrl.ControlSystemSimulation(system)
    sim.input["budget"] = budget_value
    sim.input["genre_pop"] = genre_pop_value
    sim.input["studio_backing"] = studio_backing_value
    sim.compute()

    score = sim.output["success"]
    return score, categorize(score)


def main():
    df = load_data()
    df_valid = df.dropna(subset=["budget"])

    # Use 1st/99th percentile instead of raw min/max - a few entries have
    # placeholder budgets like $1, which would otherwise badly skew the
    # "low budget" membership function toward being almost meaningless.
    budget_min = df_valid["budget"].quantile(0.01)
    budget_max = df_valid["budget"].quantile(0.99)
    print(f"Budget range used (1st-99th percentile): ${budget_min:,.0f} - ${budget_max:,.0f}")
    print(f"(raw min/max was ${df_valid['budget'].min():,.0f} - ${df_valid['budget'].max():,.0f} "
          f"- excluded as likely data-entry outliers)")

    genre_scores = compute_genre_popularity_scores(df)
    print("\n--- Genre popularity scores (0-100, derived from avg revenue) ---")
    for genre, score in sorted(genre_scores.items(), key=lambda x: -x[1]):
        print(f"  {genre:20s} {score:5.1f}")

    system, (bmin, bmax) = build_fuzzy_system(budget_min, budget_max)

    # --- Demo: run a few example scenarios ---
    print("\n--- Example predictions ---")

    examples = [
        {"label": "Big-budget Adventure film, many studios", "budget": bmax * 0.8,
         "genre": "Adventure", "studios": 8},
        {"label": "Low-budget Horror film, few studios", "budget": bmin + (bmax - bmin) * 0.05,
         "genre": "Horror", "studios": 1},
        {"label": "Mid-budget Comedy, some studios", "budget": bmin + (bmax - bmin) * 0.4,
         "genre": "Comedy", "studios": 3},
    ]

    for ex in examples:
        genre_pop_val = genre_scores.get(ex["genre"], 50)  # default to 50 if genre unseen
        score, category = predict_success(system, ex["budget"], genre_pop_val, ex["studios"], (bmin, bmax))
        print(f"\n{ex['label']}")
        print(f"  Budget: ${ex['budget']:,.0f} | Genre: {ex['genre']} (pop={genre_pop_val:.1f}) | Studios: {ex['studios']}")
        print(f"  -> Success score: {score:.1f}/100  =>  {category}")

    # --- Apply to real dataset rows for spot-checking ---
    print("\n--- Applying to 5 real movies from the dataset ---")
    sample = df_valid.dropna(subset=["production_company_names"]).sample(
        min(5, len(df_valid)), random_state=42
    )
    for _, row in sample.iterrows():
        genre_list = row["genre_names"]
        genre_pop_val = (
            np.mean([genre_scores.get(g, 50) for g in genre_list]) if genre_list else 50
        )
        studios = len(row["production_company_names"])
        score, category = predict_success(system, row["budget"], genre_pop_val, studios, (bmin, bmax))
        print(f"\n{row['title']}")
        print(f"  Actual revenue: ${row['revenue']:,.0f}" if pd.notna(row["revenue"]) else "  Actual revenue: unknown")
        print(f"  Predicted success: {score:.1f}/100 => {category}")

    print(
        "\nNote: mismatches between predicted and actual (e.g. a 'Low' "
        "prediction that made huge money) are expected and interesting -"
        " they're candidate sleeper hits. The anomaly detection module "
        "will formalize this by flagging exactly these gaps systematically."
    )


if __name__ == "__main__":
    main()
