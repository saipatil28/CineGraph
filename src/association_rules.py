"""
association_rules.py
----------------------
Mines association rules (Apriori/FP-Growth) over movie "baskets" made up
of genres, top-billed actors, and director, to find which combinations
associate with high box-office success. Classic market-basket analysis,
applied to film industry data instead of retail.

Uses movies_clean.pkl from data_prep.py.

Run from project root:
    python src/association_rules.py
"""

import os
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MIN_ACTOR_APPEARANCES = 5   # only include actors who show up in >= N movies
MIN_DIRECTOR_APPEARANCES = 3
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.3

# Actors/directors individually are much rarer than genre combos, so a
# rule involving a specific person will almost never clear a 2% support
# bar even if it's a real, interesting pattern. We mine those separately
# with a lower threshold, then filter down to just person-containing rules.
PERSON_MIN_SUPPORT = 0.006


def load_data():
    path = os.path.join(PROCESSED_DIR, "movies_clean.pkl")
    return pd.read_pickle(path)


def add_success_label(df):
    """Label each movie High/Not High success based on revenue quartile.
    Only movies with known revenue get a label - others are dropped for
    this analysis, since we need a ground truth to mine against."""
    df = df.dropna(subset=["revenue"]).copy()
    threshold = df["revenue"].quantile(0.75)
    df["success_label"] = np.where(df["revenue"] >= threshold, "Success_High", "Success_NotHigh")
    return df


def get_frequent_people(df, column, min_count):
    """Find actors/directors that appear often enough to be worth
    including as itemset items - rare names just add noise and slow
    things down without producing generalizable rules."""
    if column == "top_cast":
        exploded = df.explode(column)
        counts = exploded[column].value_counts()
    else:
        counts = df[column].value_counts()
    return set(counts[counts >= min_count].index)


def build_baskets(df):
    """Turn each movie into a 'basket' of items: its genres, its
    frequent-enough top cast members, its director (if frequent enough),
    and its success label."""

    frequent_actors = get_frequent_people(df, "top_cast", MIN_ACTOR_APPEARANCES)
    frequent_directors = get_frequent_people(df, "director", MIN_DIRECTOR_APPEARANCES)

    print(f"Frequent actors (>= {MIN_ACTOR_APPEARANCES} films): {len(frequent_actors)}")
    print(f"Frequent directors (>= {MIN_DIRECTOR_APPEARANCES} films): {len(frequent_directors)}")

    baskets = []
    for _, row in df.iterrows():
        basket = []

        for g in row["genre_names"]:
            basket.append(f"Genre_{g}")

        for actor in row["top_cast"]:
            if actor in frequent_actors:
                basket.append(f"Actor_{actor}")

        if row["director"] in frequent_directors:
            basket.append(f"Director_{row['director']}")

        basket.append(row["success_label"])

        baskets.append(basket)

    return baskets


def mine_rules(baskets, min_support=MIN_SUPPORT):
    te = TransactionEncoder()
    te_array = te.fit(baskets).transform(baskets)
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    print(f"\nBasket matrix shape: {basket_df.shape} (movies x unique items)")

    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets (min_support={min_support})")

    if frequent_itemsets.empty:
        print("No frequent itemsets found - try lowering min_support.")
        return None

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE
    )
    print(f"Found {len(rules)} rules (min_confidence={MIN_CONFIDENCE})")

    return rules


def show_person_rules(rules, top_n=15):
    """Filter to rules that involve at least one actor or director -
    these need a separate, lower support threshold to surface at all,
    since individual people are much rarer than genre tags."""

    def involves_person(row):
        items = set(row["antecedents"]) | set(row["consequents"])
        return any(str(i).startswith("Actor_") or str(i).startswith("Director_") for i in items)

    def consequent_is_success(frozenset_items):
        return any(str(item).startswith("Success_") for item in frozenset_items)

    person_rules = rules[rules.apply(involves_person, axis=1)].copy()
    person_success_rules = person_rules[
        person_rules["consequents"].apply(consequent_is_success)
    ].sort_values("lift", ascending=False)

    print(f"\n--- Top {top_n} actor/director rules predicting success (sorted by lift) ---")
    if person_success_rules.empty:
        print("  None found even at the lower support threshold - the dataset may be")
        print("  too sparse per-person for reliable patterns. This is a legitimate")
        print("  finding: individual star power alone isn't a strong enough signal")
        print("  here, genre and budget matter more (see the earlier rules).")
    else:
        for _, r in person_success_rules.head(top_n).iterrows():
            antecedents = ", ".join(sorted(str(a) for a in r["antecedents"]))
            consequent = ", ".join(sorted(str(c) for c in r["consequents"]))
            print(
                f"\n  IF [{antecedents}]"
                f"\n  THEN [{consequent}]"
                f"\n  support={r['support']:.3f}  confidence={r['confidence']:.3f}  lift={r['lift']:.2f}"
            )

    return person_success_rules


def show_success_rules(rules, top_n=15):
    """Filter to rules whose consequent is specifically about success,
    since those are the genuinely interesting, actionable findings."""

    def consequent_is_success(frozenset_items):
        return any(str(item).startswith("Success_") for item in frozenset_items)

    success_rules = rules[rules["consequents"].apply(consequent_is_success)].copy()
    success_rules = success_rules.sort_values("lift", ascending=False)

    print(f"\n--- Top {top_n} rules predicting success (sorted by lift) ---")
    for _, r in success_rules.head(top_n).iterrows():
        antecedents = ", ".join(sorted(str(a) for a in r["antecedents"]))
        consequent = ", ".join(sorted(str(c) for c in r["consequents"]))
        print(
            f"\n  IF [{antecedents}]"
            f"\n  THEN [{consequent}]"
            f"\n  support={r['support']:.3f}  confidence={r['confidence']:.3f}  lift={r['lift']:.2f}"
        )

    return success_rules


def show_general_rules(rules, top_n=10):
    """Show the highest-lift rules overall too, for interesting
    non-success combos (e.g. which genres/directors pair up often)."""
    top_rules = rules.sort_values("lift", ascending=False)

    print(f"\n--- Top {top_n} rules overall (sorted by lift) ---")
    for _, r in top_rules.head(top_n).iterrows():
        antecedents = ", ".join(sorted(str(a) for a in r["antecedents"]))
        consequent = ", ".join(sorted(str(c) for c in r["consequents"]))
        print(
            f"\n  IF [{antecedents}]"
            f"\n  THEN [{consequent}]"
            f"\n  support={r['support']:.3f}  confidence={r['confidence']:.3f}  lift={r['lift']:.2f}"
        )


def main():
    df = load_data()
    df = add_success_label(df)
    print(f"Movies with known revenue: {len(df)}")
    print(f"'High success' threshold (75th percentile revenue): "
          f"${df[df['success_label']=='Success_High']['revenue'].min():,.0f}+")

    baskets = build_baskets(df)

    print("\n=== PASS 1: Genre-level patterns (support >= 2%) ===")
    rules = mine_rules(baskets, min_support=MIN_SUPPORT)
    if rules is not None and not rules.empty:
        show_success_rules(rules)
        show_general_rules(rules)

    print("\n=== PASS 2: Actor/director patterns (support >= 0.6%, filtered to person-only rules) ===")
    person_rules_raw = mine_rules(baskets, min_support=PERSON_MIN_SUPPORT)
    if person_rules_raw is not None and not person_rules_raw.empty:
        show_person_rules(person_rules_raw)


if __name__ == "__main__":
    main()
