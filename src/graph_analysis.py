"""
graph_analysis.py
-------------------
Builds a collaboration network from movie cast + director data: an edge
connects two people if they worked on the same film together. Then runs
centrality analysis (who are the most "connected" industry figures) and
community detection (which groups of people tend to work together
repeatedly - e.g. a director's regular ensemble).

Uses movies_clean.pkl from data_prep.py.

Run from project root:
    python src/graph_analysis.py
"""

import os
import itertools
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MIN_APPEARANCES = 4  # only include people who show up in >= N films
                       # (keeps the graph readable and avoids one-off noise)


def load_data():
    path = os.path.join(PROCESSED_DIR, "movies_clean.pkl")
    return pd.read_pickle(path)


def get_people_per_film(df):
    """Return a list of (film_title, [people...]) where people = top
    cast + director for that film, each prefixed to distinguish actors
    from directors sharing the same real name (rare but possible)."""
    films = []
    for _, row in df.iterrows():
        people = [f"Actor: {a}" for a in row["top_cast"]]
        if pd.notna(row["director"]):
            people.append(f"Director: {row['director']}")
        films.append((row["title"], people))
    return films


def filter_frequent_people(films, min_appearances):
    """Count how many films each person appears in, keep only the ones
    above the threshold - this keeps the graph focused on people with
    an actual body of work, not one-time background names."""
    counts = {}
    for _, people in films:
        for p in people:
            counts[p] = counts.get(p, 0) + 1

    frequent = {p for p, c in counts.items() if c >= min_appearances}
    print(f"People appearing in >= {min_appearances} films: {len(frequent)} "
          f"(out of {len(counts)} total unique people)")
    return frequent


def build_graph(films, frequent_people):
    """Nodes = people. Edge between two people if they worked on the
    same film together, weighted by how many films they've shared."""
    G = nx.Graph()
    G.add_nodes_from(frequent_people)

    for title, people in films:
        people_in_graph = [p for p in people if p in frequent_people]
        for a, b in itertools.combinations(sorted(set(people_in_graph)), 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
                G[a][b]["films"].append(title)
            else:
                G.add_edge(a, b, weight=1, films=[title])

    # Drop isolated nodes (people who never co-appeared with another
    # frequent person, so they add nothing to the network structure)
    G.remove_nodes_from(list(nx.isolates(G)))

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def show_centrality(G, top_n=15):
    """Who are the most 'connected' people in the industry, by three
    different measures - each captures a different kind of influence."""

    degree = nx.degree_centrality(G)
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight", k=min(200, G.number_of_nodes()))

    print(f"\n--- Top {top_n} by degree centrality (most distinct collaborators) ---")
    for person, score in sorted(degree.items(), key=lambda x: -x[1])[:top_n]:
        print(f"  {person:35s} {score:.3f}  ({G.degree(person)} collaborators)")

    print(f"\n--- Top {top_n} by weighted degree (most total collaborations) ---")
    for person, score in sorted(weighted_degree.items(), key=lambda x: -x[1])[:top_n]:
        print(f"  {person:35s} {score} total film-collaborations")

    print(f"\n--- Top {top_n} by betweenness centrality (bridges between groups) ---")
    for person, score in sorted(betweenness.items(), key=lambda x: -x[1])[:top_n]:
        print(f"  {person:35s} {score:.4f}")

    return degree, weighted_degree, betweenness


def show_communities(G, min_edge_weight=2, top_n=8, max_members_shown=10):
    """Detect clusters of people who tend to work together repeatedly -
    e.g. a director's regular ensemble, or a tight-knit group of
    collaborators across several films.

    Community detection is run on a filtered subgraph that only keeps
    edges representing >= min_edge_weight shared films. Without this,
    dense collaboration graphs tend to collapse into a few huge, vague
    "everyone who worked in mainstream film" blobs instead of genuinely
    tight, interpretable clusters - one-time co-appearances add noise
    without adding real community structure.
    """

    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= min_edge_weight]
    G_strong = G.edge_subgraph(strong_edges).copy()

    print(f"\nFiltered to edges with >= {min_edge_weight} shared films: "
          f"{G_strong.number_of_edges()} edges, {G_strong.number_of_nodes()} nodes remain "
          f"(from {G.number_of_edges()} edges, {G.number_of_nodes()} nodes)")

    if G_strong.number_of_nodes() == 0:
        print("No edges survive this threshold - lowering min_edge_weight.")
        return show_communities(G, min_edge_weight=1, top_n=top_n, max_members_shown=max_members_shown)

    communities = list(greedy_modularity_communities(G_strong, weight="weight"))
    communities = sorted(communities, key=len, reverse=True)

    print(f"\n--- Found {len(communities)} communities (collaboration clusters) ---")
    print(f"Showing top {top_n} largest:\n")

    for i, community in enumerate(communities[:top_n]):
        members = sorted(community)
        print(f"Community {i+1} ({len(members)} people):")
        for m in members[:max_members_shown]:
            print(f"    {m}")
        if len(members) > max_members_shown:
            print(f"    ... and {len(members) - max_members_shown} more")
        print()

    return communities


def main():
    df = load_data()
    print(f"Loaded {len(df)} movies")

    films = get_people_per_film(df)
    frequent_people = filter_frequent_people(films, MIN_APPEARANCES)

    G = build_graph(films, frequent_people)

    if G.number_of_nodes() == 0:
        print("Graph is empty - try lowering MIN_APPEARANCES.")
        return

    show_centrality(G)
    show_communities(G)


if __name__ == "__main__":
    main()
