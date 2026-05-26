# Node ranking: global vs local importance.
#
# For each node we compute two complementary ranking dimensions:
#
# Global importance - relevance_score
#   Composite score derived from graph-level structural metrics:
#   betweenness, harmonic, core number, cluster frequency, and constraint
#   (inverted). It measures how central and reachable a node is with
#   respect to the entire network.
#
# Local importance - ego-graph size and composition
#   For each node, we extract the radius-1 ego graph and characterize it
#   by size and role composition in its neighborhood.
#
# Combined score
#   combined_score = 0.6 * relevance_score + 0.4 * ego_size_norm
#
# The two dimensions are orthogonal and complementary:
#   - A core node is globally central but locally homogeneous
#   - A bridge node is locally rich and diversified but may be less
#     prominent at the global graph level

from collections import Counter

import networkx as nx
import pandas as pd


# Weights for relevance score.
# Reflect informational value, not structural role:
#   - betweenness and harmonic capture reach across the graph
#   - core_n captures structural embedding
#   - cluster_freq captures thematic breadth
#   - constraint (inverted) rewards brokerage potential
#   - frequency is a minor signal to avoid a pure volume bias
RELEVANCE_WEIGHTS = {
    "betweenness":  0.25,
    "harmonic":     0.20,
    "core_n":       0.20,
    "cluster_freq": 0.15,
    "constraint":   0.10,  # inverted below
    "frequency":    0.10,
}


def compute_relevance_score(tags_ranking_df, metrics_norm):
    # Adds the relevance_score column to tags_ranking_df.
    # Returns the modified DataFrame sorted by relevance_score descending.
    tags_ranking_df = tags_ranking_df.copy()

    tags_ranking_df["relevance_score"] = (
        RELEVANCE_WEIGHTS["betweenness"]  * metrics_norm["betweenness"] +
        RELEVANCE_WEIGHTS["harmonic"]     * metrics_norm["harmonic"] +
        RELEVANCE_WEIGHTS["core_n"]       * metrics_norm["core_n"] +
        RELEVANCE_WEIGHTS["cluster_freq"] * metrics_norm["cluster_freq"] +
        RELEVANCE_WEIGHTS["constraint"]   * (1 - metrics_norm["constraint"]) +
        RELEVANCE_WEIGHTS["frequency"]    * metrics_norm["frequency"]
    ).values

    return tags_ranking_df.sort_values("relevance_score", ascending=False)


def ego_profile(G, node, tags_ranking_df):
    # Computes the ego-graph profile for a node:
    #   - total number of neighbors
    #   - neighbor count by role
    ego_G     = nx.ego_graph(G, node, radius=1)
    neighbours = [n for n in ego_G.nodes if n != node]

    role_lookup = tags_ranking_df.set_index("tag")["concept_type"].to_dict()

    role_counts = Counter(role_lookup.get(n, "unclassified") for n in neighbours)
    role_counts["_total"] = len(neighbours)
    role_counts["tag"]  = node

    return role_counts


def build_ego_profiles(concepts_graph, tags_ranking_df,
                       target_roles=("core", "bridge")):
    # Computes ego-graph profiles for all nodes whose role is in target_roles.
    # Returns a DataFrame indexed by tag with role counts and _total.
    profile_rows = []
    target_set = set(target_roles)
    for _, row in tags_ranking_df[tags_ranking_df["concept_type"].isin(target_set)].iterrows():
        profile = ego_profile(concepts_graph, row["tag"], tags_ranking_df)
        profile["concept_type"]    = row["concept_type"]
        profile["relevance_score"] = row["relevance_score"]
        profile_rows.append(profile)

    ego_profiles_df = pd.DataFrame(profile_rows).fillna(0)

    # Reorder columns for readability
    fixed_cols = ["tag", "concept_type", "relevance_score", "_total"]
    role_cols_present = [c for c in ["core", "bridge", "stable", "peripheral",
                                       "generic", "unclassified"]
                         if c in ego_profiles_df.columns]
    ego_profiles_df = ego_profiles_df[fixed_cols + role_cols_present]
    ego_profiles_df = ego_profiles_df.sort_values(
        ["concept_type", "_total"], ascending=[True, False]
    )
    return ego_profiles_df


def compute_combined_score(tags_ranking_df, ego_profiles_df,
                            relevance_weight=0.6, ego_weight=0.4):
    # Combines relevance_score with ego-size into a single ranking signal.
    #
    # combined_score = relevance_weight * relevance_score + ego_weight * ego_size_norm
    #
    # ego_size_norm is the percentile rank of ego size, so the two signals
    # live on a comparable [0, 1] scale.
    tags_ranking_df = tags_ranking_df.copy()

    ego_size = ego_profiles_df.set_index("tag")["_total"].to_dict()
    tags_ranking_df["ego_size"] = tags_ranking_df["tag"].map(ego_size).fillna(0)

    # Percentile-rank normalisation
    tags_ranking_df["ego_size_norm"] = tags_ranking_df["ego_size"].rank(pct=True)

    tags_ranking_df["combined_score"] = (
        relevance_weight * tags_ranking_df["relevance_score"] +
        ego_weight       * tags_ranking_df["ego_size_norm"]
    )

    return tags_ranking_df.sort_values("combined_score", ascending=False)


def select_top_nodes(tags_ranking_df, top_n,
                     allowed_roles=("core", "bridge", "stable", "peripheral")):
    # Filters out generic and unclassified, then takes the top-N rows.
    allowed = set(allowed_roles)
    return tags_ranking_df[
        tags_ranking_df["concept_type"].apply(lambda r: r in allowed)
    ].head(top_n)
