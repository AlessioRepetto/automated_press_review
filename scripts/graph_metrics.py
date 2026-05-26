# Graph metrics.
#
# Ten metrics per node, capturing complementary aspects of structural
# importance:
#
#   frequency       Volume in cluster + document signals
#   cluster_freq    Thematic diffusion (in how many distinct clusters)
#   degree          Total weighted connectivity
#   avg_neigh_deg   Connectivity of immediate neighbors
#   betweenness     Bridge potential (fraction of shortest paths)
#   harmonic        Global reachability
#   eigen           Structural prestige (connected to other prestigious nodes)
#   clustering      Compactness of the local community
#   constraint      Connection redundancy (low = brokerage)
#   core_n          Depth in the k-core decomposition
#
# All metrics are converted to percentile rank in [0, 1] so the downstream
# fuzzy classifier can reason in terms relative to the daily corpus rather
# than fixed absolute thresholds.

import networkx as nx
import pandas as pd


METRIC_COLS = [
    "frequency", "cluster_freq", "degree", "avg_neigh_deg", "betweenness",
    "harmonic", "eigen", "clustering", "constraint", "core_n",
]


def compute_node_metrics(concepts_graph,
                          similarity_by_topic_list,
                          similarity_by_same_news):
    # Computes the ten structural metrics for each node in the graph and
    # merges them with the frequency and cluster_freq signals derived from
    # the co-occurrence strings.
    #
    # Returns metrics_raw, a DataFrame indexed by tag with one column per metric.
    degree_dict = dict(concepts_graph.degree(weight="weight"))

    avg_neighbor_degree = {}
    for u in concepts_graph.nodes():
        neigh = list(concepts_graph.neighbors(u))
        if not neigh:
            avg_neighbor_degree[u] = 0.0
            continue
        num = sum(concepts_graph[u][v].get("weight", 1.0) * degree_dict[v] for v in neigh)
        den = sum(concepts_graph[u][v].get("weight", 1.0) for v in neigh)
        avg_neighbor_degree[u] = num / den if den > 0 else 0.0

    betweenness_dict = dict(nx.betweenness_centrality(concepts_graph, weight="weight"))
    harmonic_dict    = dict(nx.harmonic_centrality(concepts_graph))
    eigen_dict       = dict(nx.eigenvector_centrality(concepts_graph, weight="weight", max_iter=500))
    clustering_dict  = nx.clustering(concepts_graph, weight="weight")
    constraint_dict  = nx.constraint(concepts_graph, weight="weight")
    core_n_dict      = nx.core_number(concepts_graph)

    # Compute term frequency from the co-occurrence strings.
    # all_terms_ordered counts in how many co-occurrence strings (cluster
    # or document) each term appears, consistently with binary=True in
    # the matrix builder.
    all_terms_ordered = pd.Series(
        ("|".join(similarity_by_topic_list + similarity_by_same_news)).split("|")
    ).value_counts()

    # Cluster frequency: number of distinct clusters in which each term appears.
    # Captures thematic diffusion across topics, distinct from raw frequency
    # which instead counts total occurrences across cluster and document signals.
    cluster_freq = pd.Series(
        [t for s in similarity_by_topic_list for t in s.split("|") if t]
    ).value_counts()

    metrics_raw = pd.DataFrame({
        "tag":           list(degree_dict.keys()),
        "degree":        list(degree_dict.values()),
        "avg_neigh_deg": [avg_neighbor_degree[n] for n in degree_dict],
        "betweenness":   [betweenness_dict[n]    for n in degree_dict],
        "harmonic":      [harmonic_dict[n]        for n in degree_dict],
        "eigen":         [eigen_dict[n]            for n in degree_dict],
        "clustering":    [clustering_dict[n]       for n in degree_dict],
        "constraint":    [constraint_dict[n]       for n in degree_dict],
        "core_n":        [core_n_dict[n]           for n in degree_dict],
    })

    metrics_raw = metrics_raw.merge(
        pd.DataFrame({"tag": all_terms_ordered.index, "frequency": all_terms_ordered.values}),
        on="tag", how="left"
    ).fillna(0)

    metrics_raw = metrics_raw.merge(
        pd.DataFrame({"tag": cluster_freq.index, "cluster_freq": cluster_freq.values}),
        on="tag", how="left"
    ).fillna(0)

    return metrics_raw, cluster_freq


def normalize_metrics(metrics_raw):
    # Converts all metrics to percentile rank in [0, 1].
    metrics_norm = metrics_raw[["tag"]].copy()
    for col in METRIC_COLS:
        metrics_norm[col] = metrics_raw[col].rank(pct=True)
    return metrics_norm


def compute_correlation_matrix(metrics_norm):
    # Spearman correlation between normalized metrics. Used by the
    # fuzzy classifier to discard avg_neigh_deg if too correlated with
    # eigen (avoids double counting).
    return metrics_norm[METRIC_COLS].corr(method="spearman")
