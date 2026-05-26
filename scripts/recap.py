# Daily recap article selection.
#
# The daily editorial recap is built on top of the cluster ranking (NOT on
# the top-10-news-by-density list used for highlighted articles).
#
# Density-based scoring biases selection toward the dominant story of the
# day, producing redundant picks (e.g. several articles about the same
# event from different agencies). Selecting the top-N clusters and taking
# the top-K articles per cluster instead guarantees thematic variety in
# the input pool, which is what the recap needs.

import pandas as pd

from scripts.communities import score_news_for_community


def select_articles_for_recap(clusters_df, data, lemma_to_forms,
                              tags_ranking_df,
                              top_n_clusters=10, top_k_news_per_cluster=3,
                              min_articles=3, min_sources=2):
    # Picks, from each of the top-N most informative clusters, the top-K
    # articles most representative of that cluster.
    #
    # The same eligibility filter as the cross-source analysis is applied
    # (min_articles, min_sources) so a cluster is included only if it has
    # enough mass and source diversity to count as a real story.
    #
    # Within a cluster, articles are ranked with score_news_for_community
    # (concept density weighted by combined_score, normalised by log word
    # count), and at most one article per source per cluster is kept, to
    # avoid near-identical reports of the same event within a single cluster.

    eligible_clusters = clusters_df[
        (clusters_df["n_articles"] >= min_articles) &
        (clusters_df["n_sources"]  >= min_sources)
    ].head(top_n_clusters)

    selected_rows = []
    for _, row in eligible_clusters.iterrows():
        topic_id = row["topic_id"]
        concepts = set(row["concepts"])
        if not concepts:
            continue

        cluster_articles = data[data["Topic"] == topic_id].copy()
        if cluster_articles.empty:
            continue

        ranked = score_news_for_community(
            comm=concepts,
            news_df=cluster_articles,
            lemma_to_forms=lemma_to_forms,
            tags_ranking_df=tags_ranking_df,
            top_n=top_k_news_per_cluster,
        )
        if ranked.empty:
            continue

        ranked = ranked.copy()
        ranked["topic_id"] = topic_id
        selected_rows.append(ranked)

    if not selected_rows:
        return pd.DataFrame(
            columns=["source", "time", "text", "matched_concepts", "topic_id"]
        )

    return pd.concat(selected_rows, ignore_index=True)
