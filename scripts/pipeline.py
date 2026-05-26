"""Single-source-of-truth for the Analisi News pipeline.

This module exposes `run_pipeline()` which replicates, end-to-end, sections
4-26 of the Jupyter notebook. It is consumed by:

- `report.run_report` to build the standalone HTML deliverable
- Optionally the notebook itself, if the user wants to call it instead of
  re-running cells manually (purely additive, the notebook is not modified)

The return value is a `PipelineOutput` dataclass that bundles every object
the downstream report builder needs. Anything that is *only* used inside
the pipeline (intermediate matrices, ego profiles, etc.) is kept local.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

# ----- Pipeline configuration -------------------------------------------------
from scripts.config import (
    ALPHA,
    CORR_THRESHOLD,
    CSV_FILE,
    CUMULATIVE_THRESHOLD,
    KEYWORD_MAX_WORDS,
    MIN_ARTICLES,
    MIN_SCORES,
    MIN_SOURCES,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    N_RUNS,
    TOP_N,
    TZ_ROME,
)

# ----- NLP models -------------------------------------------------------------
from scripts.models import embedding_model, nlp_it

# ----- Pipeline modules -------------------------------------------------------
from scripts.cluster_merge import merge_similar_clusters
from scripts.cluster_ranking import (
    compute_cluster_composite_score,
    compute_cluster_metrics,
    extract_framing_per_cluster,
    map_clusters_to_concepts,
    select_eligible_top_clusters,
)
from scripts.communities import rank_communities, run_louvain_best_partition
from scripts.fuzzy_classifier import classify_nodes
from scripts.graph_builder import cooccurrence_matrix_pipe, graph_from_cooc
from scripts.graph_metrics import compute_node_metrics, normalize_metrics
from scripts.ingestion import build_news_dataframe, read_feeds_csv
from scripts.keyword_extraction import extract_keywords_corpus_with_types
from scripts.lemmatization import LemmaResolver, aggregate_tags_and_entities
from scripts.llm_narration import (
    analyse_top_communities,
    summarise_cluster,
    summarise_day_recap,
)
from scripts.preprocessing import (
    classify_keyword_type,
    clean_pipe_edges,
    merge_entity_lists,
    preprocess_text,
)
from scripts.ranking import (
    build_ego_profiles,
    compute_combined_score,
    compute_relevance_score,
    select_top_nodes,
)
from scripts.recap import select_articles_for_recap
from scripts.viz_news_render import score_news_importance
from scripts.viz_palette import set_palette


logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """Bundle of every object needed by the downstream report renderer.

    Anything used only inside the pipeline (intermediate matrices, ego
    profiles, correlation matrices) is intentionally left out.
    """

    # Timestamp the run started — used for the report header / filename
    generated_at: datetime

    # Article corpora
    df: pd.DataFrame                       # full ingestion (all dates)
    data: pd.DataFrame                     # working set (last 2 days)
    distinct_journals: list[str]
    journal_color_map: dict[str, str]

    # Graph & rankings
    concepts_graph: nx.Graph
    tags_ranking_df: pd.DataFrame
    lemma_to_forms: dict[str, set[str]]
    top_nodes: pd.DataFrame                # top-N by combined_score (used by wordcloud)

    # Section 26 — daily recap
    day_recap: str

    # Section 21 — top 10 news with highlight
    top10_news: pd.DataFrame

    # Section 23 — community detection
    community_results: dict[int, dict[str, Any]]

    # Section 24/25 — top clusters, summaries, framing
    top5_clusters: pd.DataFrame
    cluster_summaries: dict[int, dict[str, str]]
    framing_by_cluster: dict[int, pd.DataFrame]
    top_data: pd.DataFrame                 # subset of `data` restricted to top topics
    entities_set: set[str]

    # Run metadata (footer)
    n_articles: int = field(init=False)
    n_sources: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_articles = int(len(self.data))
        self.n_sources = int(self.data["source"].nunique())


# -----------------------------------------------------------------------------#
# Helper                                                                       #
# -----------------------------------------------------------------------------#
def _build_themes_df(
    data: pd.DataFrame,
    labels: np.ndarray,
    embedding_model_,
    nlp,
    stop_words_italian,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Replica sections 9-11 of the notebook: TF-IDF + KeyBERT per cluster,
    then merge into a single themes_df.

    Returns:
        themes_df: ['Topic', 'Tags', 'Entities', 'News']
        news_groups: dict used downstream by section 12 merge logic
        data: with cluster-level columns merged in
    """
    # Section 9 — TF-IDF + NER classification per cluster
    topics_dict: dict[str, list] = {
        "Topic": [-1],
        "Name": ["Misc"],
        "Entities": ["Misc"],
    }

    for label in sorted(set(labels)):
        if label == -1:
            continue

        docs = data[data["Topic"] == label]["clean text"].tolist()
        if not docs:
            continue

        docs_lower = [d.lower() for d in docs]
        vec = TfidfVectorizer().fit(docs_lower)
        terms = vec.get_feature_names_out()
        top_terms = terms[vec.idf_.argsort()][:5].tolist()

        cluster_text = " ".join(docs)
        doc_spacy = nlp(cluster_text)

        entity_terms: list[str] = []
        generic_terms: list[str] = []
        for term in top_terms:
            is_entity, _ = classify_keyword_type(term, doc_spacy, nlp)
            (entity_terms if is_entity else generic_terms).append(term)

        topics_dict["Topic"].append(label)
        topics_dict["Name"].append("|".join(generic_terms) if generic_terms else "")
        topics_dict["Entities"].append("|".join(entity_terms) if entity_terms else "")

    data = data.merge(pd.DataFrame(topics_dict))

    # Section 10 — KeyBERT per cluster
    news_groups: dict[str, list] = {"Topic": [], "News": []}
    for label in sorted(set(labels)):
        if label >= 0:
            docs = data[data["Topic"] == label]["clean text"].tolist()
            news_groups["Topic"].append(label)
            news_groups["News"].append("\n\n".join(docs))

    keywords_per_cluster = extract_keywords_corpus_with_types(
        docs=news_groups["News"],
        model=embedding_model_,
        ngram_range=(1, 2),
        top_n=5,
        stop_words=stopwords.words("italian"),
        use_mmr=True,
        diversity=0.7,
        nlp=nlp,
    )

    news_groups["Keywords"] = [
        "|".join(kl["keyword"] for kl in doc["other_keywords"])
        for doc in keywords_per_cluster
    ]
    news_groups["Entities2"] = [
        "|".join(el["keyword"] for el in doc["entities"])
        for doc in keywords_per_cluster
    ]

    data = data.merge(pd.DataFrame(news_groups)[["Topic", "Keywords", "Entities2"]])

    # Section 11 — consolidate
    themes_df = pd.DataFrame(topics_dict).merge(pd.DataFrame(news_groups))
    themes_df.drop(themes_df[themes_df["Topic"] == -1].index, inplace=True)
    themes_df.columns = ["Topic", "Keys", "Entities", "News", "Keys2", "Entities2"]

    themes_df["Entities"] = themes_df.apply(
        lambda row: merge_entity_lists(row["Entities"], row["Entities2"]), axis=1
    )
    themes_df["Entities"] = themes_df["Entities"].apply(clean_pipe_edges)
    themes_df.drop("Entities2", axis=1, inplace=True)

    themes_df["Tags"] = (themes_df["Keys"] + "|" + themes_df["Keys2"]).apply(
        lambda s: "|".join(set(s.replace("|", " ").split()))
    )
    themes_df = themes_df[["Topic", "Tags", "Entities", "News"]].copy()
    themes_df["Tags"] = themes_df["Tags"].apply(clean_pipe_edges)

    return themes_df, news_groups, data


# -----------------------------------------------------------------------------#
# Public entry point                                                           #
# -----------------------------------------------------------------------------#
def run_pipeline() -> PipelineOutput:
    """Run the full Analisi News pipeline (notebook sections 4-26).

    Returns a `PipelineOutput` carrying every object the report module needs.
    """
    # Timestamp the run started, ALWAYS in Italy's timezone.
    # datetime.now() without a tz returns the system local time — which is
    # UTC on the GitHub Actions runner, not Rome. The report is for Italian
    # readers, so generated_at must be Rome time regardless of where the
    # pipeline runs. TZ_ROME is the same zone already used for news dates.
    generated_at = datetime.now(TZ_ROME)
    logger.info("Pipeline start: %s", generated_at.isoformat())

    # -- 4. Ingestion -----------------------------------------------------------
    feeds = read_feeds_csv(CSV_FILE)
    if not feeds:
        raise SystemExit(f"No feeds found in {CSV_FILE}. Expected header: source,section,url")

    df = build_news_dataframe(feeds)
    df["date"] = pd.to_datetime(df["date"])
    df["text"] = df["title"].fillna("") + "\n" + df["description"].fillna("")
    logger.info("Articles collected (deduplicated): %d", len(df))

    # Working dataset — last 2 days
    data = df[["source", "section", "date", "time", "text"]].copy()
    last_days = list(np.sort(data["date"].unique())[-2:])
    data = data[data["date"].apply(lambda d: d in last_days)].reset_index(drop=True)

    distinct_journals = data["source"].unique().tolist()
    journal_color_map = {
        j: c for j, c in zip(distinct_journals, set_palette(len(distinct_journals)))
    }

    # -- 5. Preprocessing -------------------------------------------------------
    data["clean text"] = data["text"].apply(
        lambda t: " ".join(
            preprocess_text(" ".join(str(t).split()[:KEYWORD_MAX_WORDS]))
        )
    )

    # -- 6. Semantic clustering -------------------------------------------------
    emb = embedding_model.encode(data["text"].tolist(), show_progress_bar=False)
    umap_emb = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=0,
    ).fit_transform(emb)
    labels = HDBSCAN(
        min_cluster_size=2, metric="euclidean", cluster_selection_method="eom",
    ).fit_predict(umap_emb)
    data["Topic"] = labels
    logger.info("Clusters found: %d (excluding noise)", data["Topic"].nunique() - 1)

    # -- 9-11. Build themes_df --------------------------------------------------
    themes_df, news_groups, data = _build_themes_df(
        data, labels, embedding_model, nlp_it, None
    )

    # -- 12. KeyBERT per single news --------------------------------------------
    keywords_per_single_news = extract_keywords_corpus_with_types(
        docs=data["clean text"].tolist(),
        model=embedding_model,
        ngram_range=(1, 2), top_n=5,
        stop_words=stopwords.words("italian"),
        use_mmr=True, diversity=0.7, nlp=nlp_it,
    )
    data = data[data.columns[:7]].copy()
    data["Keywords"] = [
        "|".join(kl["keyword"] for kl in doc["other_keywords"])
        for doc in keywords_per_single_news
    ]
    data["Entities"] = [
        "|".join(el["keyword"] for el in doc["entities"])
        for doc in keywords_per_single_news
    ]
    data["Keywords"] = data["Keywords"].apply(
        lambda s: "|".join(set(s.replace("|", " ").split()))
    )
    data["Entities"] = data["Entities"].apply(
        lambda s: "|".join(set(s.replace("|", " ").split()))
    )

    # -- 13. Aggregation --------------------------------------------------------
    tags_list, entities_list = aggregate_tags_and_entities(themes_df, data)

    # -- 14. Lemmatization ------------------------------------------------------
    resolver = LemmaResolver(tags_list, entities_list)
    important_keywords_set = resolver.important_keywords_set
    lemma_to_forms = resolver.lemma_to_forms

    # -- 15. Co-occurrence inputs -----------------------------------------------
    themes_df["Tags"] = themes_df.apply(
        lambda row: merge_entity_lists(row["Entities"], row["Tags"]), axis=1
    )
    themes_df["Tags"] = themes_df["Tags"].apply(clean_pipe_edges)
    themes_df["Tags"] = themes_df["Tags"].apply(lambda s: "|".join(set(s.split("|"))))
    themes_df["Tags"] = themes_df["Tags"].apply(clean_pipe_edges)

    similarity_by_topic_list = [
        resolver.apply_lemmas_to_cooc_string(
            "|".join(t for t in s.split("|") if t in important_keywords_set)
        )
        for s in themes_df["Tags"].tolist()
    ]

    data["clean text"] = (
        data["clean text"]
        .str.replace("adnkronos", "", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.strip()
    )
    data["co_occurrences"] = data["clean text"].apply(
        lambda s: "|".join(set(s.split()))
    )

    similarity_by_same_news = [
        resolver.apply_lemmas_to_cooc_string(
            "|".join(k for k in s.split("|") if k in important_keywords_set)
        )
        for s in data["co_occurrences"].tolist()
    ]

    # -- 16. Graph construction -------------------------------------------------
    words_matrix, vocab = cooccurrence_matrix_pipe(
        similarity_by_topic_list + similarity_by_same_news
    )

    values, counts = np.unique(words_matrix.data, return_counts=True)
    weight_by_frequency = values * counts
    cumulative_perc_importance = np.cumsum(weight_by_frequency / weight_by_frequency.sum())

    i = 0
    while cumulative_perc_importance[i] < CUMULATIVE_THRESHOLD:
        i += 1
    threshold_val = max(int(values[i].item()) - 1, 2)

    mask = words_matrix.data < threshold_val
    words_matrix.data[mask] = 0
    words_matrix.eliminate_zeros()

    concepts_graph = graph_from_cooc(words_matrix, vocab.tolist())
    logger.info(
        "Graph: %d nodes, %d edges",
        concepts_graph.number_of_nodes(), concepts_graph.number_of_edges(),
    )

    # -- 17. Metrics ------------------------------------------------------------
    metrics_raw, cluster_freq = compute_node_metrics(
        concepts_graph, similarity_by_topic_list, similarity_by_same_news,
    )
    metrics_norm = normalize_metrics(metrics_raw)

    metric_cols = [
        "frequency", "cluster_freq", "degree", "avg_neigh_deg", "betweenness",
        "harmonic", "eigen", "clustering", "constraint", "core_n",
    ]
    corr = metrics_norm[metric_cols].corr(method="spearman")

    # -- 18. Fuzzy classification ----------------------------------------------
    tags_ranking_df = classify_nodes(
        metrics_norm=metrics_norm, corr=corr,
        corr_threshold=CORR_THRESHOLD, min_scores=MIN_SCORES,
    )

    # -- 19. Ranking ------------------------------------------------------------
    tags_ranking_df = compute_relevance_score(tags_ranking_df, metrics_norm)
    tags_ranking_df = tags_ranking_df.sort_values("relevance_score", ascending=False)
    ego_profiles_df = build_ego_profiles(
        concepts_graph, tags_ranking_df, target_roles=("core", "bridge"),
    )
    tags_ranking_df = compute_combined_score(tags_ranking_df, ego_profiles_df)
    tags_ranking_df = tags_ranking_df.sort_values("combined_score", ascending=False)
    top_nodes = select_top_nodes(tags_ranking_df, TOP_N)

    # -- 21. Top 10 news --------------------------------------------------------
    top10_news = score_news_importance(
        df=df, top_nodes=top_nodes, lemma_to_forms=lemma_to_forms, top_n=10,
    )

    # -- 23. Communities --------------------------------------------------------
    best_communities, best_modularity = run_louvain_best_partition(
        concepts_graph, n_runs=N_RUNS,
    )
    logger.info("Best modularity: %.4f over %d runs", best_modularity, N_RUNS)

    community_scores_df, top5_comm_idx = rank_communities(
        best_communities, tags_ranking_df, top_n=5,
    )
    community_results = analyse_top_communities(
        top_comm_indices=top5_comm_idx,
        best_communities=best_communities,
        tags_ranking_df=tags_ranking_df,
        df=df,
        lemma_to_forms=lemma_to_forms,
        concepts_graph=concepts_graph,
        api_key=MISTRAL_API_KEY,
        embedding_model=embedding_model,
        model=MISTRAL_MODEL,
        top_n=5,
    )

    # -- 23b. Conservative cluster pre-merge (LOCAL copies only) ----------------
    # Near-duplicate HDBSCAN clusters are merged immediately before section 24
    # so the cross-source coverage analysis is not fragmented across variants
    # of the same story. This works on LOCAL copies: `data`, `themes_df` and
    # `similarity_by_topic_list` above are NOT modified — everything produced
    # by sections 4-23 stays exactly as it was. Only section 24 sees the
    # merged copies.
    #
    # Embeddings are RECOMPUTED here on the current `data`: the `emb` array
    # from section 6 was built on an earlier version of `data` (before the
    # merges in sections 9-12) and is no longer row-aligned with it.
    emb_s24 = embedding_model.encode(data["text"].tolist(), show_progress_bar=False)
    merge_result = merge_similar_clusters(
        data=data,
        themes_df=themes_df,
        similarity_by_topic_list=similarity_by_topic_list,
        embeddings=emb_s24,
    )
    data_s24 = merge_result.data
    themes_df_s24 = merge_result.themes_df
    similarity_by_topic_list_s24 = merge_result.similarity_by_topic_list

    # -- 24. Cluster ranking (on the merged copies) -----------------------------
    cluster_concepts = map_clusters_to_concepts(
        themes_df_s24, similarity_by_topic_list_s24, tags_ranking_df,
    )
    clusters_df = compute_cluster_metrics(cluster_concepts, data_s24, tags_ranking_df)
    clusters_df = compute_cluster_composite_score(clusters_df, alpha=ALPHA)
    top5_clusters = select_eligible_top_clusters(
        clusters_df,
        min_articles=MIN_ARTICLES, min_sources=MIN_SOURCES, top_n=5,
    )[1]

    top_topics = top5_clusters["topic_id"].tolist()
    top_data = data_s24[data_s24["Topic"].apply(lambda t: t in top_topics)].copy()
    entities_set = resolver.entities_set

    # -- 25. Cluster summary + framing (on the merged copies) -------------------
    cluster_summaries: dict[int, dict[str, str]] = {}
    framing_by_cluster: dict[int, pd.DataFrame] = {}
    for _, row in top5_clusters.iterrows():
        topic_id = row["topic_id"]
        cluster_articles = data_s24[data_s24["Topic"] == topic_id]
        summary = summarise_cluster(
            MISTRAL_API_KEY, topic_id, cluster_articles, row["concepts"],
        )
        cluster_summaries[topic_id] = summary
        framing_by_cluster[topic_id] = extract_framing_per_cluster(
            cluster_articles, entities_set,
        )

    # -- 26. Day recap ----------------------------------------------------------
    recap_articles = select_articles_for_recap(
        clusters_df=clusters_df,
        data=data_s24,
        lemma_to_forms=lemma_to_forms,
        tags_ranking_df=tags_ranking_df,
        top_n_clusters=10,
        top_k_news_per_cluster=3,
        min_articles=MIN_ARTICLES, min_sources=MIN_SOURCES,
    )
    day_recap = summarise_day_recap(
        api_key=MISTRAL_API_KEY,
        recap_articles_df=recap_articles,
        model=MISTRAL_MODEL,
    )

    logger.info("Pipeline complete: %s", datetime.now().isoformat())

    return PipelineOutput(
        generated_at=generated_at,
        df=df, data=data,
        distinct_journals=distinct_journals,
        journal_color_map=journal_color_map,
        concepts_graph=concepts_graph,
        tags_ranking_df=tags_ranking_df,
        lemma_to_forms=lemma_to_forms,
        top_nodes=top_nodes,
        day_recap=day_recap,
        top10_news=top10_news,
        community_results=community_results,
        top5_clusters=top5_clusters,
        cluster_summaries=cluster_summaries,
        framing_by_cluster=framing_by_cluster,
        top_data=top_data,
        entities_set=entities_set,
    )
