# Community detection.
#
# Louvain community detection identifies the main themes of the day:
# groups of concepts that remain cohesive in the co-occurrence graph.
#
# Modularity stability: Louvain is non-deterministic, so we run N_RUNS
# independent passes with different seeds and keep the partition with
# maximum modularity. Communities are then sorted by the average
# combined_score of their nodes.

import math
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from scripts.config import KEYWORD_MAX_WORDS


# Structural roles considered "distinctive" — i.e. concepts specific enough
# that an article containing them genuinely belongs to the community.
# `generic` and `unclassified` are deliberately excluded: generic concepts
# (cross-cutting hubs such as place names or "gruppo") act as bridges between
# unrelated stories, and using them for article retrieval / scoring is what
# lets, e.g., a cycling article leak into an unrelated foreign-policy theme.
DISTINCTIVE_ROLES = {"core", "bridge", "stable", "peripheral"}

# Minimum number of DISTINCTIVE community concepts an article must contain to
# be assigned to that community. Articles touching only one distinctive
# concept are almost always false positives. Applied with a per-community
# fallback to 1 (see score_news_for_community) so a small community is never
# left with zero representative articles.
MIN_DISTINCTIVE_CONCEPTS = 2


def _distinctive_nodes(comm, tags_ranking_df):
    # Returns the subset of community nodes whose structural role is
    # distinctive (core/bridge/stable/peripheral). Generic / unclassified
    # nodes are dropped. tags_ranking_df must expose 'tag' and 'concept_type'.
    role_lookup = tags_ranking_df.set_index("tag")["concept_type"].to_dict()
    return {n for n in comm if role_lookup.get(n) in DISTINCTIVE_ROLES}


def run_louvain_best_partition(concepts_graph, n_runs):
    # Runs Louvain n_runs times with different seeds and keeps the
    # partition with maximum modularity.
    best_communities = None
    best_modularity  = -1.0

    for seed in range(n_runs):
        communities = nx.community.louvain_communities(
            concepts_graph,
            weight="weight",
            seed=seed,
        )
        modularity = nx.community.modularity(
            concepts_graph,
            communities,
            weight="weight",
        )
        if modularity > best_modularity:
            best_modularity  = modularity
            best_communities = communities

    return best_communities, best_modularity


def rank_communities(best_communities, tags_ranking_df, top_n=None,
                     combined_score_col="combined_score"):
    score_lookup = tags_ranking_df.set_index("tag")[combined_score_col].to_dict()

    community_scores = []
    for i, comm in enumerate(best_communities):
        scores = [score_lookup.get(n, 0.0) for n in comm]
        mean_score = np.mean(scores) if scores else 0.0
        community_scores.append((i, mean_score, len(comm)))

    community_scores_df = pd.DataFrame(
        community_scores,
        columns=["comm_idx", "mean_combined_score", "size"],
    ).sort_values("mean_combined_score", ascending=False).reset_index(drop=True)

    if top_n is None:
        top_comm_idx = community_scores_df["comm_idx"].tolist()
    else:
        top_comm_idx = community_scores_df.head(top_n)["comm_idx"].tolist()

    return community_scores_df, top_comm_idx


# =============================================================================
# Per-community news retrieval and scoring
# =============================================================================

def get_news_for_community(comm, lemma_to_forms, df, tags_ranking_df,
                           text_col="text"):
    # Retrieves all articles containing at least one DISTINCTIVE community
    # node (generic / unclassified nodes are ignored for retrieval).
    #
    # Parameters:
    #   comm            set of canonical node names belonging to the community
    #   lemma_to_forms  mapping canonical lemma to surface forms
    #   df              original news DataFrame
    #   tags_ranking_df concept roles ('tag', 'concept_type') — used to keep
    #                   only distinctive concepts
    #   text_col        column containing article text
    #
    # Note: this is the wide retrieval step (>=1 distinctive concept). The
    # >=2 distinctive-concepts rule with per-community fallback is applied
    # later, in score_news_for_community.

    distinctive = _distinctive_nodes(comm, tags_ranking_df)

    # Fallback: a community made entirely of generic nodes would otherwise
    # match nothing. Extremely rare, but keep the old behaviour in that case.
    nodes_for_retrieval = distinctive if distinctive else set(comm)

    # Build the set of all surface forms of the retained nodes
    all_forms = set()
    for node in nodes_for_retrieval:
        for form in lemma_to_forms.get(node, {node}):
            all_forms.add(form.lower())

    mask = df[text_col].str.lower().apply(
        lambda text: any(form in text for form in all_forms)
    )
    return df[mask].copy()


def build_form_to_score_community(comm, lemma_to_forms, tags_ranking_df,
                                   combined_score_col="combined_score"):
    # Builds the lookup  surface_form_lower -> (combined_score, canonical_node)
    # restricted to the DISTINCTIVE community nodes only.
    #
    # The canonical node is carried alongside the score so the scorer can
    # count how many DISTINCT distinctive concepts an article matches (two
    # surface forms of the same lemma must count once).
    score_lookup = tags_ranking_df.set_index("tag")[combined_score_col].to_dict()
    distinctive = _distinctive_nodes(comm, tags_ranking_df)
    nodes = distinctive if distinctive else set(comm)

    node_forms = {}
    for node in nodes:
        score = score_lookup.get(node, 0.0)
        for form in lemma_to_forms.get(node, {node}):
            node_forms[form.lower()] = (score, node)
    return node_forms


def compute_community_density_score(text, node_forms, max_words=None):
    # Density score for an article relative to a community.
    #
    # node_forms maps  surface_form_lower -> (combined_score, canonical_node).
    # The score is the sum of combined_scores of the DISTINCT canonical nodes
    # matched, divided by log(word_count). Distinct canonical nodes are also
    # returned (n_distinct) so the caller can apply the >=2 rule.
    #
    # Returns: (density, matched_forms, n_distinct_concepts)
    if not isinstance(text, str) or not text.strip():
        return 0.0, [], 0

    cap = max_words if max_words is not None else KEYWORD_MAX_WORDS
    words = text.lower().split()[:cap]
    text_lower = " ".join(words)
    word_count = max(len(words), 1)

    matched = []
    seen_nodes = set()       # distinct canonical nodes — for scoring & counting
    total = 0.0

    for form, (score, node) in node_forms.items():
        if form in text_lower:
            matched.append(form)
            if node not in seen_nodes:
                total += score
                seen_nodes.add(node)

    density = total / math.log(max(word_count, 2))
    return density, matched, len(seen_nodes)


def score_news_for_community(comm, news_df, lemma_to_forms, tags_ranking_df,
                              combined_score_col="combined_score", top_n=5,
                              min_distinctive=MIN_DISTINCTIVE_CONCEPTS):
    # Sorts articles by relevance to a community and applies the
    # distinctive-concepts gate.
    #
    # Score: sum of combined_score of the DISTINCT distinctive community
    # concepts found in the article, divided by log(word_count).
    #
    # Assignment rule (leva B, with fallback): an article is eligible only
    # if it contains at least `min_distinctive` distinct distinctive concepts
    # of the community. If that threshold leaves the community with ZERO
    # eligible articles, the threshold is lowered to 1 FOR THIS COMMUNITY
    # ONLY — a small community is never left without representative articles.
    #
    # Returns the top-N eligible articles by score, at most one per source.
    node_forms = build_form_to_score_community(
        comm, lemma_to_forms, tags_ranking_df, combined_score_col
    )

    news_df = news_df.copy()
    computed = news_df["text"].apply(
        lambda t: compute_community_density_score(t, node_forms)
    )
    news_df["news_score"]       = computed.apply(lambda x: x[0])
    news_df["matched_concepts"] = computed.apply(lambda x: x[1])
    news_df["n_distinctive"]    = computed.apply(lambda x: x[2])

    # Leva B — keep only articles meeting the distinctive-concepts threshold.
    eligible = news_df[news_df["n_distinctive"] >= min_distinctive]

    # Per-community fallback: if the threshold emptied the community,
    # relax it to 1 so the section is never blank.
    if eligible.empty and min_distinctive > 1:
        eligible = news_df[news_df["n_distinctive"] >= 1]

    eligible = eligible.sort_values("news_score", ascending=False)

    # At most one article per source
    selected     = []
    seen_sources = set()
    for _, row in eligible.iterrows():
        source = row.get("source", "")
        if source not in seen_sources:
            selected.append(row)
            seen_sources.add(source)
        if len(selected) == top_n:
            break

    return pd.DataFrame(selected).reset_index(drop=True)


def select_news_covering_all_nodes(comm, news_df, lemma_to_forms, tags_ranking_df,
                                     combined_score_col="combined_score", top_n=15):
    # Selects news for title generation while ensuring community coverage.
    #
    # Stage 1: top_n news by relevance score
    # Stage 2: for each DISTINCTIVE community node not yet covered, add the
    #         news item with the highest score that contains it
    #
    # Coverage targets the DISTINCTIVE nodes only: generic / unclassified
    # nodes (place names, cross-cutting hubs) are not worth fetching a
    # dedicated article for, and doing so would reintroduce the very
    # cross-story leakage this change removes.

    # Stage 1: top N by score
    top_news = score_news_for_community(
        comm, news_df, lemma_to_forms, tags_ranking_df,
        combined_score_col=combined_score_col, top_n=top_n
    )

    # Coverage is tracked over distinctive nodes only.
    distinctive = _distinctive_nodes(comm, tags_ranking_df)
    target_nodes = distinctive if distinctive else set(comm)

    node_forms = {}
    for node in target_nodes:
        for form in lemma_to_forms.get(node, {node}):
            node_forms[form.lower()] = node

    covered_nodes = set()
    for _, row in top_news.iterrows():
        text_lower = " ".join(row.get("text", "").lower().split()[:KEYWORD_MAX_WORDS])
        for form, node in node_forms.items():
            if form in text_lower:
                covered_nodes.add(node)

    uncovered = target_nodes - covered_nodes

    if not uncovered:
        return top_news

    # Stage 2: for each uncovered distinctive node, find the best news item
    # containing it. Single-node scoring uses min_distinctive=1 (a one-node
    # community can only ever match one concept).
    extra_rows = []
    already_selected_idx = set(top_news.index)

    for node in uncovered:
        forms = {f.lower() for f in lemma_to_forms.get(node, {node})}
        candidates = news_df[
            ~news_df.index.isin(already_selected_idx) &
            news_df["text"].str.lower().apply(
                lambda t: any(f in " ".join(t.split()[:KEYWORD_MAX_WORDS]) for f in forms)
            )
        ]
        if candidates.empty:
            continue
        # Score candidates and take the best one
        scored = score_news_for_community(
            {node}, candidates, lemma_to_forms, tags_ranking_df,
            combined_score_col=combined_score_col, top_n=1,
            min_distinctive=1,
        )
        if not scored.empty:
            extra_rows.append(scored.iloc[0])
            already_selected_idx.add(scored.index[0])

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        return pd.concat([top_news, extra_df], ignore_index=True)

    return top_news


# =============================================================================
# Ego-graph utilities (used by both core/bridge concept narration and
# anywhere a per-tag news retrieval is needed)
# =============================================================================

def get_news_for_tag(tag, df, lemma_to_forms, text_col="text"):
    # Retrieves all articles containing at least one surface form
    # of the canonical tag (lemma or entity).
    forms = lemma_to_forms.get(tag, {tag})

    mask = df[text_col].str.lower().apply(
        lambda text: any(form in text for form in forms)
    )
    return df[mask].copy()


def count_neighbour_forms_in_text(text, neighbour_forms_per_node):
    # Counts how many ego-graph neighbours appear (any surface form) in
    # the given text. Each neighbour is counted at most once, regardless
    # of how many surface forms match.
    text_lower = text.lower()
    return sum(
        1 for forms in neighbour_forms_per_node.values()
        if any(form in text_lower for form in forms)
    )


def rank_news_by_ego_coverage(ego_node, ego_G, news_df, lemma_to_forms):
    # Sorts articles by how many ego-graph neighbours they contain, then
    # selects the first 5 with at most one article per source.
    neighbours = [n for n in ego_G.nodes if n != ego_node]

    neighbour_forms = {
        n: lemma_to_forms.get(n, {n})
        for n in neighbours
    }

    news_df = news_df.copy()
    news_df["neighbour_count"] = news_df["text"].apply(
        lambda t: count_neighbour_forms_in_text(t, neighbour_forms)
    )
    news_df = news_df.sort_values("neighbour_count", ascending=False)

    # Select top 5 with at most one article per source
    selected     = []
    seen_sources = set()

    for _, row in news_df.iterrows():
        source = row.get("source", "")
        if source not in seen_sources:
            selected.append(row)
            seen_sources.add(source)
        if len(selected) == 5:
            break

    return pd.DataFrame(selected)


def format_selected_news(top_news_df, max_text_length=300):
    # Formats the selected top articles for display.
    # Includes source, publication time, and original text.
    # Text is truncated to max_text_length characters with [...] if needed.
    lines = []
    for i, (_, row) in enumerate(top_news_df.iterrows(), 1):
        source = row.get("source", "-")
        time   = str(row.get("time", "-"))[:5]
        text   = row.get("text", "").strip()

        if len(text) > max_text_length:
            text = text[:max_text_length].rstrip() + " [...]"

        lines.append(f"{i}. [{source} - {time}]\n{text}")
    return "\n".join(lines)


# =============================================================================
# Community heterogeneity — focused vs broad
# =============================================================================
# A community can be either FOCUSED (all articles cover the same specific
# story) or BROAD (it groups several related sub-stories — e.g. "the day's
# sport": tennis + football + Serie A). Both are legitimate, but the title
# must adapt: a focused community wants a specific title, a broad one wants
# a title that names the shared area without pretending it is a single
# story.
#
# Heterogeneity is measured as the community's mean radius: the average
# cosine distance of its representative-article embeddings from their
# centroid. Compact articles -> small radius -> focused; scattered articles
# -> large radius -> broad. The radius is compared against a single
# threshold to yield the binary label consumed by the title prompt.

# Mean-radius threshold above which a community is labelled "broad".
# The mean radius (average cosine distance of the article embeddings from
# their centroid) lives in a compressed range: focused stories sit near
# 0.0-0.15, while a set of clearly different sub-stories reaches roughly
# 0.35-0.45. 0.30 separates the two regimes. Conservative default — retune
# after inspecting real output (the radius is logged for every community).
COMMUNITY_BROAD_RADIUS_THRESHOLD = 0.30

# Labels passed to the title prompt.
HETEROGENEITY_FOCUSED = "focused"
HETEROGENEITY_BROAD = "broad"


def compute_community_heterogeneity(top_news, embedding_model, text_col="text"):
    # Classifies a community as focused or broad from the dispersion of its
    # representative articles.
    #
    # Parameters:
    #   top_news         the community's representative articles (the
    #                    `top_news` DataFrame already produced upstream)
    #   embedding_model  SentenceTransformer-like model with .encode()
    #   text_col         column holding article text
    #
    # Returns: (label, mean_radius)
    #   label       HETEROGENEITY_FOCUSED or HETEROGENEITY_BROAD
    #   mean_radius the measured mean cosine distance from the centroid
    #
    # With fewer than 2 articles dispersion is undefined; such a community
    # is treated as focused (a single story by construction).
    if top_news is None or len(top_news) < 2:
        return HETEROGENEITY_FOCUSED, 0.0

    texts = [
        t for t in top_news[text_col].tolist()
        if isinstance(t, str) and t.strip()
    ]
    if len(texts) < 2:
        return HETEROGENEITY_FOCUSED, 0.0

    emb = np.asarray(embedding_model.encode(texts, show_progress_bar=False),
                     dtype=float)

    # L2-normalise so the dot product is cosine similarity; cosine distance
    # of each article from the (normalised) centroid is then 1 - cos.
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb_unit = emb / norms

    centroid = emb_unit.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-9)

    cos_to_centroid = emb_unit @ centroid
    mean_radius = float(np.mean(1.0 - cos_to_centroid))

    label = (
        HETEROGENEITY_BROAD
        if mean_radius > COMMUNITY_BROAD_RADIUS_THRESHOLD
        else HETEROGENEITY_FOCUSED
    )
    return label, mean_radius
