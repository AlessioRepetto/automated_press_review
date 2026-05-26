# Cluster ranking - top-N cluster selection.
#
# Selects the most informatively significant HDBSCAN clusters, ordered by
# a composite score that balances:
#
#   - mean_combined_score of the concepts classified in the cluster (quality)
#   - cluster coverage normalised across all clusters (mass)
#
# These top clusters become the input for downstream cross-source analyses
# (coverage asymmetry, lexical framing), which only make sense when there
# are multiple articles from multiple sources.

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.config import ALPHA, MIN_DF_PER_SOURCE, TOP_N_FRAMING
from scripts.models import nlp_it, stop_words_italian


CLASSIFIED_ROLES = {"core", "bridge", "stable", "peripheral"}

# POS classes that carry semantic content for framing purposes.
# - NOUN: thematic concepts the source talks about
# - PROPN: secondary entities not in the global entity set
# - VERB: characterising actions (essential for framing - e.g. "accusare"
#         vs "denunciare" carries different connotation for the same event)
# - ADJ: qualifying adjectives (e.g. "drammatico", "storico", "modesto")
# Excluded: ADP, AUX, CCONJ, DET, INTJ, NUM, PART, PRON, SCONJ, SYM, X
FRAMING_ALLOWED_POS = {"NOUN", "PROPN", "VERB", "ADJ"}


# =============================================================================
# Cluster ranking
# =============================================================================

def map_clusters_to_concepts(themes_df, similarity_by_topic_list, tags_ranking_df):
    # For each HDBSCAN cluster, returns the list of canonical concepts
    # (lemmatised tags and entities) classified into structural roles
    # (core/bridge/stable/peripheral). Generic and unclassified concepts
    # are excluded: a cluster's importance must derive from structurally
    # meaningful concepts, not from cross-cutting hubs.
    role_lookup = tags_ranking_df.set_index("tag")["concept_type"].to_dict()

    cluster_topic_ids = themes_df["Topic"].tolist()

    cluster_concepts = {}
    for topic_id, cooc_string in zip(cluster_topic_ids, similarity_by_topic_list):
        concepts = [
            c for c in cooc_string.split("|")
            if c and role_lookup.get(c) in CLASSIFIED_ROLES
        ]
        cluster_concepts[topic_id] = concepts

    return cluster_concepts


def compute_cluster_metrics(cluster_concepts, data, tags_ranking_df):
    # For each cluster:
    #   mean_combined_score:   average combined_score of classified concepts
    #   n_concepts_classified: number of classified concepts
    #   n_articles:            number of articles in the cluster
    #   n_sources:             number of distinct sources covering it
    score_lookup = tags_ranking_df.set_index("tag")["combined_score"].to_dict()

    cluster_rows = []
    for topic_id, concepts in cluster_concepts.items():
        cluster_articles = data[data["Topic"] == topic_id]
        n_articles = len(cluster_articles)
        n_sources = cluster_articles["source"].nunique()

        if concepts:
            scores = [score_lookup.get(c, 0.0) for c in concepts]
            mean_score = np.mean(scores)
        else:
            mean_score = 0.0

        cluster_rows.append({
            "topic_id":              topic_id,
            "n_articles":            n_articles,
            "n_sources":             n_sources,
            "n_concepts_classified": len(concepts),
            "mean_combined_score":   mean_score,
            "concepts":              concepts,
        })

    return pd.DataFrame(cluster_rows)


def compute_cluster_composite_score(clusters_df, alpha=ALPHA):
    # Adds coverage_norm (percentile rank of n_concepts_classified) and
    # cluster_score (alpha-weighted blend of quality and mass).
    #
    # alpha = 1.0 -> pure quality (favors small clusters with few high-scoring concepts)
    # alpha = 0.0 -> pure mass (favors large clusters regardless of quality)
    # alpha = 0.6 -> quality-skewed balance (default)
    clusters_df = clusters_df.copy()
    clusters_df["coverage_norm"] = clusters_df["n_concepts_classified"].rank(pct=True)

    clusters_df["cluster_score"] = (
        alpha * clusters_df["mean_combined_score"] +
        (1 - alpha) * clusters_df["coverage_norm"]
    )
    return clusters_df.sort_values("cluster_score", ascending=False).reset_index(drop=True)


def select_eligible_top_clusters(clusters_df, min_articles, min_sources, top_n=5):
    # Filters by cross-source analyzability prerequisites, then takes top-N.
    # Filters are applied AFTER scoring, so the score reflects the
    # intrinsic informational value, not its analyzability.
    eligible = clusters_df[
        (clusters_df["n_articles"] >= min_articles) &
        (clusters_df["n_sources"]  >= min_sources)
    ].copy()
    return eligible, eligible.head(top_n).reset_index(drop=True)


# =============================================================================
# Lexical framing per cluster (cross-source TF-IDF)
# =============================================================================

def lemmatise_for_framing(text, protected_lower=None):
    # Tokenises text via spaCy and returns content lemmas (NOUN/ADJ/VERB),
    # at least 4 characters, no stopwords.
    #
    # CRITICAL: tokens whose lowercase form is in `protected_lower` are
    # preserved as their lowercase surface form (no lemmatisation), so
    # proper nouns are not damaged by Italian morphology.
    if not isinstance(text, str) or not text.strip():
        return ""

    if protected_lower is None:
        protected_lower = set()

    doc = nlp_it(text.lower())
    out = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        if tok.text in protected_lower:
            out.append(tok.text)
            continue
        if (tok.pos_ in {"NOUN", "ADJ", "VERB"}
                and len(tok.lemma_) >= 4
                and tok.lemma_ not in stop_words_italian):
            out.append(tok.lemma_)
    return " ".join(out)


def build_lemma_to_forms_per_source(cluster_articles, sources, protected_lower):
    # For each source, builds a mapping  lemma -> Counter(surface_form -> count)
    # over the lemmatised articles of that source.
    #
    # The Counter lets us pick, downstream, the surface form actually used
    # most often by THAT source for each lemma. Without this we would
    # display lemmas like "decisione" or "elezione" even when the source
    # always wrote "decisioni" or "elezioni" — distorted forms that the
    # source itself never uses.
    #
    # Mechanically: we run spaCy on the source's concatenated text, take
    # all content tokens (same POS/length/stopword filter as
    # lemmatise_for_framing), and for each one record both
    # (lemma, surface_form). Surface forms of protected entities are
    # identical to the lemma (they were not lemmatised), so the mapping
    # still works for them without special-casing.
    per_source = {s: defaultdict(Counter) for s in sources}

    for source in sources:
        texts = cluster_articles.loc[cluster_articles["source"] == source, "text"]
        joined = " ".join(t for t in texts if isinstance(t, str))
        if not joined.strip():
            continue
        doc = nlp_it(joined.lower())
        for tok in doc:
            if not tok.is_alpha:
                continue
            surface = tok.text
            if surface in protected_lower:
                # Protected: lemma == surface (we didn't lemmatise)
                lemma = surface
            else:
                if not (tok.pos_ in {"NOUN", "ADJ", "VERB"}
                        and len(tok.lemma_) >= 4
                        and tok.lemma_ not in stop_words_italian):
                    continue
                lemma = tok.lemma_
            per_source[source][lemma][surface] += 1

    return per_source


def pick_surface_form(lemma, source, per_source_lemma_to_forms):
    # Picks the surface form most frequently used by `source` for `lemma`.
    # Ties are broken alphabetically (Counter.most_common is stable, and
    # alphabetical fallback keeps day-to-day output reproducible).
    # If for any reason the lemma is unknown to that source's counter
    # (should not happen because TF-IDF terms come from the same texts),
    # we return the lemma itself as a safe fallback.
    forms = per_source_lemma_to_forms.get(source, {}).get(lemma)
    if not forms:
        return lemma
    # most_common with sorted keys as tiebreaker for determinism
    best_count = max(forms.values())
    candidates = sorted(f for f, c in forms.items() if c == best_count)
    return candidates[0]


def is_framing_term_relevant(term):
    # Returns True when the surface form is morphologically a content word
    # (NOUN / PROPN / VERB / ADJ) under spaCy isolated parsing, False otherwise.
    # Used to drop high-TF-IDF but low-information terms from the per-source
    # distinctive-terms list - typically adverbs, auxiliaries, or determiners
    # that survived the upstream lemmatisation filter because they were
    # introduced (or kept) by the surface-form substitution step.
    if not isinstance(term, str) or not term.strip():
        return False
    doc = nlp_it(term)
    if not len(doc):
        return False
    return doc[0].pos_ in FRAMING_ALLOWED_POS


def extract_framing_per_cluster(cluster_articles, entities_set, top_n=TOP_N_FRAMING):
    # Each source is treated as a document. TF-IDF identifies the terms
    # that characterise one source compared with the others within the
    # same cluster. Returns a long-form DataFrame (source, term, tfidf_score)
    # where `term` is the actual surface form most used by THAT source.
    #
    # The pipeline:
    #   1. Concatenate per source, lemmatise (entities preserved)
    #   2. TF-IDF across sources -> distinctive lemmas per source
    #   3. Filter out lemmas that are also in entities_set (case-insensitive)
    #   4. For each distinctive lemma, substitute back the surface form
    #      most frequently used by THAT source, so the displayed terms
    #      reflect the source's actual lexical choices

    entities_lower = {e.lower() for e in entities_set}
    sources = sorted(cluster_articles["source"].dropna().unique().tolist())

    # Step 1 - source documents (lemmatised, entities preserved)
    source_docs = (
        cluster_articles
        .groupby("source")["text"]
        .apply(lambda texts: lemmatise_for_framing(
            " ".join(texts), protected_lower=entities_lower
        ))
    )
    source_docs = source_docs[source_docs.str.len() > 0]

    if len(source_docs) < 2:
        return pd.DataFrame(columns=["source", "term", "tfidf_score"])

    # Step 2 - TF-IDF across source documents
    vec = TfidfVectorizer(min_df=MIN_DF_PER_SOURCE, sublinear_tf=True)
    try:
        tfidf = vec.fit_transform(source_docs.values)
    except ValueError:
        return pd.DataFrame(columns=["source", "term", "tfidf_score"])

    terms = vec.get_feature_names_out()

    # Step 4 prerequisite - precompute lemma -> surface-form counter per source
    per_source_forms = build_lemma_to_forms_per_source(
        cluster_articles, source_docs.index.tolist(), entities_lower
    )

    # Step 3 + 4 - per source, take top-N TF-IDF lemmas (entity filter
    # case-insensitive), then substitute back the most-used surface form,
    # then apply a POS relevance filter to drop function words that
    # accidentally rank high in TF-IDF.
    #
    # We oversample the initial pool (3 * top_n) so that the POS filter
    # has room to discard candidates without dropping below top_n for the
    # source. If after filtering fewer than top_n remain (e.g. very thin
    # source corpus), we keep what we have rather than padding with noise.
    POOL_MULTIPLIER = 3
    rows = []
    for i, source in enumerate(source_docs.index):
        scores = tfidf[i].toarray().ravel()
        candidates = [
            (terms[j], scores[j]) for j in scores.argsort()[::-1]
            if scores[j] > 0 and terms[j] not in entities_lower
        ][:top_n * POOL_MULTIPLIER]

        kept = 0
        for lemma, score in candidates:
            if kept >= top_n:
                break
            surface = pick_surface_form(lemma, source, per_source_forms)
            if not is_framing_term_relevant(surface):
                continue
            rows.append({
                "source":      source,
                "term":        surface,
                "tfidf_score": float(score),
            })
            kept += 1

    return pd.DataFrame(rows)
