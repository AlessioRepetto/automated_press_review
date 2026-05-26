"""Conservative pre-merge of near-duplicate HDBSCAN clusters.

HDBSCAN (like the BERTopic-style pipeline it feeds) routinely produces
several clusters that are, in substance, variants of the same specific
story. The KeyBERT documentation notes the same phenomenon for keyword
clusters. Left untouched, these near-duplicates fragment the cross-source
coverage analysis of section 24: the same story is measured several times,
each time on a thinner slice of articles.

This module performs a single, deliberately cautious merging pass run
*immediately before* section 24. It does NOT touch any global pipeline
structure: it operates on local copies of `data`, `themes_df` and
`similarity_by_topic_list`, and only those copies are handed to section 24.
Everything produced by sections 4-23 (graph, structural metrics, role
classification, Louvain communities) is left exactly as it was.

Design — three cautious choices, all confirmed with the project owner:

1. Two independent signals, combined with AND. A pair of clusters is merged
   only if BOTH hold:
     - centroid cosine similarity >= MERGE_CENTROID_THRESHOLD
       (centroid = mean SentenceTransformer embedding of the cluster's news)
     - Jaccard overlap of their concept sets >= MERGE_JACCARD_THRESHOLD
       (concept set = the pipe-separated tags of similarity_by_topic_list,
       generics included — the wider set, per the owner's choice)
   Two independent tests make a false positive far less likely than either
   signal alone: linguistic proximity without shared concepts, or shared
   generic concepts without semantic proximity, are both rejected.

2. Conservative thresholds. Defaults are high; only near-certain duplicates
   merge. The thresholds are module-level constants so they can be retuned
   in one place after inspecting real output.

3. Pairs only, no transitivity. Candidate pairs are ranked by combined
   strength and merged greedily, but each original cluster may take part in
   at most ONE merge per run. If A merges with B, a later B~C pair is
   skipped. This guarantees every merged cluster is the union of at most two
   original clusters — no chain drift into a confused mega-cluster.

Every merge is logged with both similarity figures, so the operator can see
exactly what was merged and retune the thresholds if needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# Tunable parameters — deliberately conservative                                #
# -----------------------------------------------------------------------------#
# Master switch: set to False to disable merging entirely.
ENABLE_CLUSTER_MERGE = True

# Centroid cosine similarity above which two clusters are "semantically
# the same". 0.92 is high: only near-duplicate stories pass.
MERGE_CENTROID_THRESHOLD = 0.92

# Jaccard overlap of concept sets above which two clusters "talk about the
# same things". 0.50 means at least half the combined concept vocabulary is
# shared.
MERGE_JACCARD_THRESHOLD = 0.50


@dataclass
class MergeResult:
    """Local, merged copies handed to section 24. Globals are untouched."""

    data: pd.DataFrame                       # copy of `data`, Topic relabelled
    themes_df: pd.DataFrame                  # copy, merged rows collapsed
    similarity_by_topic_list: list[str]      # copy, merged entries combined
    merges: list[tuple[int, int, float, float]]  # (kept, absorbed, cos, jac)


# -----------------------------------------------------------------------------#
# Helpers                                                                       #
# -----------------------------------------------------------------------------#
def _cluster_centroids(
    data: pd.DataFrame,
    embeddings: np.ndarray,
    topic_ids: list[int],
) -> dict[int, np.ndarray]:
    """Mean embedding per cluster.

    `embeddings` is row-aligned with `data` as it was at clustering time,
    so data.index is used positionally. Noise (-1) is never passed in.
    """
    centroids: dict[int, np.ndarray] = {}
    topic_array = data["Topic"].to_numpy()
    for topic_id in topic_ids:
        mask = topic_array == topic_id
        if mask.sum() == 0:
            continue
        centroids[topic_id] = embeddings[mask].mean(axis=0)
    return centroids


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _concept_set(cooc_string: str) -> set[str]:
    """Concept set of a cluster from its similarity_by_topic_list entry.

    Per the owner's choice this is the WIDE set: every pipe-separated tag,
    generics included — not only the structurally classified concepts.
    """
    if not isinstance(cooc_string, str):
        return set()
    return {c for c in cooc_string.split("|") if c}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard index of two sets. 0.0 when both are empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# -----------------------------------------------------------------------------#
# Public entry point                                                            #
# -----------------------------------------------------------------------------#
def merge_similar_clusters(
    data: pd.DataFrame,
    themes_df: pd.DataFrame,
    similarity_by_topic_list: list[str],
    embeddings: np.ndarray,
) -> MergeResult:
    """Merge near-duplicate clusters on local copies, for section 24 only.

    Args:
        data: the working article DataFrame (must have a 'Topic' column).
        themes_df: per-cluster themes; must have a 'Topic' column and be
            row-aligned with `similarity_by_topic_list`.
        similarity_by_topic_list: per-cluster pipe-joined concept strings,
            aligned by position with themes_df rows.
        embeddings: SentenceTransformer embeddings, row-aligned with `data`.

    Returns:
        MergeResult with local merged copies of the three structures and the
        list of merges performed. The inputs themselves are NOT modified.
    """
    # Work on copies — globals must stay untouched.
    data = data.copy()
    themes_df = themes_df.reset_index(drop=True).copy()
    similarity_by_topic_list = list(similarity_by_topic_list)

    # Guard: embeddings must be row-aligned with `data`. They are consumed
    # positionally against data's Topic column, so a length mismatch would
    # otherwise surface as a cryptic IndexError deep inside _cluster_centroids.
    if len(embeddings) != len(data):
        raise ValueError(
            f"embeddings rows ({len(embeddings)}) must match data rows "
            f"({len(data)}). The embeddings must be recomputed on the SAME "
            f"`data` passed here — an earlier embedding array from before the "
            f"section 9-12 merges is no longer aligned."
        )

    if not ENABLE_CLUSTER_MERGE:
        logger.info("Cluster merge disabled (ENABLE_CLUSTER_MERGE=False).")
        return MergeResult(data, themes_df, similarity_by_topic_list, [])

    # themes_df rows and similarity_by_topic_list are aligned by position.
    # A cluster's topic_id is themes_df['Topic']; map it to its row index.
    topic_ids: list[int] = themes_df["Topic"].tolist()
    if len(topic_ids) != len(similarity_by_topic_list):
        logger.warning(
            "themes_df rows (%d) and similarity_by_topic_list (%d) are not "
            "aligned; skipping cluster merge.",
            len(topic_ids), len(similarity_by_topic_list),
        )
        return MergeResult(data, themes_df, similarity_by_topic_list, [])

    row_of_topic = {tid: i for i, tid in enumerate(topic_ids)}

    # Centroids (only for clusters present in themes_df; noise is not here).
    centroids = _cluster_centroids(data, embeddings, topic_ids)

    # Concept sets, indexed by topic_id.
    concept_sets = {
        tid: _concept_set(similarity_by_topic_list[row_of_topic[tid]])
        for tid in topic_ids
    }

    # -- 1. Score every unordered pair --------------------------------------
    candidates: list[tuple[float, float, float, int, int]] = []
    for i in range(len(topic_ids)):
        for j in range(i + 1, len(topic_ids)):
            t_i, t_j = topic_ids[i], topic_ids[j]
            if t_i not in centroids or t_j not in centroids:
                continue

            cos = _cosine(centroids[t_i], centroids[t_j])
            if cos < MERGE_CENTROID_THRESHOLD:
                continue

            jac = _jaccard(concept_sets[t_i], concept_sets[t_j])
            if jac < MERGE_JACCARD_THRESHOLD:
                continue

            # Both tests passed — record. Strength = sum of the two signals.
            candidates.append((cos + jac, cos, jac, t_i, t_j))

    if not candidates:
        logger.info("Cluster merge: no near-duplicate pairs found.")
        return MergeResult(data, themes_df, similarity_by_topic_list, [])

    # -- 2. Greedy pairing, each cluster used at most once ------------------
    candidates.sort(reverse=True)  # strongest pairs first
    used: set[int] = set()
    merges: list[tuple[int, int, float, float]] = []

    for _, cos, jac, t_i, t_j in candidates:
        if t_i in used or t_j in used:
            continue
        # Keep the lower topic_id, absorb the higher — deterministic.
        kept, absorbed = sorted((t_i, t_j))
        merges.append((kept, absorbed, cos, jac))
        used.add(t_i)
        used.add(t_j)
        logger.info(
            "Merged cluster %d into cluster %d (centroid sim=%.3f, jaccard=%.3f)",
            absorbed, kept, cos, jac,
        )

    # -- 3. Apply merges to the local copies --------------------------------
    for kept, absorbed, _, _ in merges:
        # 3a. Relabel articles of the absorbed cluster.
        data.loc[data["Topic"] == absorbed, "Topic"] = kept

        # 3b. Merge the themes_df row + similarity_by_topic_list entry.
        row_kept = row_of_topic[kept]
        row_absorbed = row_of_topic[absorbed]

        merged_concepts = (
            concept_sets[kept] | concept_sets[absorbed]
        )
        similarity_by_topic_list[row_kept] = "|".join(sorted(merged_concepts))

        # Merge the themes_df 'Tags' / 'Entities' / 'News' text columns by
        # concatenation, so the kept row reflects both clusters.
        for col in ("Tags", "Entities", "News"):
            if col in themes_df.columns:
                a = str(themes_df.at[row_kept, col] or "")
                b = str(themes_df.at[row_absorbed, col] or "")
                sep = "|" if col in ("Tags", "Entities") else "\n\n"
                themes_df.at[row_kept, col] = (a + sep + b).strip(sep)

    # 3c. Drop the absorbed rows from themes_df + similarity list.
    absorbed_rows = sorted(
        (row_of_topic[absorbed] for _, absorbed, _, _ in merges),
        reverse=True,
    )
    for r in absorbed_rows:
        del similarity_by_topic_list[r]
    absorbed_topic_ids = {absorbed for _, absorbed, _, _ in merges}
    themes_df = themes_df[
        ~themes_df["Topic"].isin(absorbed_topic_ids)
    ].reset_index(drop=True)

    logger.info(
        "Cluster merge complete: %d pair(s) merged, %d clusters -> %d.",
        len(merges), len(topic_ids), len(topic_ids) - len(merges),
    )

    return MergeResult(data, themes_df, similarity_by_topic_list, merges)
