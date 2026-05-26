# Co-occurrence matrix and graph construction.
#
# The two lists of co-occurrence strings (cluster + document) are
# concatenated and passed to cooccurrence_matrix_pipe, which uses binary
# term presence per string (a term that co-occurs multiple times in the
# same string is counted only once) and computes X^T @ X.
#
# Adaptive edge-weight threshold:
# The threshold below which edges are pruned is derived from the
# cumulative distribution of weights. We keep edges whose weights
# contribute at least CUMULATIVE_THRESHOLD of the total weight, with a
# hard minimum of 2 to require at least one independent confirmation.

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def cooccurrence_matrix_pipe(strings, min_df=1, return_vocab=True):
    # Builds a sparse co-occurrence matrix from a list of pipe-separated strings.
    # Uses binary term presence per string (a term that appears multiple times
    # in the same string counts only once), then computes X^T @ X.
    #
    # Parameters:
    #   strings       list of pipe-separated strings
    #   min_df        minimum document frequency to include a term
    #   return_vocab  if True, also returns the vocabulary array
    #
    # Returns (C, vocab) if return_vocab is True; otherwise only C.
    # C is a sparse CSR matrix of shape (n_terms, n_terms).
    vec = CountVectorizer(
        tokenizer=lambda s: s.split("|"),
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        binary=True,
    )
    X = vec.fit_transform(strings)
    C = (X.T @ X).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()

    return (C, vec.get_feature_names_out()) if return_vocab else C


def graph_from_cooc(C, vocab, directed=False, remove_isolates=True):
    # Converts a sparse co-occurrence matrix into a labeled NetworkX graph.
    # Nodes are labeled with vocabulary strings,
    # edge weights reflect co-occurrence counts.
    G = nx.from_scipy_sparse_array(
        C, create_using=nx.Graph() if not directed else nx.DiGraph()
    )
    G = nx.relabel_nodes(G, {i: vocab[i] for i in range(len(vocab))})
    if remove_isolates:
        G.remove_nodes_from(list(nx.isolates(G)))
    return G


def adaptive_threshold(words_matrix, cumulative_threshold):
    # Derives the edge-weight pruning threshold from the cumulative
    # distribution of weights. Returns the integer threshold value to apply.
    values, counts = np.unique(words_matrix.data, return_counts=True)

    weight_by_frequency        = values * counts
    cumulative_perc_importance = np.cumsum(weight_by_frequency / weight_by_frequency.sum())

    i = 0
    while cumulative_perc_importance[i] < cumulative_threshold:
        i += 1

    return max(int(values[i].item()) - 1, 2)


def prune_matrix_inplace(words_matrix, threshold_val):
    # Zero-out edges below the threshold and compact the sparse matrix.
    mask = words_matrix.data < threshold_val
    words_matrix.data[mask] = 0
    words_matrix.eliminate_zeros()
    return words_matrix
