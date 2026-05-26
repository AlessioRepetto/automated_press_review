# Semantic clustering (UMAP + HDBSCAN) and cluster representation.
#
# Articles are embedded with a multilingual sentence transformer, projected
# to 5 dimensions through UMAP (cosine metric, min_dist=0 to favor compact
# clusters), and clustered with HDBSCAN using cluster_selection_method='eom'.
# Articles assigned to cluster -1 are treated as noise.
#
# Cluster representation: each HDBSCAN cluster is summarized by its top
# TF-IDF terms (lower IDF = more cluster-specific). These terms are split
# into named entities and generic tags through the NER helper, mirroring
# the structure of the KeyBERT outputs.

import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

from scripts.preprocessing import classify_keyword_type


def compute_embeddings(texts, embedding_model):
    # Encodes a list of texts with the sentence transformer.
    return embedding_model.encode(texts, show_progress_bar=True)


def reduce_dimensions(embeddings, n_neighbors=15, n_components=5,
                     min_dist=0.0, metric="cosine", random_state=0):
    # UMAP dimensionality reduction. Defaults chosen to favor compact
    # clusters downstream (min_dist=0, cosine metric).
    return UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    ).fit_transform(embeddings)


def cluster_articles(umap_embeddings, min_cluster_size=2,
                     metric="euclidean", cluster_selection_method="eom"):
    # HDBSCAN clustering on UMAP-reduced embeddings. Returns the label array.
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    return model.fit_predict(umap_embeddings)


def build_cluster_representation(data, labels, nlp_it):
    # Builds the per-cluster TF-IDF representation, then splits the top
    # terms into named entities and generic tags through NER.
    #
    # Returns the topics_dict (with the noise-cluster placeholder).
    topics_dict = {
        "Topic":    [-1],
        "Name":     ["Misc"],
        "Entities": ["Misc"],
    }

    for label in sorted(set(labels)):
        if label == -1:
            continue

        docs     = data[data["Topic"] == label]["clean text"].tolist()
        docs_raw = data[data["Topic"] == label]["text"].tolist()
        if not docs:
            continue

        # TF-IDF: lower IDF score = more specific to this cluster.
        # Fit on already lowercased documents so all extracted terms are lowercase.
        docs_lower = [d.lower() for d in docs]
        vec        = TfidfVectorizer().fit(docs_lower)
        terms      = vec.get_feature_names_out()
        top_terms  = terms[vec.idf_.argsort()][:5].tolist()  # already lowercase from the fit

        # Parse the concatenated raw text only once for NER
        cluster_text = " ".join(docs)
        doc_spacy    = nlp_it(cluster_text)

        entity_terms  = []
        generic_terms = []
        for term in top_terms:
            is_entity, _ = classify_keyword_type(term, doc_spacy, nlp_it)
            if is_entity:
                entity_terms.append(term)
            else:
                generic_terms.append(term)

        topics_dict["Topic"].append(label)
        topics_dict["Name"].append("|".join(generic_terms) if generic_terms else "")
        topics_dict["Entities"].append("|".join(entity_terms) if entity_terms else "")

    return topics_dict


def build_news_groups(data, labels):
    # Builds per-cluster concatenated documents (excluding noise).
    # Returns a dict with keys Topic, News.
    news_groups = {"Topic": [], "News": []}
    for label in sorted(set(labels)):
        if label >= 0:
            docs = data[data["Topic"] == label]["clean text"].tolist()
            news_groups["Topic"].append(label)
            news_groups["News"].append("\n\n".join(docs))
    return news_groups


def merge_entity_lists(list1, list2):
    # Concatenates two pipe-separated entity strings, handling empty cases.
    if not list1 and not list2:
        return ""
    if not list1:
        return list2
    if not list2:
        return list1
    return list1 + "|" + list2


def clean_pipe_edges(s):
    # Removes any leading or trailing pipes from a string.
    return s.strip("|")


def build_themes_dataframe(topics_dict, news_groups):
    # Consolidates cluster-level entities and tags from the TF-IDF and
    # KeyBERT signals into a single themes_df with columns
    # Topic, Tags, Entities, News.
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
    return themes_df
