# Custom KeyBERT-style keyword extraction with Maximal Marginal Relevance.
#
# The MMR variant used here is more aggressive than the canonical one:
# candidates whose cosine similarity with an already selected keyword
# exceeds `max_similarity_between_keywords` are filtered before scoring.
#
# Three levels:
# - `mmr`: diversity-aware selection
# - `extract_keywords_single_doc`: embedding-based extraction on a single document
# - `extract_keywords_single_doc_with_types`: splits the result into named entities
#   and generic tags through NER
# - `extract_keywords_corpus_with_types`: corpus-level wrapper

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.preprocessing import classify_keyword_type


def mmr(doc_embedding, candidate_embeddings, candidates,
        top_n=5, diversity=0.7, max_similarity_between_keywords=0.8):
    # Maximal Marginal Relevance selection.
    #
    # Balances relevance to the document and redundancy among selected keywords.
    # Keywords that are too similar to an already selected one are excluded upstream
    # through max_similarity_between_keywords.
    #
    # Main parameters:
    #   doc_embedding                   array (1, dim)
    #   candidate_embeddings            array (n_cand, dim)
    #   candidates                      list of candidate strings
    #   top_n                           number of keywords to return
    #   diversity                       weight of the diversity term vs relevance, in [0,1]
    #   max_similarity_between_keywords redundancy threshold for discarding candidates
    #
    # Returns a tuple (selected_keywords, selected_scores).
    sim_doc     = cosine_similarity(candidate_embeddings, doc_embedding)
    sim_between = cosine_similarity(candidate_embeddings)

    selected_idx  = []
    candidate_idx = list(range(len(candidates)))

    first = int(np.argmax(sim_doc))
    selected_idx.append(first)
    candidate_idx.remove(first)

    for _ in range(top_n - 1):
        best_idx   = None
        best_score = -np.inf

        for i in candidate_idx:
            sim_to_selected = max(sim_between[i][selected_idx])
            if sim_to_selected >= max_similarity_between_keywords:
                continue

            relevance  = float(sim_doc[i].item())
            redundancy = sim_to_selected
            score      = diversity * relevance - (1 - diversity) * redundancy

            if score > best_score:
                best_score = score
                best_idx   = i

        if best_idx is None:
            break

        selected_idx.append(best_idx)
        candidate_idx.remove(best_idx)

    return (
        [candidates[i] for i in selected_idx],
        [float(sim_doc[i].item()) for i in selected_idx],
    )


def extract_keywords_single_doc(doc, model, ngram_range=(1, 3),
                                 top_n=5, stop_words=None,
                                 use_mmr=True, diversity=0.7):
    # Extracts the top-N keywords from a single document through embedding similarity
    # in KeyBERT style.
    # The document is lowercased at input, so casing is
    # handled in one place and all output keywords are guaranteed to be lowercase.
    # Returns a list of tuples (keyword, relevance score).
    doc = doc.lower()
    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit([doc])
    except ValueError:
        # Empty document or document made only of stopwords after preprocessing
        return []
    candidates = vectorizer.get_feature_names_out()
    if len(candidates) == 0:
        return []

    doc_embedding        = model.encode([doc])
    candidate_embeddings = model.encode(list(candidates))
    similarities         = cosine_similarity(candidate_embeddings, doc_embedding).reshape(-1)

    if use_mmr:
        selected_kw, selected_scores = mmr(
            doc_embedding,
            candidate_embeddings,
            candidates,
            top_n=min(top_n, len(candidates)),
            diversity=diversity,
        )
        # Lowercase at the source because CountVectorizer can preserve casing
        return [(kw.lower().strip(), score) for kw, score in zip(selected_kw, selected_scores)]
    else:
        top_n = min(top_n, len(candidates))
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return [(candidates[i].lower().strip(), float(similarities[i])) for i in top_indices]


def extract_keywords_single_doc_with_types(doc, model, ngram_range=(1, 3),
                                            top_n=5, stop_words=None,
                                            use_mmr=True, diversity=0.7, nlp=None):
    # Wrapper around extract_keywords_single_doc that classifies each keyword
    # as a named entity or generic keyword using spaCy NER.
    # All keywords and entities are lowercased before being returned.
    #
    # Returns a dict:
    #   {
    #       "entities":       [{"keyword": str, "score": float, "label": str}, ...],
    #       "other_keywords": [{"keyword": str, "score": float}, ...]
    #   }
    if nlp is None:
        raise ValueError("A spaCy nlp object must be provided (e.g. nlp=nlp_it).")

    # NER requires the document with original casing, so spaCy
    # can recognize proper nouns. Keyword extraction instead uses the
    # lowercase copy. extract_keywords_single_doc applies lowercase internally,
    # but passing doc_lower already makes the intention explicit.
    doc_spacy = nlp(doc)
    doc_lower = doc.lower()
    kw_list   = extract_keywords_single_doc(doc_lower, model, ngram_range, top_n, stop_words, use_mmr, diversity)

    entities = []
    others   = []

    for kw, score in kw_list:
        is_entity, label = classify_keyword_type(kw, doc_spacy, nlp)
        if is_entity:
            entities.append({"keyword": kw, "score": score, "label": label})
        else:
            others.append({"keyword": kw, "score": score})

    return {"entities": entities, "other_keywords": others}


def extract_keywords_corpus_with_types(docs, model, ngram_range=(1, 3),
                                        top_n=5, stop_words=None,
                                        use_mmr=True, diversity=0.7, nlp=None):
    # Applies extract_keywords_single_doc_with_types to a list of documents.
    # Returns a list of dicts {"text": str, "entities": [...], "other_keywords": [...]}.
    if nlp is None:
        raise ValueError("A spaCy nlp object must be provided.")

    return [
        {
            "text": doc,
            **extract_keywords_single_doc_with_types(
                doc, model, ngram_range, top_n, stop_words, use_mmr, diversity, nlp
            ),
        }
        for doc in docs
    ]
