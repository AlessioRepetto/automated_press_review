# Text preprocessing and named-entity classification helper.
#
# - `preprocess_text`: Italian tokenization, stopword removal,
#   alphabetic filter.
# - `classify_keyword_type`: decides whether a candidate keyword is a
#   named entity (and with which spaCy label) or a generic tag. Strategy
#   scales from exact matches on entity spans, to PROPN token-level
#   matches inside spans, and finally to isolated parsing for monolithic
#   candidates.

import pandas as pd
from nltk.tokenize import word_tokenize

from scripts.models import stop_words_italian


def preprocess_text(text):
    # Tokenizes Italian text and keeps only alphabetic non-stopword tokens.
    if pd.isna(text):
        return []

    text = str(text).lower()
    tokens = word_tokenize(text, language="italian")
    return [
        w for w in tokens
        if w not in stop_words_italian and w.isalpha()
    ]


def classify_keyword_type(keyword, doc_spacy, nlp):
    # Checks whether 'keyword' is a named entity or a proper noun inside doc_spacy.
    # The comparison is done in lowercase to be case-insensitive,
    # regardless of how the text was originally capitalized.
    #
    # Strategy in priority order:
    #   1. Exact match with a recognized entity span
    #   2. PROPN token-level match inside an entity span
    #   3. Fallback: isolated parsing of the keyword, returns PROPN if it is a single token
    #
    # Returns a tuple (is_entity, label) where label belongs to
    # {PER, PERSON, ORG, LOC, GPE, MISC, PROPN, OTHER}
    kw_norm = keyword.lower().strip()

    # 1) Exact match with an entity span
    for ent in doc_spacy.ents:
        if kw_norm == ent.text.lower().strip() and ent.label_ in ("PER", "PERSON", "ORG", "LOC", "GPE", "MISC"):
            return True, ent.label_

    # 2) PROPN token-level match inside an entity span
    for ent in doc_spacy.ents:
        if ent.label_ not in ("PER", "PERSON", "ORG", "LOC", "GPE", "MISC"):
            continue
        for token in ent:
            if token.text.lower().strip() == kw_norm and token.pos_ == "PROPN":
                return True, ent.label_

    # 3) Isolated parsing of the keyword (single-token case)
    kw_doc = nlp(keyword)
    if len(kw_doc) == 1 and kw_doc[0].pos_ == "PROPN":
        return True, "PROPN"

    return False, "OTHER"


def merge_entity_lists(list1, list2):
    # Concatenates two pipe-separated entity strings, handling empty cases
    if not list1 and not list2:
        return ''
    if not list1:
        return list2
    if not list2:
        return list1
    return list1 + '|' + list2


def clean_pipe_edges(s):
    # Removes any leading or trailing pipes from a string
    return s.strip('|')
