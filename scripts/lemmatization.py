# Lemmatization.
#
# Asymmetric lemmatization: only generic tags are lemmatized, while named
# entities remain in their original form to preserve their identity (e.g.
# "Meloni" must not become "melone").
#
# The module exposes:
# - `aggregate_tags_and_entities`: corpus-level aggregation with inclusivity
#   sweep and POS filter.
# - `LemmaResolver`: encapsulates raw_lemmas, lemma_to_forms, tags_set,
#   entities_set, and the co-occurrence string normaliser. Stateful by
#   design: the lemma map is built once per daily corpus, and the
#   normaliser is called many times against it.

import itertools
from collections import defaultdict

from scripts.config import BLOCKLIST
from scripts.models import nlp_it


def aggregate_tags_and_entities(themes_df, data):
    # Aggregates cluster-level and document-level signals into two global
    # corpus sets: tags_list (generic concepts) and entities_list (named
    # entities). Mutual exclusivity is enforced multiple times: each
    # transformation (merge, lowercase, POS filtering) may reintroduce
    # overlap.
    #
    # POS filter:
    #   - tags keep {NOUN, PROPN, ADJ, VERB}
    #   - entities keep {PROPN, NOUN}
    # VERBs are included among tags because, in Italian headlines, verbs
    # can be semantically dense (e.g. arrestare, vincere, dimettere).
    #
    # IMPORTANT - entity inclusivity rule.
    # A token is treated as an entity if it shows up as an entity ANYWHERE in
    # the corpus: in any cluster's Entities column, in any document's
    # Entities column, or as a proper noun under spaCy's isolated parsing.
    # This is intentionally aggressive. KeyBERT classifies tokens per-cluster,
    # and NER on isolated cluster contexts can fail. If a token is EVER
    # recognised as an entity by any signal in the corpus, treat it as an
    # entity everywhere. This is safer than letting proper nouns leak into
    # the lemmatised tag stream.
    #
    # Returns (tags_list, entities_list).

    tags_from_clusters  = itertools.chain.from_iterable(
        themes_df["Tags"].apply(lambda t: t.split("|")).tolist()
    )
    tags_from_documents = itertools.chain.from_iterable(
        data["Keywords"].apply(lambda t: t.split("|")).tolist()
    )
    tags_list = list(set(list(tags_from_clusters) + list(tags_from_documents)))

    entities_from_clusters  = itertools.chain.from_iterable(
        themes_df["Entities"].apply(lambda t: t.replace(" ", "|").split("|")).tolist()
    )
    entities_from_documents = itertools.chain.from_iterable(
        data["Entities"].apply(lambda t: t.split("|")).tolist()
    )
    entities_list = list(set(list(entities_from_clusters) + list(entities_from_documents)))

    # First mutual-exclusivity pass
    tags_list = [t for t in tags_list if t not in set(entities_list)]

    # Safety net: lowercase, strip, and deduplication.
    # All tokens should already be lowercase from upstream stages,
    # but this guarantees correctness even if the pipeline is partially rerun.
    tags_list     = list(set(t.lower().strip() for t in tags_list     if t.strip()))
    entities_list = list(set(e.lower().strip() for e in entities_list if e.strip()))

    # Reapply mutual exclusivity after lowercasing, which may have introduced new overlaps
    entities_set_temp = set(entities_list)
    tags_list = [t for t in tags_list if t not in entities_set_temp]

    # POS filter.
    # Keep only the informatively relevant parts of speech.
    # VERB remains among the tags because some verbs (arrestare, vincere, dimettere)
    # are semantically dense in Italian headlines. Auxiliary or function verbs
    # are handled downstream by the BLOCKLIST.
    # Use nlp_it.pipe() for performance rather than nlp_it() per token.

    ALLOWED_POS_TAGS     = {"NOUN", "PROPN", "ADJ", "VERB"}
    ALLOWED_POS_ENTITIES = {"PROPN", "NOUN"}

    tags_list = [
        t for t, doc in zip(tags_list, nlp_it.pipe(tags_list))
        if doc[0].pos_ in ALLOWED_POS_TAGS
    ]

    entities_list = [
        e for e, doc in zip(entities_list, nlp_it.pipe(entities_list))
        if doc[0].pos_ in ALLOWED_POS_ENTITIES
    ]

    # Inclusivity sweep: any token that spaCy parses as PROPN in isolation is
    # promoted to entities_list, even if KeyBERT NER never tagged it. This
    # catches proper nouns that slipped through KeyBERT's per-cluster
    # classification because the cluster context was too thin for NER.
    # Comparison is on the surface form; both lists are already lowercased.
    already_entity = set(entities_list)
    propn_candidates = []
    for t, doc in zip(tags_list, nlp_it.pipe(tags_list)):
        if t in already_entity:
            continue
        if doc[0].pos_ == "PROPN":
            propn_candidates.append(t)

    if propn_candidates:
        entities_list = list(set(entities_list + propn_candidates))
        print(f"Promoted {len(propn_candidates)} tokens from tags to entities "
              f"because they parse as PROPN in isolation")

    # Reapply mutual exclusivity after the POS filter and the PROPN sweep.
    entities_set_temp = set(entities_list)
    tags_list = [t for t in tags_list if t not in entities_set_temp]

    # Remove tokens in BLOCKLIST
    tags_list     = [t for t in tags_list     if t not in BLOCKLIST]
    entities_list = [t for t in entities_list if t not in BLOCKLIST]

    return tags_list, entities_list


class LemmaResolver:
    # Encapsulates the lemma maps and the co-occurrence normaliser.
    #
    # State:
    #   - tags_set / entities_set: corpus-wide token sets after aggregation
    #   - raw_lemmas: original generic surface form -> canonical lemma
    #   - lemma_to_forms: canonical lemma -> set of surface forms
    #     (entities map to themselves)
    #   - important_keywords_set: tags ∪ entities, in their original
    #     surface form, used to filter co-occurrence inputs *before*
    #     lemma replacement
    #
    # Built once per daily corpus and shared across the downstream
    # graph-building, ranking, and narration stages.

    def __init__(self, tags_list, entities_list):
        self.tags_set     = set(tags_list)
        self.entities_set = set(entities_list)

        # Build raw_lemmas with a safety guard.
        #
        # The naive form raw_lemmas = {tag: nlp_it(tag)[0].lemma_ for tag in tags_list}
        # is what produces ugly outputs like "Meloni" -> "melone" and
        # "Maldive" -> "maldivo": spaCy's Italian lemmatiser inflects proper nouns
        # that look like common Italian plurals. The PROPN promotion in
        # aggregate_tags_and_entities already moves these out of tags_list,
        # but we add a second line of defence here:
        #
        #   - If the lemma form is itself in entities_set, the token is some
        #     inflection of a known entity -> keep the original surface form.
        #   - If the token's POS is PROPN under isolated parsing, keep the
        #     surface form regardless of what the lemmatiser tries to do.
        #   - Otherwise, accept the lemma (standard path for common nouns,
        #     adjectives, and verbs).
        #
        # This guard never lemmatises into something semantically wrong; in the
        # worst case it just leaves the surface form unchanged, which is a no-op
        # downstream because lemma_to_forms[lemma] then maps the form to itself.

        self.raw_lemmas = {}
        for tag, doc in zip(tags_list, nlp_it.pipe(tags_list)):
            candidate_lemma = doc[0].lemma_.lower()
            if doc[0].pos_ == "PROPN":
                # Proper noun: never lemmatise (defensive)
                self.raw_lemmas[tag] = tag
            elif candidate_lemma in self.entities_set:
                # The lemma collides with a known entity surface form
                # (the "Meloni -> melone" pattern). Keep the surface form.
                self.raw_lemmas[tag] = tag
            else:
                self.raw_lemmas[tag] = candidate_lemma

        # important_keywords_set is built from ORIGINAL forms so tokens in the
        # article and cluster text are found before lemmatization is applied
        self.important_keywords_set = set(tags_list + entities_list)

        # Invert raw_lemmas: canonical lemma to list of original forms
        self.lemma_to_forms = defaultdict(set)
        for original_form, lemma in self.raw_lemmas.items():
            self.lemma_to_forms[lemma].add(original_form)

        # For entities, the form IS the canonical form (no lemmatization
        # applied), so they map to themselves
        for entity in entities_list:
            self.lemma_to_forms[entity].add(entity)

    def apply_lemmas_to_cooc_string(self, tag_string):
        # Normalizes a pipe-separated co-occurrence string:
        #   - Entities:      left as they are (no lemmatization)
        #   - Generic tags:  replaced with the canonical lemma (via raw_lemmas)
        #   - Unknown tokens: silently discarded
        #
        # After replacement, duplicates that emerge from lemma collapsing
        # are removed.
        #
        # entities_set is intentionally checked first. Mutual exclusivity was
        # enforced in aggregate_tags_and_entities (tags_set has no overlap with
        # entities_set), but in theory a token could appear in raw_lemmas with
        # the same string as an entity (homonymy). Prioritizing entities_set
        # ensures that such tokens are always treated as named entities and
        # never lemmatized.
        result = []
        for token in tag_string.split("|"):
            token = token.strip()
            if not token:
                continue
            if token in self.entities_set:
                result.append(token)
            elif token in self.tags_set:
                result.append(self.raw_lemmas.get(token, token).lower())
            # Tokens outside both sets are silently discarded

        seen   = set()
        deduped = []
        for t in result:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return "|".join(deduped)
