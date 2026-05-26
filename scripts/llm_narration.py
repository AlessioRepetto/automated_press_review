# LLM narration (Mistral-anchored).
#
# Four narration paths, each with strict prompting and exponential-backoff retries:
#
# - describe_ego_node:    short description of a top concept (core or bridge)
# - title_community:      max-7-word natural-phrase title for a Louvain community
# - summarise_cluster:    title + description for an HDBSCAN cluster (JSON)
# - summarise_day_recap:  daily editorial recap from a thematically diverse
#                          article selection
#
# Prompts are anchored on neighbor roles and concrete article excerpts to
# avoid hallucinations, and are explicitly instructed not to force a
# narrative thread when topics are heterogeneous.

import json
import time
from collections import defaultdict

import networkx as nx
from mistralai import Mistral

from scripts.communities import (
    HETEROGENEITY_BROAD,
    HETEROGENEITY_FOCUSED,
    compute_community_heterogeneity,
    get_news_for_community,
    get_news_for_tag,
    rank_news_by_ego_coverage,
    score_news_for_community,
)


# =============================================================================
# Concept-level narration (core / bridge)
# =============================================================================

def describe_ego_node(client, model, tag, role, n_neighbours,
                       neighbour_context, news_context,
                       max_retries=5, initial_wait=10):
    # Generates the short description of a concept via LLM, anchored on neighbors
    # of the ego graph and on a sample of representative articles.
    # Implements retries with exponential backoff in case of rate limiting.
    role_explanation = {
        "core":   "ricorre in molti contesti diversi ed e' al centro di numerose notizie.",
        "bridge": "collega temi diversi che normalmente non si incrociano.",
    }.get(role, "e' una parola rilevante nel panorama informativo di oggi.")

    prompt = f"""Sei un redattore di una newsletter di informazione.

Oggi la parola "{tag}" {role_explanation}
E' direttamente associata a {n_neighbours} altri concetti nelle notizie di oggi.

Queste sono le notizie piu' rappresentative in cui compare:
{news_context}

I concetti a essa piu' vicini nelle notizie di oggi:
{neighbour_context}

Scrivi UNA sola frase breve (massimo 40 parole) che spieghi perche' "{tag}" e' rilevante oggi.
Regole:
- Se le notizie hanno un tema comune evidente, parti da quel tema con un fatto concreto
- Se le notizie sono su temi diversi, descrivi onestamente questa diversita',
  spiega che la parola collega contesti distinti, citandone due o tre brevemente
- Non forzare un filo narrativo che non esiste
- Non usare frasi conclusive tipo "insomma", "non si scappa", "e' la chiave per"
- Niente stile da analista, scrivi come parleresti a un amico informato
- Non iniziare con "Oggi" e non usare virgolette attorno alla parola
- Massimo 40 parole
- Se i temi sono diversi, NON fare una frase generica di chiusura finale
"""

    wait = initial_wait
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                print(f"   Rate limit hit, waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                wait *= 2  # exponential backoff
            else:
                raise  # immediately re-raise errors that are not rate limits

    raise RuntimeError(f"Max retries ({max_retries}) exceeded for tag '{tag}'")


def analyse_top_words(top_tags, tags_ranking_df, df, lemma_to_forms,
                      concepts_graph, cluster_freq, similarity_by_topic_list,
                      api_key, model="mistral-large-latest"):
    # Full pipeline: for each tag, generate a short LLM description
    # anchored on concrete news facts, then select and display
    # the most representative articles sorted by neighbor coverage
    # of the ego graph.
    #
    # Returns a dict {tag: {"role": str, "description": str, "top_news": pd.DataFrame}}
    client      = Mistral(api_key=api_key)
    role_lookup = tags_ranking_df.set_index("tag")["concept_type"].to_dict()
    results     = {}

    for tag in top_tags:
        print(f"\n-- Analysing: {tag} --")

        # Retrieve news containing the ego node
        news_df = get_news_for_tag(tag, df, lemma_to_forms)
        print(f"   News found: {len(news_df)}")

        if news_df.empty:
            print(f"   No news found for '{tag}', skipping.")
            continue

        # Build the ego graph and sort news by neighbor coverage
        ego_G    = nx.ego_graph(concepts_graph, tag, radius=1)
        top_news = rank_news_by_ego_coverage(tag, ego_G, news_df, lemma_to_forms)

        # Build the neighbor context for the LLM prompt
        neighbour_roles = {
            n: role_lookup.get(n, "other")
            for n in ego_G.nodes if n != tag
        }
        neighbours_by_role = defaultdict(list)
        for n, role in neighbour_roles.items():
            neighbours_by_role[role].append(n)

        neighbour_context = "\n".join(
            f"- {role}: {', '.join(nodes)}"
            for role, nodes in neighbours_by_role.items()
            if role != "other"
        )

        # Build the compact news context from the already selected news
        news_context = "\n".join(
            f"- {row.get('text', '')[:150].strip()}"
            for _, row in top_news.iterrows()
        )

        # LLM: short description anchored on concrete facts
        role         = role_lookup.get(tag, "core")
        n_neighbours = concepts_graph.degree(tag)

        print(f"   Generating description...")
        description = describe_ego_node(
            client, model, tag, role, n_neighbours, neighbour_context, news_context
        )

        results[tag] = {
            "role":        role,
            "description": description,
            "top_news":    top_news,
        }

        print(f"   Done.")

    return results


# =============================================================================
# Community-level narration
# =============================================================================

def title_community(client, model, comm, visible_news, tags_ranking_df,
                    heterogeneity=HETEROGENEITY_FOCUSED,
                    max_retries=5, initial_wait=10):
    # Generates a short, natural-phrase title (max 7 words) for the community.
    #
    # The title is built EXCLUSIVELY from the texts of `visible_news` — the
    # very articles shown to the reader for this community. This guarantees
    # the title cannot mention anything the reader does not see: previously
    # the title was built on a wider, coverage-completed set of articles,
    # which could surface sub-themes absent from the displayed top-5.
    #
    # The community concept list is intentionally NOT passed to the model:
    # those concepts are a superset of what the visible articles contain,
    # and exposing them would reopen the same title/articles mismatch.
    #
    # `heterogeneity` (HETEROGENEITY_FOCUSED / HETEROGENEITY_BROAD) is computed
    # upstream from the dispersion of the community's representative articles.
    # It tells the model which register to use: a focused community wants a
    # specific title, a broad one wants a title that names the shared area
    # without pretending the sub-stories are a single event.

    # Context is the text of the visible articles only.
    news_context = "\n".join(
        f"- {str(row.get('text', ''))[:150].strip()}"
        for _, row in visible_news.iterrows()
    )

    # The guidance adapts to the measured heterogeneity of the community.
    # The model is TOLD whether the community is focused or broad — it does
    # not guess. To keep the prompt sharp, ONLY the block relevant to the
    # actual community type is injected: instruction, few-shot examples and
    # rules are all selected by the same if/else, so the model is never
    # shown guidance for the other case.
    if heterogeneity == HETEROGENEITY_BROAD:
        register_instruction = (
            "QUESTA COMMUNITY E' AMPIA: raccoglie piu' sotto-storie diverse "
            "ma appartenenti a una stessa area. Il titolo deve nominare "
            "l'AREA COMUNE che le contiene TUTTE. E' un errore grave "
            "scegliere una sola delle sotto-storie: il titolo che nomina un "
            "filone e ne ignora altri presenti nel gruppo non e' accettabile."
        )
        examples_block = """Esempi di titolo per una community AMPIA come questa:
- Notizie su: Roland Garros, infortunio di Messi, Serie A
  -> Titolo: Lo sport del giorno tra campi e polemiche
  (tennis, calcio internazionale e Serie A sono storie diverse: il titolo
  copre TUTTE nominando l'area comune; "Roland Garros" sarebbe sbagliato
  perche' ignora il calcio e la Serie A)
- Notizie su: elezioni comunali, apertura delle scuole, protesta dei balneari
  -> Titolo: Le questioni che agitano la vita pubblica
  (le elezioni e la vicenda scuola/turismo sono filoni distinti: il titolo
  li abbraccia entrambi; "Elezioni amministrative" lascerebbe fuori meta'
  del gruppo)
- Notizie su: manovra di governo, sanita', trasporti, sciopero
  -> Titolo: Le tensioni nella gestione del Paese
  (sotto-temi diversi di politica interna: titolo sull'area, non su uno)"""
        rules_block = """Regole:
- Il titolo deve essere una frase naturale e scorrevole in italiano,
  non un elenco di parole chiave. Sono ammessi articoli e preposizioni.
- Massimo 7 parole
- Il titolo deve essere TRASVERSALE: deve coprire TUTTE le notizie del
  gruppo. Prima di rispondere, verifica mentalmente che ogni notizia
  fornita rientri nel titolo che hai scelto. Se anche una sola resta
  fuori, il titolo e' troppo stretto: allargalo.
- NON nominare una singola sotto-storia, un singolo luogo o un singolo
  protagonista: sarebbe ingiusto verso le altre notizie del gruppo
- Il titolo nomina l'area tematica comune, non l'evento piu' citato
- Non usare virgolette o punteggiatura finale"""
    else:
        register_instruction = (
            "QUESTA COMMUNITY E' FOCALIZZATA: le notizie ruotano attorno "
            "a un singolo tema o evento specifico. Il titolo deve essere "
            "concreto e puntuale, deve nominare QUEL tema specifico."
        )
        examples_block = """Esempi di titolo per una community FOCALIZZATA come questa:
- Notizie su: elezioni comunali, affluenza, urne, ballottaggi
  -> Titolo: Affluenza in calo alle elezioni amministrative
  (le notizie parlano tutte dello stesso evento: il titolo e' puntuale)
- Notizie su: decisioni della BCE, tassi, mutui, spread
  -> Titolo: I tassi della BCE pesano sui mutui
  (un solo tema economico ben definito: titolo concreto)"""
        rules_block = """Regole:
- Il titolo deve essere una frase naturale e scorrevole in italiano,
  non un elenco di parole chiave. Sono ammessi articoli e preposizioni.
- Massimo 7 parole
- Il titolo deve essere specifico e puntuale sul tema preciso del gruppo
- Deve comunque coprire tutte le notizie: se una notizia resta fuori,
  rivedi il titolo
- NON combinare due entita' (nomi, luoghi, persone) se non compaiono
  insieme nelle notizie fornite: non inventare collegamenti
- Non usare virgolette o punteggiatura finale"""

    prompt = f"""Sei un redattore di una newsletter di informazione.
Ti vengono fornite le notizie che verranno mostrate al lettore per questo
gruppo.
Notizie mostrate al lettore:
{news_context}

Il tuo compito e' scrivere UN titolo, una frase breve e naturale di
MASSIMO 7 parole, che identifichi il tema di queste notizie e copra TUTTE
le notizie elencate sopra. Il titolo deve riferirsi SOLO a queste notizie:
non introdurre temi, luoghi o protagonisti che non compaiano qui sopra.

{register_instruction}

{examples_block}

{rules_block}
"""

    wait = initial_wait
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                print(f"   Rate limit hit, waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                wait *= 2
            else:
                raise

    raise RuntimeError(f"Max retries ({max_retries}) exceeded for community title generation")


def analyse_top_communities(top_comm_indices, best_communities, tags_ranking_df,
                              df, lemma_to_forms, concepts_graph,
                              api_key, embedding_model, model="mistral-large-latest",
                              top_n=5):
    # For each top community: retrieve relevant news, sort it by score,
    # generate a title via LLM, and display the results.
    #
    # `embedding_model` is used to measure each community's heterogeneity
    # (focused vs broad) from the dispersion of its representative articles,
    # so the title can be calibrated to the right level of specificity.
    #
    # Returns a dict {comm_idx: {"title": str, "nodes": set, "top_news": pd.DataFrame}}.
    client  = Mistral(api_key=api_key)
    results = {}

    for comm_idx in top_comm_indices:
        comm = best_communities[comm_idx]
        print(f"\n-- Community {comm_idx}  ({len(comm)} nodes) --")

        # Retrieve and sort news (retrieval now filters on distinctive
        # concepts, so tags_ranking_df is required).
        news_df  = get_news_for_community(comm, lemma_to_forms, df, tags_ranking_df)
        print(f"   News found: {len(news_df)}")

        if news_df.empty:
            print("   No news found, skipping.")
            continue

        top_news = score_news_for_community(
            comm, news_df, lemma_to_forms, tags_ranking_df, top_n=top_n
        )

        # Measure heterogeneity from the representative articles, so the
        # title prompt knows whether to be specific or broad.
        heterogeneity, radius = compute_community_heterogeneity(
            top_news, embedding_model,
        )
        print(f"   Heterogeneity: {heterogeneity} (radius={radius:.3f})")

        # Title generation — built ONLY from the visible top-5 articles,
        # so the title always matches what the reader actually sees.
        print("   Generating title...")
        title = title_community(
            client, model, comm, top_news, tags_ranking_df,
            heterogeneity=heterogeneity,
        )

        results[comm_idx] = {
            "title":    title,
            "nodes":    comm,
            "top_news": top_news,
        }

        print(f"   Title: {title}")

    return results


# =============================================================================
# Cluster-level narration (title + description in a single JSON call)
# =============================================================================

def summarise_cluster(api_key, topic_id, cluster_articles, concepts,
                       model="mistral-large-latest",
                       max_articles=15, max_chars_per_article=200,
                       max_retries=5, initial_wait=10):
    # Generates a title (max 6 words) and a short description (1-2 sentences)
    # for a single HDBSCAN cluster through a single Mistral call
    # returning structured JSON.
    #
    # The article context is built directly from cluster_articles
    # (no scoring is needed; the cluster is already a coherent semantic unit).
    # Concepts are passed as thematic anchors.
    #
    # Returns a dict {"title": str, "description": str}.

    # Cap both the number of articles and the length per article, so the prompt stays bounded
    snippets = "\n".join(
        f"- [{row['source']}] {str(row['text'])[:max_chars_per_article].strip()}"
        for _, row in cluster_articles.head(max_articles).iterrows()
    )

    prompt = f"""Sei un redattore di una newsletter di informazione.
Ti vengono forniti alcuni articoli che parlano dello stesso evento e i concetti chiave estratti.

Concetti chiave (in ordine di importanza):
{', '.join(concepts) if concepts else '(nessun concetto classificato)'}

Articoli:
{snippets}

Restituisci ESCLUSIVAMENTE un oggetto JSON valido con questa struttura:
{{"title": "...", "description": "..."}}

Regole:
- "title": massimo 6 parole, specifico e concreto, niente virgolette, niente punteggiatura finale
- "description": 1-2 frasi (max 40 parole) che sintetizzano il contenuto fattuale, neutre, in italiano
- Nessun testo prima o dopo il JSON, nessun blocco markdown
"""

    client  = Mistral(api_key=api_key)

    wait = initial_wait
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            return {
                "title":       parsed.get("title", "").strip(),
                "description": parsed.get("description", "").strip(),
            }
        except json.JSONDecodeError:
            # Malformed JSON; retry with the same prompt before giving up
            if attempt == max_retries - 1:
                raise RuntimeError(f"Cluster {topic_id}: invalid JSON after {max_retries} attempts")
            continue
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                print(f"   Rate limit, waiting {wait}s (retry {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                wait *= 2
            else:
                raise

    raise RuntimeError(f"Cluster {topic_id}: max retries exceeded")


# =============================================================================
# Daily editorial recap
# =============================================================================

def summarise_day_recap(api_key, recap_articles_df,
                        model="mistral-large-latest",
                        max_chars_per_article=450,
                        max_concepts_per_article=8,
                        max_retries=5,
                        initial_wait=10):
    # Generates the daily recap from a thematically diverse selection of
    # articles (top-N clusters * top-K articles per cluster).
    # Output is Italian markdown, structured as a short editorial recap
    # of the last 24 hours of coverage, with strict factual constraints
    # enforced via the prompt.
    if recap_articles_df.empty:
        return "Nessuna notizia disponibile per il riassunto."

    # Group by topic_id so the LLM sees thematic blocks
    blocks = []
    block_counter = 1
    for topic_id, group in recap_articles_df.groupby("topic_id", sort=False):
        block_articles = []
        for _, row in group.iterrows():
            source = row.get("source", "Fonte non disponibile")
            time_value = row.get("time", "")
            if hasattr(time_value, "strftime"):
                time_str = time_value.strftime("%d/%m/%Y %H:%M")
            else:
                time_str = str(time_value)

            text = str(row.get("text", "")).strip()[:max_chars_per_article]

            matched_concepts = row.get("matched_concepts", [])
            if isinstance(matched_concepts, (list, tuple, set)):
                concepts_str = ", ".join(
                    list(matched_concepts)[:max_concepts_per_article]
                )
            else:
                concepts_str = str(matched_concepts)

            block_articles.append(
                f"Notizia {block_counter}\n"
                f"Fonte: {source}\n"
                f"Data/Ora pubblicazione: {time_str}\n"
                f"Concetti evidenziati: {concepts_str}\n"
                f"Testo: {text}\n"
            )
            block_counter += 1

        blocks.append(
            f"=== Tema {topic_id} ===\n" + "\n---\n".join(block_articles)
        )

    articles_context = "\n\n".join(blocks)

    prompt = f"""Sei un redattore di una newsletter di informazione.

Ti viene fornita una selezione di notizie pubblicate nelle ULTIME 24 ORE,
raggruppate per tema. Per ogni tema ci sono fino a 3 articoli, presi dai
cluster di notizie piu' significativi della giornata.

Il tuo compito e' scrivere un breve riassunto editoriale in italiano delle
notizie pubblicate nelle ULTIME 24 ORE.

Regole NON NEGOZIABILI:
- Usa SOLO le informazioni presenti nelle notizie fornite. Non aggiungere
  contesto esterno, anche se lo conosci.
- NON inventare cause, conseguenze, dettagli, date, luoghi o nomi che non
  siano gia' presenti nei testi.
- NON dare per scontato QUANDO sono accaduti i fatti. Gli articoli sono
  stati pubblicati nelle ultime 24 ore, ma gli eventi raccontati possono
  essere accaduti in qualunque momento (anche nel passato remoto). Se il
  testo non specifica quando, NON dirlo.
- Usa formule come "nelle ultime 24 ore sono state pubblicate notizie su...",
  "viene riportato che...", "e' stato annunciato che...", invece di
  "oggi e' successo", "questa mattina", "in giornata".
- NON elencare le notizie una per una: identifica i principali temi portanti
  della giornata e raccontali in modo continuo.
- Mantieni un tono sobrio, chiaro e accessibile a un cittadino medio.
  Niente enfasi giornalistica artificiale.
- Lunghezza: circa 150-220 parole.
- Restituisci solo il testo finale, senza premesse e senza titolo.

Notizie raggruppate per tema:
{articles_context}
"""

    client = Mistral(api_key=api_key)

    wait = initial_wait
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limited" in str(e).lower():
                print(f"Rate limit hit, waiting {wait}s before retry "
                      f"{attempt + 1}/{max_retries}...")
                time.sleep(wait)
                wait *= 2
            else:
                raise

    raise RuntimeError("Max retries exceeded for daily recap summary generation")
