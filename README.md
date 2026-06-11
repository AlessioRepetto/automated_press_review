# Automated Press Review - Italian News Intelligence Pipeline

![La Parola Data Logo](images/la_parola_data_logo.jpg)
 
> Public product: **La Parola Data** - a daily, automatically generated reading of the Italian news cycle, published to GitHub Pages.
> Live page: https://alessiorepetto.github.io/la_parola_data/
 
---
 
## Why this project matters

In a digital information environment shaped by volume, speed, repetition, and editorial framing, understanding the news is not only a matter of access to information. It is also a matter of structure, comparison, and critical reading.

Every day, the various news sources publish hundreds of articles on overlapping events. Each source selects its own priorities, vocabulary, emphasis, and narrative angle. Reading a single source can therefore provide only a partial view, while following all available sources manually is unrealistic for an individual reader.

This project was created to explore whether data science and artificial intelligence can support a more reflective way of reading the news.

The goal is not to replace journalism, nor to claim neutrality through automation. The project starts from the assumption that every representation of the news is naturally partial, and its purpose is to make this partiality easier to observe by analysing multiple sources together, identifying recurring themes, detecting connections between concepts, and reducing the dependence on the framing of any single outlet.

From this perspective, the project has both a technical and an ethical motivation: using automation not to accelerate passive consumption of information, but to support critical reading, comparison, and awareness.

The final output, **La Parola Data**, is an automatically generated press review that attempts to summarise the daily Italian news cycle by relying on a structured analytical pipeline before any generative narration is produced.
 
---
 
## Overview

**Automated Press Review** is an end-to-end NLP, graph analysis, and LLM pipeline that transforms the daily flow of Italian online news into a structured and automatically published press review.

The system reads the public RSS feeds of ten national outlets - AGI, Adnkronos, ANSA, Corriere della Sera, Il Fatto Quotidiano, Il Giornale, Il Sole 24 Ore, Internazionale, Rai, and Repubblica - and processes them as a single daily corpus rather than as a set of isolated articles.

The pipeline cleans and deduplicates the collected articles, groups them semantically, extracts relevant concepts, builds a co-occurrence graph, detects the day's thematic communities, and uses a large language model to narrate the result in publication-ready Italian.

The output is an editorial artifact: a daily wordcloud, a "last 24 hours in brief" recap, a set of highlighted articles, the main thematic communities, and a coverage-and-framing view across outlets. The whole process runs unattended in the cloud and republishes an updated HTML page - the public product, **La Parola Data** - three times a day.

The project is not designed as a simple RSS aggregator. Its purpose is to analyse the structure of the news cycle: which topics dominate the day, which concepts connect different stories, which articles are representative of broader themes, and how recurring patterns emerge across multiple sources.
 
---
 
## Added Value

The added value of the project lies in the analytical layer built between raw news articles and the final generated report.

Instead of asking a language model to summarise a set of articles directly, the system first organises the news corpus through data science techniques:

- semantic clustering groups articles that refer to similar events or themes, even when they use different wording;
- concept extraction identifies the most relevant terms emerging from each cluster;
- graph analysis shows how concepts are connected across the daily coverage;
- community detection identifies broader thematic areas;
- structural role classification distinguishes central, bridge, stable, peripheral, and generic concepts;
- representative article selection anchors the report to concrete source material.

Only after this analytical structure has been created is the LLM used to generate the final narration.

This design choice is central to the project. The LLM is not treated as an oracle and is not asked to define the structure of reality on its own. It is used as a narrative interface over a prior analytical process based on embeddings, clustering, graph metrics, source aggregation, and explicit rules.

In this sense, the project explores a more responsible use of generative AI: one where automation supports interpretation, but does not hide the need for method, transparency, and critical judgement.

The project is deliberately designed around two ideas that recur throughout the implementation: **distribution-aware logic** and a **cloud-first execution model**. Thresholds are derived from the current day's corpus rather than from fixed constants, so the system remains more robust when news volume or thematic spread changes. The report can also be rendered fully in memory and published without relying on persistent local storage, making scheduled execution on ephemeral cloud runners more reliable.
 
---
 
## Objectives
 
- Surface, from a high-volume multi-source news stream, the information that genuinely matters
- Retell it concisely and more impartially than any single outlet, by anchoring narration to what multiple sources actually report
- Ingest and normalise heterogeneous RSS feeds into a clean, deduplicated daily corpus
- Cluster news semantically without a fixed number of topics, using density-based clustering
- Extract interpretable key concepts per cluster and connect them in a co-occurrence graph
- Assign each concept a structural role (`core`, `bridge`, `stable`, `peripheral`, `generic`) using fuzzy, percentile-based classification
- Detect the day's thematic communities and select representative articles for each
- Generate sober, editorially consistent Italian narration via an LLM (community titles, cluster summaries, daily recap)
- Publish a self-contained HTML report to GitHub Pages on a daily schedule, with no manual intervention
---
 
## How it works
 
The pipeline is a single linear flow, orchestrated by `scripts/pipeline.py::run_pipeline()` and shared by both the development notebook and the production runner.
 
```
[RSS feeds]
   |
   v
[ingestion]          download, dedup, HTML cleaning, browser-header retry  -> news dataframe
   |
   v
[preprocessing]      lemmatisation, NER, text normalisation
   |
   v
[embeddings + UMAP]  multilingual sentence embeddings + dimensionality reduction
   |
   v
[HDBSCAN]            density-based clustering (no preset K)
   |
   v
[cluster merge]      conservative pre-merge of near-duplicate clusters
   |
   v
[KeyBERT + MMR]      per-cluster key-concept extraction
   |
   v
[co-occurrence graph]  NetworkX graph of concepts
   |
   v
[role classification]  fuzzy roles from percentile-normalised structural metrics
   |
   v
[Louvain]            community detection on the graph
   |
   v
[LLM narration]      community titles, cluster summaries, daily recap (Mistral)
   |
   v
[report + publish]   in-memory HTML rendering -> publish to GitHub Pages
```
 
### Ingestion
 
Feeds are downloaded, deduplicated and stripped of HTML (BeautifulSoup). Requests carry realistic browser headers and retry with backoff on transient failures (`403`/`5xx`/network), but never retry on `404`/`410`, where the resource is gone for good. This handles the occasional `403` some outlets return to cloud-runner requests: the rejections are intermittent rather than IP-persistent, so a later attempt typically succeeds.
 
### Semantic clustering
 
Each article is encoded with a multilingual SentenceTransformer (`paraphrase-multilingual-mpnet-base-v2`), reduced with UMAP, and clustered with HDBSCAN. HDBSCAN is chosen deliberately: it discovers the number of topics from the data and isolates noise, instead of forcing a preset `K` onto a corpus whose size and shape change every day.
 
### Key-concept extraction and the co-occurrence graph
 
A custom KeyBERT + MMR routine extracts the salient concepts per cluster. Concepts are then linked in a co-occurrence graph whose edge weight combines two signals:
 
```
similarity(A, B) = similarity_by_topic_list + similarity_by_same_news
```
 
i.e. how often two concepts share a thematic cluster, and how often they appear in the same article. A third word-level kNN term was removed after it proved to add noise without information.
 
### Fuzzy node roles
 
Every concept receives a role computed from structural metrics, normalised by percentile rank rather than absolute values. The `generic` role takes priority over `core` and `bridge` when its score dominates, which keeps ubiquitous background words ("anno", "italia") out of the day's headline concepts. Graph density is controlled with a cumulative-weight edge threshold (keep the edges that together cover 70% of total weight), more robust on atypical days than a fixed percentile cut.
 
### Communities and narration
 
Louvain detects the thematic communities; only distinctive roles (everything except `generic`) count toward a community's representative articles. An LLM (Mistral) then writes the editorial layer: community titles calibrated to each community's heterogeneity, cluster summaries, and a daily recap that follows a fixed hard-news-then-soft-news ordering. Timestamps are emitted in Europe/Rome regardless of the runner's clock.
 
---
 
## The daily report
 
A reader of La Parola Data gets a compact, source-agnostic reading of the day:
 
- **Daily wordcloud** of the day's leading concepts
- **"Last 24 hours in brief"** — a sober recap that orders hard news first and soft news (sport, culture, entertainment) last
- **Highlighted articles** — the most representative pieces of the day
- **Main thematic communities** — the day's macro-topics, each with an LLM-written title and its representative articles
- **Coverage and framing** — how different outlets cover the same clusters, and with which words
The development notebook goes further than the published page. Beyond the sections above, it includes **ego-graph views of individual concepts** and a **per-concept narration** that attributes a meaning and a structural role to a single word within the day's discourse. These exploratory analyses are intentionally left out of the public report to keep it focused and quick to read; they remain available in the notebook for deeper inspection.
 
---
 
## Architecture highlights
 
A few design decisions that go beyond "make it run":
 
- **Distribution-aware thresholds.** Fuzzy-classification cutoffs are derived from the current day's corpus percentiles, so the system adapts to days with few or many articles and to concentrated or dispersed topics.
- **`cluster_freq` as a non-redundant feature.** The number of distinct clusters a term appears in was added as a role feature precisely because its correlation with raw frequency was only ~0.68 — it carries genuine information about how cross-cutting a concept is.
- **Conservative cluster merge.** Near-duplicate clusters are merged only when they pass a double `AND` test (centroid cosine and Jaccard on the topic list), pairwise and non-transitively, on freshly recomputed embeddings so the merge never silently corrupts the main pipeline state.
- **Cloud-safe rendering.** HTML rendering is split into an in-memory string builder and a disk-writing wrapper; the publisher accepts either a path or an in-memory string. This is what lets the report exist and ship without persistent storage.
- **Lemmatisation timing.** Lemmatisation is applied after the co-occurrence strings are built, with lowercasing enforced at the source. Proper nouns get a promotion sweep to avoid degenerate lemmas (e.g. "Meloni" collapsing to "melone" on thin contexts).
---
 
## Repository Structure
 
```
automated_press_review/
|
|-- .github/workflows/
|   `-- daily-report.yml         # scheduled cloud execution (cron + manual dispatch)
|
|-- images/
|   `-- .gitkeep         
|   `-- la parola data logo.jpg  # project logo
|
|-- notebooks/
|   `-- Analisi_news.ipynb       # development orchestration + extra exploratory views
|
|-- scripts/                     # pipeline library (NLP / graph / narration / viz)
|   |-- config.py                # env loading and pipeline constants
|   |-- models.py                # spaCy / SentenceTransformer / NLTK loaders
|   |-- ingestion.py             # RSS download and parsing
|   |-- preprocessing.py         # text preprocessing and NER helper
|   |-- keyword_extraction.py    # custom KeyBERT with MMR
|   |-- clustering.py            # UMAP + HDBSCAN + cluster representation
|   |-- cluster_merge.py         # conservative pre-merge of duplicate clusters
|   |-- lemmatization.py         # lemma maps and co-occurrence normalisation
|   |-- graph_builder.py         # co-occurrence matrix and graph
|   |-- graph_metrics.py         # node-level structural metrics
|   |-- fuzzy_classifier.py      # fuzzy node-role classification
|   |-- ranking.py               # relevance score, ego profile, combined score
|   |-- communities.py           # Louvain detection and community analysis
|   |-- cluster_ranking.py       # top-cluster selection and framing
|   |-- recap.py                 # daily-recap article selection
|   |-- llm_narration.py         # Mistral-anchored narration
|   |-- pipeline.py              # end-to-end orchestration, single entry point
|   |-- viz_palette.py           # shared colour palettes and visual constants
|   |-- viz_wordcloud.py         # daily wordcloud
|   |-- viz_news_render.py       # highlighted-news image rendering
|   |-- viz_ego_graph.py         # ego-graph plotting (notebook-only view)
|   |-- viz_communities.py       # community graph / wordcloud rendering
|   `-- viz_framing.py           # framing tables and coverage bars
|
|-- report/                      # builds and publishes the public HTML site
|   |-- config.py                # palette, labels, report paths
|   |-- run_report.py            # entry point (pipeline + render + publish)
|   |-- data_collector.py        # PipelineOutput -> report_data dict
|   |-- html_renderer.py         # render_html_string (memory) + render_html (disk)
|   |-- image_builder.py         # inline base64 PNGs for the report
|   |-- github_publisher.py      # publishes the HTML to the site repo
|   |-- templates/
|   |   |-- template.html
|   |   `-- style.css
|   `-- README.md                # report-layer notes
|
|-- sources/
|   `-- news_feeds.csv           # RSS feed definitions (source, section, url)
|
|-- .gitignore
|-- requirements.txt             # local environment (CUDA torch)
|-- requirements-cloud.txt       # cloud environment (CPU torch)
`-- README.md
```
 
> A local `.env` (Mistral key, publishing token, paths, timezone) is read by `scripts/config.py`; it is gitignored and never committed.
 
---
 
## Tech Stack
 
| Area | Tools |
|---|---|
| Language | Python 3.13 |
| NLP | spaCy (`it_core_news_lg`), SentenceTransformers (`paraphrase-multilingual-mpnet-base-v2`), custom KeyBERT + MMR, NLTK |
| Dimensionality reduction & clustering | UMAP, HDBSCAN |
| Graphs | NetworkX, Louvain community detection |
| LLM | Mistral API (`mistral-large-latest`) via the `mistralai` SDK |
| Reporting | Jinja2 for the HTML template; matplotlib / seaborn / WordCloud / Pillow / adjustText for inline images |
| Cloud execution | GitHub Actions (Ubuntu CPU runner) |
| Publishing | GitHub Contents API to a separate GitHub Pages repo |
 
---
 
## Setup
 
1. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the spaCy model (once):
   ```
   python -m spacy download it_core_news_lg
   ```
3. Create a local `.env` with the required values (Mistral key, publishing token, paths, timezone). It is read by `scripts/config.py` and is gitignored.
4. Run the pipeline and publish the report:
   ```
   python -m report.run_report --verbose
   ```
   Add `--save-local` to also write the HTML and JSON to disk (default is publish-only, the cloud-first behaviour).
For development and inspection, `notebooks/Analisi_news.ipynb` chains the same stages by importing from `scripts/`; no algorithm is defined in the notebook itself. The notebook also produces the ego-graph and single-concept views that are not part of the published report.
 
---
 
## Scheduled cloud execution
 
Production runs on **GitHub Actions** (`ubuntu-latest`, CPU only, Python 3.13), three times a day via cron plus manual dispatch. The job installs the CPU build of `torch`, downloads the NLP assets, runs `report/run_report.py`, and publishes the rendered HTML to the GitHub Pages repo through the GitHub Contents API. No state persists between runs: the report is built in memory and pushed directly.
 
Sensitive values (`MISTRAL_API_KEY`, the publishing token) are injected as GitHub Actions Secrets; non-sensitive configuration lives in the workflow `env:` block.
 
**Containerized fallback.** If an upstream feed ever refused cloud-runner requests outright -- a persistent block that no retry or browser-header workaround could defeat -- the pipeline is portable to a containerized batch job (for example a Google Cloud Run Job triggered by Cloud Scheduler). This is a documented contingency, not the default deployment; nothing observed so far points to it being necessary.
 
---
 
## Skills Demonstrated
 
- End-to-end NLP pipeline design and orchestration
- Semantic embeddings and density-based clustering (UMAP + HDBSCAN)
- Keyword extraction (custom KeyBERT + MMR)
- Graph and network analysis (co-occurrence graphs, structural roles, Louvain communities)
- Fuzzy, distribution-aware classification
- LLM integration and prompt design for controlled editorial output
- Bias-aware, multi-source aggregation and editorial synthesis
- Cloud-first, stateless system design and scheduled deployment (GitHub Actions, GitHub Pages)
- Production engineering: retry logic, timezone correctness, in-memory rendering, secret management
---
 
## Author
 
**Alessio Repetto**
 
GitHub: https://github.com/AlessioRepetto

LinkedIn: https://www.linkedin.com/in/alessiorepetto/
