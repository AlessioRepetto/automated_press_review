# Automated Press Review - Italian News Intelligence Pipeline
 
> Public product: **La Parola Data** — a daily, automatically generated reading of the Italian news cycle, published to GitHub Pages.
> Live page: https://alessiorepetto.github.io/la_parola_data/
 
---
 
## Overview
 
**Automated Press Review** is an end-to-end NLP and graph pipeline that turns the daily flow of Italian press into a single, structured intelligence report. Every day, across several time slots, it ingests the public RSS feeds of the main national outlets (AGI, ANSA, Adnkronos, Repubblica, Rai), clusters the news by topic, builds a co-occurrence graph of the key concepts, detects the thematic communities of the day, and lets a large language model narrate the result in publication-ready Italian.
 
The output is not a dashboard of raw metrics but an editorial artifact: a daily wordcloud, a "last 24 hours in brief" recap, a set of highlighted articles, the main thematic communities, and a coverage-and-framing view across outlets. The whole process runs unattended in the cloud and republishes an updated HTML page two to three times a day.
 
The project is deliberately designed around two ideas that recur throughout: **distribution-aware logic** (thresholds derived from the current day's corpus rather than fixed constants, so the system stays robust when the news volume or thematic spread changes) and a **cloud-first execution model** (the report can be rendered fully in memory and published without touching disk, which is what makes scheduled execution on ephemeral runners reliable).
 
---
 
## Objectives
 
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
|-- scripts/                     # pipeline library (NLP / graph / narration)
|   |-- config.py                # env loading and pipeline constants
|   |-- ingestion.py             # RSS download and parsing
|   |-- pipeline.py              # end-to-end orchestration, single entry point
|   |-- cluster_merge.py         # conservative pre-merge of duplicate clusters
|   |-- communities.py           # Louvain + heterogeneity + selection levers
|   |-- llm_narration.py         # all prompts toward Mistral
|   `-- viz_*.py                 # graphical rendering for the report
|
|-- report/                      # everything that builds the public HTML site
|   |-- config.py                # palette, labels, report paths
|   |-- run_report.py            # entry point (pipeline + render + publish)
|   |-- data_collector.py        # PipelineOutput -> report_data dict
|   |-- html_renderer.py         # render_html_string (memory) + render_html (disk)
|   |-- image_builder.py         # inline base64 PNGs for the report
|   |-- github_publisher.py      # publishes the HTML to the site repo
|   `-- templates/
|       |-- template.html
|       `-- style.css
|
|-- sources/news_feeds.csv       # RSS feed definitions (source, section, url)
|-- requirements.txt             # local environment (CUDA torch)
|-- requirements-cloud.txt       # cloud environment (CPU torch)
`-- .env                         # local only, NOT committed
```
 
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
3. Copy `.env.example` to `.env` and fill in the values (Mistral key, publishing token, paths, timezone):
   ```
   cp .env.example .env
   ```
4. Run the pipeline and publish the report:
   ```
   python -m report.run_report --verbose
   ```
   Add `--save-local` to also write the HTML and JSON to disk (default is publish-only, the cloud-first behaviour).
For development and inspection, `notebooks/Analisi_news.ipynb` chains the same stages by importing from `scripts/`; no algorithm is defined in the notebook itself.
 
---
 
## Scheduled cloud execution
 
Production runs on **GitHub Actions** (`ubuntu-latest`, CPU only, Python 3.13), three times a day via cron plus manual dispatch. The job installs the CPU build of `torch`, downloads the NLP assets, runs `report/run_report.py`, and publishes the rendered HTML to the GitHub Pages repo through the GitHub Contents API. No state persists between runs: the report is built in memory and pushed directly.
 
Sensitive values (`MISTRAL_API_KEY`, the publishing token) are injected as GitHub Actions Secrets; non-sensitive configuration lives in the workflow `env:` block.
 
**Containerized fallback.** If an upstream feed ever refused cloud-runner requests outright -- a persistent block that no retry or browser-header workaround could defeat -- the runner is portable to a containerized batch job (build the root `Dockerfile`, deploy as a Google Cloud Run Job, trigger with Cloud Scheduler). This is a documented contingency, not the default deployment; nothing observed so far points to it being necessary.
 
---
 
## Skills Demonstrated
 
- End-to-end NLP pipeline design and orchestration
- Semantic embeddings and density-based clustering (UMAP + HDBSCAN)
- Keyword extraction (custom KeyBERT + MMR)
- Graph and network analysis (co-occurrence graphs, structural roles, Louvain communities)
- Fuzzy, distribution-aware classification
- LLM integration and prompt design for controlled editorial output
- Cloud-first, stateless system design and scheduled deployment (GitHub Actions, GitHub Pages)
- Production engineering: retry logic, timezone correctness, in-memory rendering, secret management
---
 
## Author
 
**Alessio Repetto**
 
GitHub: https://github.com/AlessioRepetto
LinkedIn: https://www.linkedin.com/in/alessiorepetto/
