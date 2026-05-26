# Analisi News - Intelligence Pipeline

End-to-end NLP and graph-based pipeline for the daily analysis of Italian news.
Starting from RSS feeds, it produces a daily wordcloud of top concepts, an
LLM-narrated description of core and bridge concepts, the top highlighted
articles, and the main thematic communities with their representative news.

## Project structure

```
Analisi_News/
├── .env                 # environment variables (not committed)
├── .env.example         # template for .env
├── requirements.txt
├── notebooks/
│   └── Analisi_news.ipynb         # orchestration notebook
├── scripts/                       # importable library
│   ├── config.py                  # env loading and pipeline constants
│   ├── models.py                  # spaCy / SentenceTransformer / NLTK
│   ├── ingestion.py               # RSS ingestion
│   ├── preprocessing.py           # text preprocessing and NER helper
│   ├── keyword_extraction.py      # custom KeyBERT with MMR
│   ├── clustering.py              # UMAP + HDBSCAN + cluster representation
│   ├── lemmatization.py           # lemma maps and co-occurrence normalisation
│   ├── graph_builder.py           # co-occurrence matrix and graph
│   ├── graph_metrics.py           # node-level structural metrics
│   ├── fuzzy_classifier.py        # fuzzy node-role classification
│   ├── ranking.py                 # relevance score, ego profile, combined score
│   ├── communities.py             # Louvain detection and community analysis
│   ├── cluster_ranking.py         # top-5 cluster selection and framing
│   ├── llm_narration.py           # Mistral-anchored narration
│   ├── recap.py                   # daily-recap article selection
│   ├── viz_palette.py             # shared colour palettes and visual constants
│   ├── viz_news_render.py         # highlighted-news image rendering
│   ├── viz_ego_graph.py           # ego-graph plotting
│   ├── viz_communities.py         # community graph / wordcloud rendering
│   ├── viz_wordcloud.py           # daily wordcloud
│   └── viz_framing.py             # framing tables and coverage bars
└── sources/
    └── news_feeds.csv             # RSS feed definitions (source, section, url)
```

## Setup

1. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download required NLP assets (once):
   ```
   python -m spacy download it_core_news_lg
   ```
3. Copy `.env.example` to `.env` and fill in the values:
   ```
   cp .env.example .env
   ```
4. Open `notebooks/Analisi_news.ipynb` and run the cells in order.

## Configuration

- **Environment variables** (in `.env`): paths, HTTP settings, timezone,
  font folder, Mistral credentials. Loaded by `scripts/config.py`.
- **Pipeline constants** (in `scripts/config.py`): `TOP_N`, `KEYWORD_MAX_WORDS`,
  `CUMULATIVE_THRESHOLD`, `ALPHA`, `MIN_ARTICLES`, `MIN_SOURCES`,
  `MIN_SCORES`, `BLOCKLIST`, etc. Edit here, not in the notebook.

## Architectural notes

- **Notebook is pure orchestration.** All algorithms, helpers, and rendering
  logic live in `scripts/`. The notebook imports from there and chains the
  stages; no function or class is defined in a notebook cell. This makes
  the pipeline reusable from other entry points (CLI, scheduler, tests).
- **`LemmaResolver` encapsulates lemma maps.** What used to be a set of
  loose globals (`tags_set`, `entities_set`, `raw_lemmas`, `lemma_to_forms`,
  `important_keywords_set`, `apply_lemmas_to_cooc_string`) is now a single
  stateful object built once per daily corpus.
- **Visual palette is centralized in `viz_palette.py`.** `ROLE_COLORS`,
  `ROLE_HIGHLIGHT_COLORS`, `EGO_*`, `COMM_*`, and the source colour palette
  factory all live in one module, removing the forward-reference issue of
  the original notebook (where `ROLE_COLORS` was defined late but referenced
  by upstream rendering helpers).
- **Community visualisation is pure.** `plot_community` receives
  `community_results` as an explicit parameter rather than reading it from
  a module-level global.
- **`build_notebook.py`** (root) regenerates `notebooks/Analisi_news.ipynb`
  from declarative cell definitions. Use it if you want to edit the
  orchestration without hand-merging notebook JSON.

## Portability caveat

`FONT_DIR` in `.env` defaults to `C:\Windows\Fonts` (the original Windows
target). On Linux or macOS, point it at a folder containing TTF files and
adjust the filenames referenced by `scripts/viz_news_render.py` if needed.
