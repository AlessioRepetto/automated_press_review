"""Report-specific configuration: UI palette, paths, font fallback.

Kept intentionally separate from `scripts/config.py` to avoid polluting
pipeline configuration with rendering concerns.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env so GitHub credentials (and any other secrets) are available.
# Idempotent: safe even if the pipeline already called load_dotenv().
load_dotenv()


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# Paths                                                                        #
# -----------------------------------------------------------------------------#
REPORT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = REPORT_DIR / "templates"
PROJECT_ROOT = REPORT_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

REPORT_FILENAME_FORMAT = "la_parola_data_{stamp}.html"
JSON_FILENAME_FORMAT = "la_parola_data_{stamp}.json"
STAMP_FORMAT = "%Y%m%d_%H%M"



# -----------------------------------------------------------------------------#
# GitHub publishing                                                            #
# -----------------------------------------------------------------------------#
# The report is pushed to a GitHub Pages repo as `index.html` after each run.
# The token is read from the environment (.env) and is NEVER hardcoded.
#
# Required .env entry:
#     GITHUB_TOKEN=github_pat_xxxxxxxxxxxxxxxxxxxx
#
# The remaining values have sensible defaults and can be overridden in .env
# if the target repo ever changes.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "AlessioRepetto")
GITHUB_REPO = os.getenv("GITHUB_REPO", "la_parola_data")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_TARGET_PATH = os.getenv("GITHUB_TARGET_PATH", "index.html")

# Number of attempts per GitHub request before giving up (retry on
# timeouts / connection errors / HTTP 5xx with exponential backoff).
GITHUB_MAX_RETRIES = int(os.getenv("GITHUB_MAX_RETRIES", "3"))

# Set PUBLISH_TO_GITHUB=0 in .env to skip the publish step (e.g. local testing).
PUBLISH_TO_GITHUB = os.getenv("PUBLISH_TO_GITHUB", "1") not in ("0", "false", "False")

# -----------------------------------------------------------------------------#
# Editorial UI palette (NOT the chart palette — that one stays in viz_palette) #
# -----------------------------------------------------------------------------#
COLOR_PRIMARY = "#2E6C80"      # accent 1 — section accents, rules, badges
COLOR_SECONDARY = "#6A9A86"    # accent 2 — secondary accents, quote bars
COLOR_HEADING = "#355C7D"      # all titles (H1/H2/H3)
COLOR_TEXT = "#000000"         # body text
COLOR_MUTED = "#666666"        # captions, footer, dates
COLOR_BG = "#FFFFFF"           # main background
COLOR_BG_SOFT = "#F7F4EE"      # soft warm off-white for boxed callouts


# -----------------------------------------------------------------------------#
# Node role palette                                                            #
# -----------------------------------------------------------------------------#
# Duplicated (intentionally) from the pipeline's viz_* role-color map. We
# accept this small duplication to keep the report module decoupled from the
# pipeline visualization internals: the legend stays valid even if the
# pipeline renames or reorganises its palette.
#
# IMPORTANT: these MUST stay aligned with the colors used by
# `render_highlighted_news_grid` (scripts/viz_news_render.py). If the
# pipeline ever changes its role palette, update both places.
ROLE_COLORS: dict[str, str] = {
    "core":       "#355C7D",
    "bridge":     "#8C2F39",
    "stable":     "#6A9A86",
    "peripheral": "#6F5A8A",
    "generic":    "#9E9E9E",   # neutral grey — generic is the "background" role
}


# -----------------------------------------------------------------------------#
# Legend (shown right after the report title)                                  #
# -----------------------------------------------------------------------------#
# Order is editorial: from most-structurally-central to most-marginal,
# with 'generic' last because it represents lexical background noise rather
# than the day's substantive core.
LEGEND_ROLES: list[dict[str, str]] = [
    {
        "key": "core",
        "label": "Parole centrali",
        "description": (
            "Queste parole rappresentano i concetti che reggono la "
            "struttura del discorso, ricorrono in molte notizie diverse "
            "e sono al cuore di temi ben definiti."
        ),
    },
    {
        "key": "bridge",
        "label": "Parole-ponte",
        "description": (
            "Concetti che collegano racconti distinti: appaiono in fatti "
            "del giorno altrimenti separati e rivelano le connessioni "
            "meno ovvie tra di essi."
        ),
    },
    {
        "key": "stable",
        "label": "Parole tematiche",
        "description": (
            "Concetti saldamente legati a un singolo tema, in "
            "corrispondenza ai quali appaiono spesso, definendo in tal "
            "modo la loro narrazione."
        ),
    },
    {
        "key": "peripheral",
        "label": "Parole di contorno",
        "description": (
            "Concetti marginali: pochi legami, presenza puntuale. "
            "Aggiungono dettagli e sfumature senza incidere sulla "
            "struttura complessiva."
        ),
    },
    {
        "key": "generic",
        "label": "Parole trasversali",
        "description": (
            "Termini che ricorrono in molte notizie e temi diversi. La "
            "loro diffusione non li rende centrali: sono lo sfondo "
            "lessicale del giorno, più che il suo cuore."
        ),
    },
]



# -----------------------------------------------------------------------------#
# Deep-dive intro — shown right after opening the collapsible block            #
# -----------------------------------------------------------------------------#
# A short, non-technical orientation: one framing sentence (the tool's value
# proposition, in plain language) plus a numbered table-of-contents of the
# four sections that follow. Order MUST match the template section order.
DEEP_DIVE_INTRO: dict[str, object] = {
    "title": "Cosa troverai qui sotto",
    "lead": (
        "Ogni giorno le stesse notizie vengono raccontate da molte "
        "testate, con toni e angolazioni diversi. Questa sezione mette "
        "in fila ci\u00f2 che le diverse fonti hanno in comune e ci\u00f2 che le "
        "distingue, per offrirti un quadro pi\u00f9 ordinato e meno schierato "
        "di quello che \u00e8 successo."
    ),
    "sections": [
        {
            "title": "Cosa raccontano i colori",
            "description": (
                "Una breve guida per interpretare le parole evidenziate "
                "nelle pagine seguenti."
            ),
        },
        {
            "title": "Alcune notizie rilevanti",
            "description": (
                "Gli articoli del giorno con maggior peso, con i concetti "
                "chiave messi in evidenza."
            ),
        },
        {
            "title": "I principali temi portanti",
            "description": (
                "I grandi filoni attorno a cui ruotano le notizie, con "
                "qualche articolo che li rappresenta."
            ),
        },
        {
            "title": "Copertura delle principali informazioni",
            "description": (
                "Come le diverse testate hanno trattato le storie pi\u00f9 "
                "importanti, e con quali parole."
            ),
        },
    ],
}

# -----------------------------------------------------------------------------#
# Layout                                                                       #
# -----------------------------------------------------------------------------#
TITLE_REPORT = "LA PAROLA DATA"
SUBTITLE_REPORT = (
    "Le notizie del giorno lette attraverso le parole che le legano. "
    "Un'analisi quotidiana per orientarsi nel rumore dell'informazione."
)

# Static labels (italian)
LABELS = {
    "section_legend_title": "Cosa raccontano i colori: una breve guida per il lettore",
    "section_legend_intro": (
        "Le parole evidenziate non sono scelte a caso: sono "
        "<strong>concetti chiave</strong> estratti dalle notizie del "
        "giorno e classificati in base al ruolo che svolgono. Ogni "
        "colore racconta un modo diverso in cui una parola abita il "
        "discorso quotidiano."
    ),
    "details_open_label": "Approfondisci l'analisi del giorno",
    "details_hint": "Clicca per esplorare concetti, temi e copertura delle notizie",
    "section_recap": "Le ultime 24 ore in breve",
    "section_top_news": "Alcune notizie rilevanti",
    "section_communities": "I principali temi portanti",
    "section_coverage": "Copertura delle principali informazioni",
    "subsection_distinct_terms": "Termini distintivi per testata",
    "caption_wordcloud": "I concetti più rilevanti della giornata",
    "footer_generated": "Generato il",
    "footer_articles": "articoli analizzati",
    "footer_sources": "fonti considerate",
    "no_data": "(nessun dato disponibile per questa sezione)",
    "framing_unavailable": "Analisi di framing non disponibile per questo cluster (fonti insufficienti).",
}


# -----------------------------------------------------------------------------#
# Font fallback                                                                #
# -----------------------------------------------------------------------------#
# The pipeline's viz_news_render module currently uses Windows arial.ttf
# (hardcoded). To keep the script portable across OSes we expose a helper
# that returns the first existing font path among candidates, with logging.

_FONT_CANDIDATES_REGULAR = [
    r"C:\Windows\Fonts\arial.ttf",                                      # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",                   # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux alt
    "/Library/Fonts/Arial.ttf",                                          # macOS
    "/System/Library/Fonts/Helvetica.ttc",                               # macOS fallback
]

_FONT_CANDIDATES_BOLD = [
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _first_existing(candidates: list[str]) -> Optional[str]:
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_font_paths() -> tuple[Optional[str], Optional[str]]:
    """Return (regular, bold) font paths or (None, None) if nothing is found.

    Emits a warning when the Windows path is missing — that path is what the
    pipeline currently hardcodes, so its absence means downstream renderers
    relying on it will fail unless they accept the fallback.
    """
    regular = _first_existing(_FONT_CANDIDATES_REGULAR)
    bold = _first_existing(_FONT_CANDIDATES_BOLD)

    if regular is None:
        logger.warning(
            "No system font found among candidates: %s. "
            "Image rendering may fall back to PIL default font.",
            _FONT_CANDIDATES_REGULAR,
        )
    elif not regular.startswith(("C:", "/Library", "/System")):
        logger.info("Using fallback font for image rendering: %s", regular)

    return regular, bold
