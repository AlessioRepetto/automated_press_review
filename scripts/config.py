# Project configuration.
#
# Two layers:
#   1. Environment variables loaded from `.env` via python-dotenv.
#      Secrets, paths, and deployment-dependent settings live here.
#   2. Pipeline constants defined as module-level Python attributes.
#      These tune the analytical behaviour of the pipeline and should
#      be changed here, not halfway through the notebook.
#
# The notebook imports from this module and never redefines these values.

import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


# =============================================================================
# Environment variables
# =============================================================================
# Resolve the project root (one level above scripts/) and load the .env
# placed there. This makes the loader robust to the working directory
# from which the notebook or a script is launched.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _require(name):
    # Reads an environment variable and raises if it is missing or empty.
    # Used for variables the pipeline cannot run without.
    value = os.getenv(name)
    if value is None or value == "":
        raise EnvironmentError(
            f"Required environment variable {name!r} is not set. "
            f"Check the .env file at {PROJECT_ROOT / '.env'}."
        )
    return value


# CSV file with feed sources. Path is resolved relative to PROJECT_ROOT
# when it is not absolute, so the notebook works regardless of the
# current working directory.
_csv_raw = _require("CSV_FILE")
CSV_FILE = str(
    Path(_csv_raw) if os.path.isabs(_csv_raw) else PROJECT_ROOT / _csv_raw
)

# Name of the section treated as a front-page signal
MAIN_SECTION_NAME = _require("MAIN_SECTION_NAME")

# HTTP settings for RSS ingestion
REQUEST_TIMEOUT = int(_require("REQUEST_TIMEOUT"))
USER_AGENT = _require("USER_AGENT")

# Timezone for date normalization
TZ_ROME = ZoneInfo(_require("TZ_ROME"))

# Mistral API
MISTRAL_API_KEY = _require("MISTRAL_API_KEY")
MISTRAL_MODEL = _require("MISTRAL_MODEL")


# =============================================================================
# Pipeline constants (analytical parameters, not deployment settings)
# =============================================================================

# Keyword extraction window: consider only the first N words of each article
KEYWORD_MAX_WORDS = 50

# Top-N concepts retained for visualization and reporting
TOP_N = 100

# Number of Louvain runs: keep the partition with maximum modularity
N_RUNS = 50

# Cumulative percentage of total edge weight retained when pruning the
# co-occurrence graph. Lower values (0.60-0.75) keep more edges on
# asymmetric distributions.
CUMULATIVE_THRESHOLD = 0.95

# Cluster ranking trade-off between average concept quality and cluster mass:
#   alpha = 1.0 -> pure quality
#   alpha = 0.0 -> pure mass
#   alpha = 0.6 -> quality-skewed balance (default)
ALPHA = 0.6

# Eligibility filters for cross-source cluster analyses (asymmetry, framing).
MIN_ARTICLES = 3
MIN_SOURCES = 2

# Minimum fuzzy score by role for assignment in the node classifier.
# `stable` has a slightly lower threshold because membership in a thematic
# cluster tends to produce more moderate scores by construction.
MIN_SCORES = {
    "generic":    0.55,
    "core":       0.55,
    "bridge":     0.55,
    "stable":     0.45,
    "peripheral": 0.55,
}

# Threshold for dropping avg_neigh_deg when too correlated with eigen
CORR_THRESHOLD = 0.85

# Tokens to exclude regardless of the extraction source.
# News-agency names, stopword leftovers, or low-information terms
# that survive preprocessing.
BLOCKLIST = {
    "adnkronos", "ansa", "agi",
}

# Framing analysis parameters
MIN_DF_PER_SOURCE = 1   # a term must appear at least once in the source text
TOP_N_FRAMING = 8       # distinctive terms retained per (cluster, source) pair


# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
