"""Build the `report_data` dict that feeds the Jinja2 template, and
serialize it to JSON for reproducibility.

The dict is intentionally flat and JSON-serializable: this makes both the
template logic trivial and the artifact debuggable / replayable without
re-running the pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from report.config import (
    DEEP_DIVE_INTRO,
    LABELS,
    LEGEND_ROLES,
    ROLE_COLORS,
    STAMP_FORMAT,
    SUBTITLE_REPORT,
    TITLE_REPORT,
)
from report.image_builder import (
    build_coverage_bar_image,
    build_wordcloud_image,
)
from scripts.viz_news_render import render_news_html
from scripts.pipeline import PipelineOutput

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# Helpers                                                                      #
# -----------------------------------------------------------------------------#
def _format_article_date(row: pd.Series) -> str:
    """Format the publication date/time of an article as 'DD/MM/YYYY HH:MM'."""
    date = row.get("date")
    time_ = row.get("time")
    parts: list[str] = []
    if pd.notna(date):
        try:
            parts.append(pd.to_datetime(date).strftime("%d/%m/%Y"))
        except Exception:
            parts.append(str(date))
    if pd.notna(time_) and str(time_).strip():
        parts.append(str(time_))
    return " ".join(parts)


def _truncate_text(text: str, max_words: int = 50) -> str:
    """Truncate a string to `max_words` whitespace-separated tokens."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "…"


def _framing_to_table(framing_df: pd.DataFrame) -> dict[str, Any]:
    """Pivot the long-form framing DataFrame into a rank x source table
    suitable for HTML rendering.

    Returns:
        {
          'headers': [source1, source2, ...],   # column order = source
          'rows':    [[term_rank1_src1, term_rank1_src2, ...],
                      [term_rank2_src1, ...]],
        }
    """
    if framing_df is None or framing_df.empty:
        return {"headers": [], "rows": []}

    cols = {c.lower(): c for c in framing_df.columns}
    src_col = cols.get("source") or framing_df.columns[0]
    term_col = cols.get("term") or cols.get("keyword") or framing_df.columns[1]
    rank_col = cols.get("rank")

    if rank_col is None:
        framing_df = framing_df.copy()
        framing_df["__rank__"] = framing_df.groupby(src_col).cumcount() + 1
        rank_col = "__rank__"

    pivot = (
        framing_df.pivot_table(
            index=rank_col, columns=src_col,
            values=term_col, aggfunc="first",
        )
        .fillna("")
        .sort_index()
    )
    return {
        "headers": [str(c) for c in pivot.columns],
        "rows": [[str(cell) for cell in row] for row in pivot.values.tolist()],
    }


def _build_legend_roles() -> list[dict[str, str]]:
    """Build the role legend shown right after the title.

    Joins narrative descriptions (LEGEND_ROLES) with display colors
    (ROLE_COLORS), in the order defined by LEGEND_ROLES.
    """
    return [
        {
            "key": role["key"],
            "label": role["label"],
            "description": role["description"],
            "color": ROLE_COLORS.get(role["key"], "#999999"),
        }
        for role in LEGEND_ROLES
    ]


# -----------------------------------------------------------------------------#
# Per-section builders                                                         #
# -----------------------------------------------------------------------------#
def _build_recap_section(output: PipelineOutput) -> dict[str, Any]:
    return {"text": output.day_recap or ""}


def _build_top_news_section(output: PipelineOutput) -> dict[str, Any]:
    """Top 10 articles, rendered as inline HTML (concepts wrapped in
    role-coloured spans) rather than a PNG image.

    `articles` is still emitted as plain metadata so the JSON artifact
    stays inspectable without parsing the HTML.
    """
    if output.top10_news is None or output.top10_news.empty:
        return {"html": None, "articles": []}

    rows = [row for _, row in output.top10_news.iterrows()]
    matched = [row.get("matched_concepts", []) for _, row in output.top10_news.iterrows()]

    html = render_news_html(
        news_rows=rows,
        matched_concepts_per_row=matched,
        lemma_to_forms=output.lemma_to_forms,
        tags_ranking_df=output.tags_ranking_df,
    )

    articles: list[dict[str, Any]] = []
    for _, row in output.top10_news.iterrows():
        articles.append({
            "source": str(row.get("source", "")),
            "date": _format_article_date(row),
            "text": _truncate_text(str(row.get("text", "")), max_words=50),
            "matched_concepts": list(row.get("matched_concepts", []) or []),
        })
    return {"html": html, "articles": articles}


def _build_communities_section(output: PipelineOutput) -> list[dict[str, Any]]:
    """One entry per community: the LLM-generated title and the representative
    articles rendered as inline HTML (per user spec — no other content)."""
    communities: list[dict[str, Any]] = []
    for comm_idx, content in (output.community_results or {}).items():
        top_news = content.get("top_news")
        title = str(content.get("title", "")).replace("*", "").strip()

        if top_news is None or top_news.empty:
            communities.append({
                "index": int(comm_idx),
                "title": title,
                "html": None,
                "is_empty": True,
            })
            continue

        rows = [row for _, row in top_news.iterrows()]
        matched = [row.get("matched_concepts", []) for _, row in top_news.iterrows()]

        html = render_news_html(
            news_rows=rows,
            matched_concepts_per_row=matched,
            lemma_to_forms=output.lemma_to_forms,
            tags_ranking_df=output.tags_ranking_df,
        )
        communities.append({
            "index": int(comm_idx),
            "title": title,
            "html": html,
            "is_empty": False,
        })
    return communities


def _build_coverage_section(output: PipelineOutput) -> list[dict[str, Any]]:
    """Top 5 clusters with: title, description, coverage bar PNG, framing table."""
    clusters: list[dict[str, Any]] = []

    for _, row in output.top5_clusters.iterrows():
        topic_id = int(row["topic_id"])
        summary = output.cluster_summaries.get(topic_id, {})
        title = str(summary.get("title", f"Cluster {topic_id}"))
        description = str(summary.get("description", ""))

        bar_uri = build_coverage_bar_image(
            output.top_data[output.top_data["Topic"] == topic_id],
            output.journal_color_map,
            title=title,
        )

        framing_df = output.framing_by_cluster.get(topic_id)
        framing_table = _framing_to_table(framing_df)
        framing_available = bool(framing_table["headers"])

        clusters.append({
            "topic_id": topic_id,
            "title": title,
            "description": description,
            "bar_data_uri": bar_uri,
            "framing_table": framing_table,
            "framing_available": framing_available,
            "n_articles": int(row.get("n_articles", 0)),
            "n_sources": int(row.get("n_sources", 0)),
        })
    return clusters


# -----------------------------------------------------------------------------#
# Public entry points                                                          #
# -----------------------------------------------------------------------------#
def build_report_data(output: PipelineOutput) -> dict[str, Any]:
    """Build the dict consumed by the Jinja2 template."""
    logger.info("Building report_data dict")

    report_data: dict[str, Any] = {
        "title": TITLE_REPORT,
        "subtitle": SUBTITLE_REPORT,
        "labels": LABELS,
        "generated_at": output.generated_at,
        "generated_at_iso": output.generated_at.isoformat(timespec="minutes"),
        "generated_at_display": output.generated_at.strftime("%d/%m/%Y - %H:%M"),
        "stamp": output.generated_at.strftime(STAMP_FORMAT),

        "n_articles": output.n_articles,
        "n_sources": output.n_sources,
        "sources": sorted(output.distinct_journals),

        "deep_dive_intro": DEEP_DIVE_INTRO,                   # NEW
        "legend_roles": _build_legend_roles(),                # NEW

        "wordcloud": {"image_data_uri": build_wordcloud_image(output.top_nodes)},
        "recap": _build_recap_section(output),
        "top_news": _build_top_news_section(output),
        "communities": _build_communities_section(output),
        "coverage": _build_coverage_section(output),
    }
    return report_data


def _json_default(obj: Any) -> Any:
    """JSON fallback for non-natively-serializable objects."""
    import datetime as _dt
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_report_data_json(report_data: dict[str, Any], path: Path) -> None:
    """Serialize report_data to JSON.

    Note: data-URI strings (base64 images) are *included*. The JSON is
    therefore as large as the HTML — that is intentional, since the goal
    is full reproducibility / replay without re-running the pipeline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("report_data JSON written: %s", path)
