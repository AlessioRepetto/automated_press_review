"""Image building for the HTML report.

Builds the two remaining raster assets — the daily wordcloud and the
per-cluster coverage bar charts — by re-running the pipeline's matplotlib
visualisations off-screen and encoding each PNG as a base64 data URI for
inline embedding.

News articles are NO LONGER rendered as images: they are emitted as inline
HTML by `scripts.viz_news_render.render_news_html` (see data_collector).
That removed the news path's dependency on a system font; this module's
font handling is therefore relevant ONLY to the wordcloud and bar charts.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # off-screen — must precede pyplot import
import matplotlib.pyplot as plt

from report.config import resolve_font_paths

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# Font fallback patch                                                          #
# -----------------------------------------------------------------------------#
# The wordcloud / bar-chart renderers may still pull a Windows-only font path.
# To keep the report OS-portable we monkey-patch ImageFont.truetype so that a
# non-existent path falls back to the first available system font. No-op on
# Windows with arial.ttf present.
def _install_font_fallback() -> None:
    from PIL import ImageFont

    regular, bold = resolve_font_paths()
    if regular is None:
        return

    original = ImageFont.truetype

    def patched_truetype(font=None, size=10, *args, **kwargs):  # noqa: ANN001
        candidate = font
        if isinstance(candidate, (str, os.PathLike)) and not os.path.exists(candidate):
            name = str(candidate).lower()
            replacement = bold if ("bd" in name or "bold" in name) and bold else regular
            logger.debug(
                "ImageFont.truetype: %s not found, falling back to %s",
                candidate, replacement,
            )
            candidate = replacement
        return original(candidate, size, *args, **kwargs)

    ImageFont.truetype = patched_truetype  # type: ignore[assignment]


_install_font_fallback()


# -----------------------------------------------------------------------------#
# Encoding helpers                                                             #
# -----------------------------------------------------------------------------#
def _figure_to_data_uri(fig, dpi: int = 150) -> str:
    """Encode a matplotlib Figure as a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _capture_current_figure_or_close() -> Optional[str]:
    """Capture the *current* matplotlib figure as a data URI, if any exists."""
    fig_nums = plt.get_fignums()
    if not fig_nums:
        return None
    fig = plt.figure(fig_nums[-1])
    return _figure_to_data_uri(fig)


# -----------------------------------------------------------------------------#
# Public builders                                                              #
# -----------------------------------------------------------------------------#
def build_wordcloud_image(top_nodes) -> Optional[str]:
    """Daily wordcloud."""
    from scripts.viz_wordcloud import plot_daily_wordcloud
    plt.close("all")
    try:
        # show_legend=False: the report has its own role legend section.
        plot_daily_wordcloud(top_nodes, show_legend=False)
    except Exception as e:  # pragma: no cover
        logger.exception("Wordcloud rendering failed: %s", e)
        return None
    return _capture_current_figure_or_close()


def build_coverage_bar_image(
    top_data_slice, journal_color_map, title: str,
) -> Optional[str]:
    """Horizontal stacked bar of per-source coverage for one cluster."""
    from scripts.viz_framing import describe_cat_column_stacked

    if top_data_slice is None or top_data_slice.empty:
        return None

    plt.close("all")
    try:
        # show_title=False: the cluster title is already the section heading
        # in the report; custom_title is kept for non-report callers.
        describe_cat_column_stacked(
            top_data_slice, "source", journal_color_map, custom_title=title,
            show_title=False,
        )
    except Exception as e:  # pragma: no cover
        logger.exception("Coverage bar rendering failed: %s", e)
        return None

    return _capture_current_figure_or_close()
