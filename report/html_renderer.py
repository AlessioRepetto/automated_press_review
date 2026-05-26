"""Render the report_data dict into the final HTML via Jinja2.

Kept deliberately thin: all data preparation is in `data_collector`, all
styling is in `templates/style.css`. This module only assembles them.

Two entry points:
    - render_html_string(report_data) -> str
        Assembles the HTML and RETURNS it as a string. No disk access.
        Used for in-memory publishing (cloud / scheduled runs).
    - render_html(report_data, output_path) -> None
        Thin wrapper: renders the string and writes it to a file.
        Used when a local copy is explicitly requested.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from report.config import TEMPLATES_DIR

logger = logging.getLogger(__name__)


def _make_env() -> Environment:
    """Build the Jinja2 environment.

    Notes:
        - `select_autoescape` is enabled for HTML (XSS-safe by default).
        - The day recap, however, comes from the LLM as Markdown-like text;
          we render it in the template using the `|safe` filter on a
          *bleached* version, see `_sanitize_recap` below.
    """
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "htm"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _sanitize_recap_markdown(text: str) -> str:
    """Convert a small, well-defined Markdown subset to HTML safely.

    The LLM returns the daily recap as a paragraph or a few paragraphs
    occasionally containing **bold**, *italic*, and line breaks. We DO
    NOT use a full markdown engine to avoid pulling a heavy dependency
    just for this; instead we apply a tight conversion limited to the
    constructs the prompt is expected to produce.

    The template uses `|safe` on the output, so we ensure no raw HTML
    from the LLM is preserved (we escape first, then re-inject our own
    tags).
    """
    from html import escape
    import re

    if not text:
        return ""

    s = escape(text)
    # **bold**
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s, flags=re.DOTALL)
    # *italic* (avoid matching inside already-replaced bold)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s, flags=re.DOTALL)
    # Paragraph breaks: blank line -> </p><p>
    paragraphs = re.split(r"\n\s*\n", s.strip())
    paragraphs = [p.replace("\n", "<br>") for p in paragraphs if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paragraphs)


def render_html_string(report_data: dict[str, Any]) -> str:
    """Render the report HTML and RETURN it as a string (no disk access).

    This is the single place where the Jinja2 assembly happens. Both the
    in-memory publish path and the local-save path go through here.
    """
    env = _make_env()

    # Inject the sanitized recap so the template can use it as `|safe`
    report_data = dict(report_data)  # shallow copy
    recap = report_data.get("recap", {}).get("text", "") or ""
    report_data["recap_html"] = _sanitize_recap_markdown(recap)

    # Inline the CSS so the HTML stays a single self-contained file
    css_path = TEMPLATES_DIR / "style.css"
    report_data["inline_css"] = css_path.read_text(encoding="utf-8")

    template = env.get_template("template.html")
    return template.render(**report_data)


def render_html(report_data: dict[str, Any], output_path: Path) -> None:
    """Render the report HTML and write it to `output_path`.

    Thin wrapper over `render_html_string` for the case where a local
    file copy is explicitly requested.
    """
    html = render_html_string(report_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written: %s", output_path)
