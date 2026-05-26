# Highlighted-news rendering — HTML.
#
# Renders the day's news with matched concepts highlighted, as inline HTML
# (role-coloured <span> elements). The HTML report and the notebook both
# consume this; there is no longer an image-based renderer (the previous
# PIL/matplotlib PNG functions were removed once the report moved to HTML).
#
# Lemmatisation is inverted through `lemma_to_forms` so that all surface
# variants of a matched concept are highlighted, not only the canonical
# lemma.

import math
import re

import pandas as pd

from scripts.viz_palette import ROLE_COLORS, ROLE_HIGHLIGHT_COLORS


def clean_token(token):
    # Strip punctuation and lowercase a single token, for lookup in the
    # form_to_color map. Kept separate so the highlight match rule is
    # in one place: tokens are matched in their punctuation-stripped,
    # lowercase form.
    return token.strip(".,;:!?()[]{}\"\'").lower()


def build_form_to_color(matched_concepts, lemma_to_forms, role_lookup):
    # Maps every surface form of every matched concept to (highlight_color,
    # text_color, role). highlight_color is the lighter pastel used as
    # rectangle background; text_color is the stronger role colour used
    # for the bold-text rendering style.
    form_to_color = {}
    for concept in matched_concepts:
        role        = role_lookup.get(concept, "unclassified")
        highlight   = ROLE_HIGHLIGHT_COLORS.get(role, ROLE_HIGHLIGHT_COLORS["unclassified"])
        text_color  = ROLE_COLORS.get(role, "#555555")
        for form in lemma_to_forms.get(concept, {concept}):
            form_to_color[form.lower()] = (highlight, text_color, role)
    return form_to_color


def format_news_header(row):
    # Builds the per-article header "{source}  -  {dd/mm/yyyy}  {hh:mm}".
    # Date is included only when row["date"] exists and is non-empty.
    # Accepts pandas Timestamps, datetime objects, or strings.
    source = row.get("source", "-") if hasattr(row, "get") else "-"
    time_v = row.get("time", "-")   if hasattr(row, "get") else "-"
    date_v = row.get("date", None)  if hasattr(row, "get") else None

    time_str = str(time_v)[:5] if time_v is not None else "-"

    if date_v is None or (isinstance(date_v, float) and pd.isna(date_v)):
        date_str = ""
    elif hasattr(date_v, "strftime"):
        date_str = date_v.strftime("%d/%m/%Y")
    else:
        date_str = str(date_v).strip()
        if " " in date_str:
            date_str = date_str.split(" ")[0]
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str)
        if m:
            date_str = f"{m.group(3)}/{m.group(2)}/{m.group(1)}"

    if date_str:
        return f"{source}  -  {date_str}  {time_str}"
    return f"{source}  -  {time_str}"


# =============================================================================
# News importance scoring (used by the top-10 selection)
# =============================================================================

def build_form_to_score(top_nodes, lemma_to_forms,
                          combined_score_col="combined_score"):
    # Builds the lookup surface_form_lower -> combined_score, used by
    # the per-article scoring. Each top-N node contributes its
    # combined_score to every one of its surface forms.
    form_to_score = {}
    for _, row in top_nodes.iterrows():
        score = row[combined_score_col]
        for form in lemma_to_forms.get(row["tag"], {row["tag"]}):
            form_to_score[form.lower()] = score
    return form_to_score


def compute_news_density_score(text, form_to_score, max_words=50):
    # Density score for an article: sum of combined_scores of distinct
    # matched surface forms in the first `max_words` tokens, divided by
    # log(word_count). Cap on max_words prevents long articles from
    # accumulating score on tokens far beyond the headline lead.
    # Returns (density, matched_forms, word_count).
    if not isinstance(text, str) or not text.strip():
        return 0.0, [], 0

    text_lower = text.lower()
    words      = text_lower.split()
    word_count = min(len(words), max_words)
    text_lower = " ".join(words[:max_words])

    matched  = []
    total    = 0.0
    seen     = set()

    for form, score in form_to_score.items():
        if form in text_lower and form not in seen:
            matched.append(form)
            total += score
            seen.add(form)

    density = total / math.log(max(word_count, 2))
    return density, matched, word_count


def score_news_importance(df, top_nodes, lemma_to_forms,
                            combined_score_col="combined_score",
                            text_col="text", top_n=3):
    # Article importance score and ranking, based on the density of top-N
    # concepts (by combined_score) present in the first 50 words of each
    # article.
    #
    # Score formula:
    #   sum of combined_score of each top-N concept found in the article
    #   divided by log(word_count) (density normalisation)
    #
    # Returns the top-N articles sorted by score with columns
    # source, date (if available), time, text, matched_concepts,
    # word_count, news_score.

    form_to_score = build_form_to_score(top_nodes, lemma_to_forms,
                                          combined_score_col)

    results = df.copy()
    computed = results[text_col].apply(
        lambda t: compute_news_density_score(t, form_to_score)
    )

    results["news_score"]       = computed.apply(lambda x: x[0])
    results["matched_concepts"] = computed.apply(lambda x: x[1])
    results["word_count"]       = computed.apply(lambda x: x[2])

    return (
        results
        .sort_values("news_score", ascending=False)
        .head(top_n)
        [[c for c in ["source", "date", "time", "text", "matched_concepts",
                       "word_count", "news_score"] if c in results.columns]]
        .reset_index(drop=True)
    )


# =============================================================================
# HTML rendering of highlighted news
# =============================================================================
# Renders the highlighted news as an HTML string: matched concepts are
# pixels with PIL, it emits an HTML string where matched concepts are wrapped
# in <span> elements coloured by node role.
#
# Why a separate function (and not a flag on the image renderer):
#   the image renderer's parameters (width, padding, font_size, line_spacing,
#   article_gap, ...) are all raster concepts with no HTML equivalent — line
#   wrapping, font sizing and spacing are the browser's / CSS's job. A new
#   function keeps each renderer honest about what it actually does.
#
# Highlight match rule is IDENTICAL to the image path:
#   - tokens are matched in punctuation-stripped, lowercase form (clean_token)
#   - every surface form of every matched concept is highlighted, via the
#     lemma_to_forms inversion (build_form_to_color)
#   - highlighting is per-article (a concept matched only in article 3 is
#     highlighted only there)
#
# Visual style: matched word = role-coloured text + underline, NO background
# (the news block background stays white). Colours come from the CSS classes
# role-core / role-bridge / role-stable / role-peripheral / role-generic /
# role-unclassified, so the actual hex values live in the report stylesheet,
# not here.

from html import escape as _html_escape

# Canonical role -> CSS class name. Any role not in this map (including the
# "unclassified" fallback produced by build_form_to_color) is rendered with
# the role-unclassified class.
_ROLE_CSS_CLASS = {
    "core":       "role-core",
    "bridge":     "role-bridge",
    "stable":     "role-stable",
    "peripheral": "role-peripheral",
    "generic":    "role-generic",
}
_DEFAULT_ROLE_CSS_CLASS = "role-unclassified"


def _role_css_class(role):
    # Maps a node role to its CSS class, with the unclassified fallback.
    return _ROLE_CSS_CLASS.get(role, _DEFAULT_ROLE_CSS_CLASS)


def _render_text_segment(segment, form_to_color, max_words=None):
    # Renders one text segment (a title or a description) as inline HTML.
    # Tokenises on whitespace (same rule as the image renderer), optionally
    # trims to max_words, escapes every token, and wraps matched tokens in a
    # <span class="news-kw role-..."> element.
    #
    # max_words=None means "no trimming" (used for the title, which must
    # never be cut mid-sentence). A integer cap is used for the description.
    if not segment or not segment.strip():
        return "", False

    words = segment.split()
    trimmed = max_words is not None and len(words) > max_words
    if trimmed:
        words = words[:max_words]

    rendered = []
    for token in words:
        key = clean_token(token)
        safe = _html_escape(token)
        if form_to_color and key in form_to_color:
            _, _, role = form_to_color[key]
            css = _role_css_class(role)
            rendered.append(f'<span class="news-kw {css}">{safe}</span>')
        else:
            rendered.append(safe)

    return " ".join(rendered), trimmed


def _render_news_body_html(text, form_to_color, max_words):
    # Renders a single article body as inline HTML.
    #
    # The article `text` is built upstream as `title + "\n" + description`.
    # Title and description are split on that first newline and rendered as
    # two lines separated by a <br>. Both lines share the same style and the
    # same highlighting logic; only the layout separates them.
    #
    # Trimming rule: the title is never trimmed (it must stay whole); the
    # max_words cap applies to the description only. If the text has no
    # newline (e.g. an article with title only, or description only) it is
    # rendered as a single line with no <br>.
    if not isinstance(text, str) or not text.strip():
        return ""

    if "\n" in text:
        title_part, desc_part = text.split("\n", 1)
    else:
        title_part, desc_part = text, ""

    title_html, _ = _render_text_segment(title_part, form_to_color, max_words=None)
    desc_html, trimmed = _render_text_segment(desc_part, form_to_color, max_words)

    if trimmed:
        desc_html += " [...]"

    # Join title and description; emit a <br> only when both are present.
    if title_html and desc_html:
        return f'{title_html}<br>{desc_html}'
    return title_html or desc_html


def render_news_html(news_rows, matched_concepts_per_row,
                     lemma_to_forms, tags_ranking_df,
                     max_words_per_article=60):
    # Builds the highlighted-news HTML for a set of articles.
    #
    # Parameters (same substantive inputs as the image renderer):
    #   news_rows                 iterable of news rows (each must expose
    #                             'source'/'time'/'text', may expose 'date')
    #   matched_concepts_per_row  list aligned with news_rows: canonical
    #                             concepts matched in each article
    #   lemma_to_forms            canonical lemma -> surface forms
    #   tags_ranking_df           must contain 'tag' and 'concept_type'
    #   max_words_per_article     body trim cap (mirrors the image renderer)
    #
    # Returns:
    #   An HTML string: one <article class="news-item"> per input row, each
    #   with a header (source - date time, never highlighted) and a body
    #   (matched concepts wrapped in role-coloured spans).
    #
    # The role-colour CSS classes are expected to be defined by the report
    # stylesheet. When used in the notebook, wrap the result in
    # IPython.display.HTML to preview it.
    role_lookup = tags_ranking_df.set_index("tag")["concept_type"].to_dict()

    articles_html = []
    for row, matched_concepts in zip(news_rows, matched_concepts_per_row):
        # Per-article highlight map — identical construction to the image path
        form_to_color = build_form_to_color(
            matched_concepts, lemma_to_forms, role_lookup
        )

        header = _html_escape(format_news_header(row))
        body = _render_news_body_html(
            str(row.get("text", "")).strip(),
            form_to_color,
            max_words_per_article,
        )

        articles_html.append(
            '<article class="news-item">'
            f'<div class="news-item-header">{header}</div>'
            f'<div class="news-item-body">{body}</div>'
            '</article>'
        )

    return '<div class="news-list">' + "\n".join(articles_html) + "</div>"


# -----------------------------------------------------------------------------#
# Notebook helper                                                              #
# -----------------------------------------------------------------------------#
# CSS that colours the role spans. In the final HTML report this lives in the
# report stylesheet (report/templates/style.css); inside a Jupyter notebook
# that stylesheet is not loaded, so render_news_html output would show in
# plain black. _NOTEBOOK_ROLE_CSS is a self-contained copy used ONLY for
# notebook preview — it is intentionally NOT emitted by render_news_html so
# the report keeps a single source of truth for its styling.
_NOTEBOOK_ROLE_CSS = """<style>
.news-list .news-item { border-top: 1px solid #d6d3cc; padding: 14px 0; }
.news-list .news-item-header { font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .07em; color: #2E6C80;
    margin-bottom: 6px; }
.news-list .news-item-body { font-size: 15px; line-height: 1.6; color: #000; }
.news-kw { text-decoration: underline; text-underline-offset: 2px;
    font-weight: 600; background: transparent; }
.news-kw.role-core         { color: #355C7D; }
.news-kw.role-bridge       { color: #8C2F39; }
.news-kw.role-stable       { color: #6A9A86; }
.news-kw.role-peripheral   { color: #6F5A8A; }
.news-kw.role-generic      { color: #9E9E9E; }
.news-kw.role-unclassified { color: #555555; }
</style>"""


def display_news_html(news_rows, matched_concepts_per_row,
                      lemma_to_forms, tags_ranking_df,
                      max_words_per_article=60):
    # Notebook convenience wrapper: builds the news HTML via render_news_html
    # AND renders it inline with the role colours applied.
    #
    # Use this inside a Jupyter notebook instead of manually pairing
    # render_news_html with IPython.display.HTML — it bundles the role-colour
    # CSS so the preview matches the final report.
    #
    # Has no return value; it displays directly. For the report pipeline use
    # render_news_html (the report supplies its own stylesheet).
    from IPython.display import HTML, display

    body_html = render_news_html(
        news_rows, matched_concepts_per_row,
        lemma_to_forms, tags_ranking_df,
        max_words_per_article=max_words_per_article,
    )
    display(HTML(_NOTEBOOK_ROLE_CSS + body_html))
