"""Standalone entry point for the Analisi News HTML report.

Usage:
    python -m report.run_report
    # or
    python report/run_report.py

What it does:
    1. Runs the entire Analisi News pipeline (scripts.pipeline.run_pipeline)
    2. Re-renders all visual assets as base64-embedded PNGs
    3. Renders the Jinja2 template into a single self-contained HTML string
    4. Publishes that HTML to the GitHub Pages repo as `index.html`
    5. ONLY with --save-local: also writes the HTML and the report_data
       JSON to <output-dir> for local inspection / debugging

Local disk behaviour:
    By default the report is published WITHOUT touching the local disk —
    the HTML is rendered in memory and sent straight to GitHub. This makes
    the run safe for a scheduled cloud environment with no persistent disk.
    Pass --save-local to also keep a local copy of the HTML and the JSON.

The publish step can be disabled with --no-publish or by setting
PUBLISH_TO_GITHUB=0 in `.env`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure project root is on PYTHONPATH when executed as `python report/run_report.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from report.config import (
    ARTIFACTS_DIR,
    GITHUB_BRANCH,
    GITHUB_MAX_RETRIES,
    GITHUB_OWNER,
    GITHUB_REPO,
    GITHUB_TARGET_PATH,
    GITHUB_TOKEN,
    JSON_FILENAME_FORMAT,
    PUBLISH_TO_GITHUB,
    REPORT_FILENAME_FORMAT,
)
from report.data_collector import build_report_data, dump_report_data_json
from report.github_publisher import (
    GitHubPublishError,
    GitHubTarget,
    publish_html,
)
from report.html_renderer import render_html_string
from scripts.pipeline import run_pipeline


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _publish_to_github(html_content: str, generated_at_display: str,
                       saved_locally: bool) -> None:
    """Publish the HTML (in memory) to GitHub as index.html.

    Failures here are logged but DO NOT raise. When the report was also
    saved locally the message points the user to that file; when it was
    not (the default), the message simply notes the run can be retried.
    """
    target = GitHubTarget(
        owner=GITHUB_OWNER,
        repo=GITHUB_REPO,
        branch=GITHUB_BRANCH,
        path=GITHUB_TARGET_PATH,
        token=GITHUB_TOKEN,
        max_retries=GITHUB_MAX_RETRIES,
    )
    commit_message = f"Aggiornamento report \u2014 {generated_at_display}"

    log = logging.getLogger("run_report")
    try:
        commit_url = publish_html(html_content, target, commit_message)
        log.info("Published to GitHub Pages. Commit: %s", commit_url)
    except GitHubPublishError as e:
        if saved_locally:
            log.error(
                "GitHub publish failed \u2014 a local copy of the HTML was "
                "saved (see above). Reason: %s", e,
            )
        else:
            log.error(
                "GitHub publish failed. No local copy was saved "
                "(run without --save-local). Re-run to retry. Reason: %s", e,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate (and publish) the Analisi News HTML daily report.",
    )
    parser.add_argument(
        "--save-local", action="store_true",
        help="Also save the HTML and the report_data JSON to --output-dir. "
             "By default nothing is written to disk (publish only).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ARTIFACTS_DIR,
        help=f"Directory for HTML + JSON output when --save-local is set "
             f"(default: {ARTIFACTS_DIR}).",
    )
    parser.add_argument(
        "--no-publish", action="store_true",
        help="Skip the GitHub publish step (overrides PUBLISH_TO_GITHUB).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    log = logging.getLogger("run_report")

    log.info("=== Analisi News \u2014 report generation ===")

    # 1. Pipeline
    output = run_pipeline()

    # 2. Report data + images
    report_data = build_report_data(output)

    # 3. Render the HTML in memory (always — no disk access here).
    html_content = render_html_string(report_data)

    # 4. Optionally save HTML + JSON to disk. The JSON follows the HTML:
    #    both are written together, or neither is.
    if args.save_local:
        stamp = report_data["stamp"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        html_path = args.output_dir / REPORT_FILENAME_FORMAT.format(stamp=stamp)
        json_path = args.output_dir / JSON_FILENAME_FORMAT.format(stamp=stamp)

        html_path.write_text(html_content, encoding="utf-8")
        dump_report_data_json(report_data, json_path)

        log.info("HTML saved: %s", html_path)
        log.info("JSON saved: %s", json_path)
    else:
        log.info("Local save skipped (default; pass --save-local to enable).")

    # 5. Publish to GitHub Pages (from the in-memory HTML).
    if args.no_publish:
        log.info("Publish skipped (--no-publish).")
    elif not PUBLISH_TO_GITHUB:
        log.info("Publish skipped (PUBLISH_TO_GITHUB=0 in .env).")
    else:
        _publish_to_github(
            html_content,
            report_data["generated_at_display"],
            saved_locally=args.save_local,
        )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
