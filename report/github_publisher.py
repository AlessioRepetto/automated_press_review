"""Publish the generated report to a GitHub repository via the REST API.

Used to push the daily HTML to a GitHub Pages repo as `index.html`,
overwriting the previous version so that Pages always serves the latest
report.

Why the REST API (and not git):
    - No `git` binary required on the host, no local clone.
    - A single file update maps cleanly to one endpoint:
      PUT /repos/{owner}/{repo}/contents/{path}
    - GitHub Pages picks up the commit automatically.

Resilience:
    Network requests are wrapped in an exponential-backoff retry. Transient
    failures (timeouts, connection errors, HTTP 5xx) are retried; permanent
    failures (401/403/404 — bad token, missing permission, wrong repo) fail
    immediately, since retrying them is pointless.

Authentication:
    A fine-grained Personal Access Token with `Contents: read and write`
    scoped to the target repo only. The token is read from the environment
    (loaded from `.env` by python-dotenv) and is NEVER hardcoded or logged.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

_API_ROOT = "https://api.github.com"

# Per-request timeout, in seconds. Generous because the HTML embeds base64
# images (wordcloud, bar charts) and can be a heavy upload on a slow link.
_TIMEOUT = 60

# Retry policy for transient failures.
_DEFAULT_MAX_RETRIES = 3        # total attempts per request
_INITIAL_BACKOFF = 2.0          # seconds; doubles after each failed attempt

# HTTP status codes that are transient and worth retrying.
_RETRYABLE_STATUS = {500, 502, 503, 504}


@dataclass
class GitHubTarget:
    """Coordinates of the file to publish."""

    owner: str
    repo: str
    branch: str = "main"
    path: str = "index.html"          # path within the repo
    token: str = ""                   # PAT — injected at call time, never logged
    max_retries: int = _DEFAULT_MAX_RETRIES

    @property
    def contents_url(self) -> str:
        return f"{_API_ROOT}/repos/{self.owner}/{self.repo}/contents/{self.path}"


class GitHubPublishError(RuntimeError):
    """Raised when the publish step fails permanently."""


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int,
    **kwargs,
) -> requests.Response:
    """Perform an HTTP request with exponential-backoff retry.

    Retries on:
        - requests.Timeout / requests.ConnectionError (network transient)
        - HTTP 5xx responses (server-side transient)
        - HTTP 429 (treated as a generic transient)

    Does NOT retry on other 4xx responses: those are returned to the caller
    as-is so it can produce a precise diagnostic.

    Raises:
        GitHubPublishError: if every attempt fails with a transient error.
    """
    backoff = _INITIAL_BACKOFF
    last_reason = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(method, url, timeout=_TIMEOUT, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_reason = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                logger.warning(
                    "GitHub request %s failed (%s). Retry %d/%d in %.0fs...",
                    method, type(e).__name__, attempt, max_retries, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise GitHubPublishError(
                f"GitHub request failed after {max_retries} attempts "
                f"(network error). Last reason: {last_reason}"
            ) from e

        # Transient server-side error or rate limit -> retry
        if resp.status_code in _RETRYABLE_STATUS or resp.status_code == 429:
            last_reason = f"HTTP {resp.status_code}"
            if attempt < max_retries:
                logger.warning(
                    "GitHub request %s returned %s. Retry %d/%d in %.0fs...",
                    method, resp.status_code, attempt, max_retries, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise GitHubPublishError(
                f"GitHub request failed after {max_retries} attempts. "
                f"Last reason: {last_reason}"
            )

        # Any other response (2xx or a permanent 4xx): return to caller.
        return resp

    # Unreachable, but keeps type checkers happy.
    raise GitHubPublishError(f"GitHub request failed. Last reason: {last_reason}")


def _get_existing_sha(target: GitHubTarget) -> str | None:
    """Return the blob SHA of the file currently at `target.path`, or None
    if the file does not exist yet (first publish).

    The SHA is required by the GitHub API to update an existing file —
    it is GitHub's optimistic-concurrency guard against blind overwrites.
    """
    resp = _request_with_retry(
        "GET",
        target.contents_url,
        max_retries=target.max_retries,
        headers=_auth_headers(target.token),
        params={"ref": target.branch},
    )

    if resp.status_code == 200:
        sha = resp.json().get("sha")
        logger.debug("Existing %s found (sha=%s)", target.path, sha)
        return sha
    if resp.status_code == 404:
        logger.info("%s does not exist yet — first publish.", target.path)
        return None

    raise GitHubPublishError(
        f"Unexpected response while checking {target.path}: "
        f"HTTP {resp.status_code} — {resp.text[:300]}"
    )


def _resolve_html_bytes(html_source) -> bytes:
    """Return the HTML as UTF-8 bytes from either a file path or a string.

    Disambiguation: if `html_source` is a path-like value pointing to an
    existing file, the file is read; otherwise `html_source` is treated as
    the HTML content itself. A short str that happens to look like HTML is
    not a valid existing path, so it falls through to the content branch.
    """
    # PathLike -> always a path
    if isinstance(html_source, os.PathLike):
        with open(html_source, "rb") as f:
            return f.read()

    if isinstance(html_source, str):
        # An existing file path -> read it; anything else -> it IS the HTML.
        try:
            if os.path.isfile(html_source):
                with open(html_source, "rb") as f:
                    return f.read()
        except (OSError, ValueError):
            # e.g. the string is longer than the OS path limit -> it's content
            pass
        return html_source.encode("utf-8")

    raise GitHubPublishError(
        f"publish_html: unsupported html_source type {type(html_source).__name__}; "
        f"expected an HTML string or a file path."
    )


def publish_html(
    html_source,
    target: GitHubTarget,
    commit_message: str,
) -> str:
    """Publish HTML to GitHub, overwriting any existing version.

    Args:
        html_source: the HTML to publish. Accepts either:
            - the HTML content itself, as a `str` (in-memory publishing,
              no disk involved — used for cloud / scheduled runs); or
            - a path (`str` or `os.PathLike`) to an HTML file on disk.
            The two cases are told apart heuristically: a value that is an
            existing file path is read from disk, otherwise it is treated
            as the HTML content. To remove any ambiguity, pass an explicit
            `str` of HTML when publishing from memory.
        target: repository coordinates + token + retry policy.
        commit_message: message for the commit GitHub will create.

    Returns:
        The HTML URL of the commit created by the publish.

    Raises:
        GitHubPublishError: on any permanent failure (auth, missing repo)
            or after retries are exhausted on transient failures.
    """
    if not target.token:
        raise GitHubPublishError(
            "No GitHub token provided. Set GITHUB_TOKEN in your .env file."
        )

    # 1. Obtain the HTML bytes — from disk if `html_source` is an existing
    #    file path, otherwise treat `html_source` as the HTML content itself.
    html_bytes = _resolve_html_bytes(html_source)
    content_b64 = base64.b64encode(html_bytes).decode("ascii")

    # 2. Resolve the SHA of the existing file, if any.
    existing_sha = _get_existing_sha(target)

    # 3. Build the PUT payload.
    payload: dict[str, object] = {
        "message": commit_message,
        "content": content_b64,
        "branch": target.branch,
    }
    if existing_sha is not None:
        payload["sha"] = existing_sha       # required to overwrite

    # 4. Create or update the file.
    resp = _request_with_retry(
        "PUT",
        target.contents_url,
        max_retries=target.max_retries,
        headers=_auth_headers(target.token),
        json=payload,
    )

    if resp.status_code in (200, 201):
        commit = resp.json().get("commit", {})
        commit_url = commit.get("html_url", "(url unavailable)")
        action = "updated" if existing_sha else "created"
        logger.info("Published %s (%s). Commit: %s",
                    target.path, action, commit_url)
        return commit_url

    # Permanent failures: clear, actionable diagnostics.
    if resp.status_code == 401:
        raise GitHubPublishError(
            "GitHub authentication failed (HTTP 401). The token is missing, "
            "invalid, or expired."
        )
    if resp.status_code == 403:
        raise GitHubPublishError(
            "GitHub authorization failed (HTTP 403). The token likely lacks "
            "'Contents: write' permission on this repository."
        )
    if resp.status_code == 404:
        raise GitHubPublishError(
            "Repository or path not found (HTTP 404). Check owner/repo names "
            "and that the token has access to this repository."
        )
    if resp.status_code == 409:
        raise GitHubPublishError(
            "Conflict (HTTP 409) — the file changed between read and write. "
            "Re-run the publish step."
        )
    raise GitHubPublishError(
        f"GitHub publish failed: HTTP {resp.status_code} — {resp.text[:300]}"
    )
