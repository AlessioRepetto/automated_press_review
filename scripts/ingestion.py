# RSS ingestion.
#
# Feed definitions are read from a CSV file with columns `source, section, url`.
# Each feed is downloaded, parsed through `feedparser`, and HTML in summaries
# is cleaned with `BeautifulSoup`. Dates are normalized to the Europe/Rome
# timezone.
#
# Deduplication is based on the concatenation `source + title + description`.

import csv
import logging
import time
from datetime import datetime, timezone

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

from scripts.config import REQUEST_TIMEOUT, TZ_ROME, USER_AGENT

logger = logging.getLogger(__name__)

# Browser-like request headers.
#
# Some sites reject RSS requests coming from cloud/datacenter IPs when the
# User-Agent openly identifies a bot (e.g. "AnalisiNewsBot/1.0"). Sending a
# realistic browser User-Agent — plus the Accept / Accept-Language headers a
# real browser always sends — works around the "soft" form of that block.
#
# This does NOT defeat a block based purely on the IP range: if a server
# refuses datacenter IPs regardless of headers, no User-Agent will help.
# The configured USER_AGENT is kept as a fallback for the retry below.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/rss+xml,*/*;q=0.8"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}

# Per-request retry for transient rejections (intermittent 403, 5xx, network
# errors). A datacenter 403 is sometimes intermittent — a second attempt a
# few seconds later can succeed.
_HTTP_MAX_ATTEMPTS = 3
_HTTP_BACKOFF_SECONDS = 4


def read_feeds_csv(csv_path):
    # Reads feed definitions from CSV.
    # Returns a list of tuples (source, section, url).
    feeds = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source  = (row.get("source")  or "").strip()
            section = (row.get("section") or "").strip()
            url     = (row.get("url")     or "").strip()
            if source and section and url:
                feeds.append((source, section, url))
    return feeds


def http_get(url):
    # Performs a GET request with browser-like headers and a small retry.
    #
    # Behaviour:
    #   - Sends _BROWSER_HEADERS (realistic browser User-Agent + Accept
    #     headers) to get past "soft" anti-bot filters that reject obvious
    #     bot User-Agents coming from datacenter IPs.
    #   - Retries up to _HTTP_MAX_ATTEMPTS times on transient failures
    #     (network errors, HTTP 5xx, and 403 — which can be intermittent
    #     from cloud IPs), waiting _HTTP_BACKOFF_SECONDS between attempts.
    #   - On a 404 / 410 (the resource is genuinely gone) it does NOT retry:
    #     it raises immediately, since retrying a missing feed is pointless.
    #
    # On definitive failure the underlying requests exception is raised, so
    # the caller behaves exactly as before (this change does not alter the
    # pipeline's fail-fast behaviour — it only makes the request itself
    # more robust).
    last_exc = None

    for attempt in range(1, _HTTP_MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(
                url, headers=_BROWSER_HEADERS, timeout=REQUEST_TIMEOUT
            )
            # Genuinely-gone resources: do not retry, fail straight away.
            if resp.status_code in (404, 410):
                resp.raise_for_status()
            # Transient HTTP errors (incl. 403): retry if attempts remain.
            if resp.status_code == 403 or resp.status_code >= 500:
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.content

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_exc = e
            # 404 / 410 -> permanent, stop immediately.
            if status in (404, 410):
                raise
            # Other HTTP errors -> retry if attempts remain.
            if attempt < _HTTP_MAX_ATTEMPTS:
                logger.warning(
                    "http_get %s -> HTTP %s, retry %d/%d in %ds",
                    url, status, attempt, _HTTP_MAX_ATTEMPTS,
                    _HTTP_BACKOFF_SECONDS,
                )
                time.sleep(_HTTP_BACKOFF_SECONDS)
                continue
            raise

        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < _HTTP_MAX_ATTEMPTS:
                logger.warning(
                    "http_get %s -> %s, retry %d/%d in %ds",
                    url, type(e).__name__, attempt, _HTTP_MAX_ATTEMPTS,
                    _HTTP_BACKOFF_SECONDS,
                )
                time.sleep(_HTTP_BACKOFF_SECONDS)
                continue
            raise

    # Defensive: the loop always returns or raises, but just in case.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"http_get failed for {url} with no captured error")


def parse_datetime_local(entry):
    # Extracts the publication date from a feed entry dict.
    # Tries 'published', 'updated', 'pubDate' in order; otherwise returns the current UTC time.
    # Always returns a timezone-aware datetime converted to Europe/Rome.
    for k in ("published", "updated", "pubDate"):
        val = entry.get(k)
        if val:
            try:
                dt = dtparser.parse(val)
                if not dt.tzinfo:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(TZ_ROME)
            except Exception:
                pass
    return datetime.now(timezone.utc).astimezone(TZ_ROME)


def normalise_feed_whitespace(text):
    # Normalises whitespace in feed text:
    #   - Replaces non-breaking spaces (\xa0 / &nbsp;) with regular spaces
    #   - Collapses runs of whitespace into single spaces
    # Used during RSS ingestion to keep titles and descriptions clean after
    # BeautifulSoup tag removal, which can leave behind nbsp characters
    # and stray multiple spaces.
    return " ".join(text.replace("\xa0", " ").split())


def fetch_feed_entries(source, section, url):
    # Downloads and parses a single RSS / Atom feed.
    # Removes HTML from summaries through BeautifulSoup.
    # Normalises non-breaking spaces (\xa0 / &nbsp;) into regular spaces.
    # I use separator=" " in get_text() to prevent removed HTML tags
    # from merging adjacent words (e.g. <b>word</b>next would become "wordnext").
    content = http_get(url)
    parsed  = feedparser.parse(content)
    if parsed.bozo:
        logging.warning("Malformed feed: %s (%s)", section, url)

    results = []
    for e in getattr(parsed, "entries", []) or []:
        results.append({
            "source":      source,
            "section":     section,
            "title":       normalise_feed_whitespace((e.get("title") or "").strip()),
            "description": normalise_feed_whitespace(BeautifulSoup(
                (e.get("summary") or e.get("description") or ""), "html.parser"
            ).get_text(separator=" ")),
            "link":        (e.get("link") or "").strip(),
            "guid":        (e.get("id")   or e.get("guid") or "").strip(),
            "dt_local":    parse_datetime_local(e),
        })
    return results


def build_news_dataframe(feeds):
    # Downloads all feeds, flattens them into a DataFrame, deduplicates,
    # and sorts by date descending.
    # Deduplication key: source + title + description.
    all_entries = []
    for source, section, url in feeds:
        logging.info("Fetching feed: %-20s %s", section, url)
        all_entries.extend(fetch_feed_entries(source, section, url))

    rows = []
    for e in all_entries:
        dt = e["dt_local"]
        rows.append({
            "source":      e["source"],
            "section":     e["section"],
            "date":        dt.strftime("%Y-%m-%d"),
            "time":        dt.strftime("%H:%M:%S"),
            "title":       e["title"],
            "description": e["description"],
            "link":        e["link"],
            "guid":        e["guid"],
        })

    df = pd.DataFrame(rows, columns=[
        "source", "section", "date", "time", "title", "description", "link", "guid"
    ])

    dup_flag = (df["source"] + " " + df["title"] + " " + df["description"]).duplicated(keep="first")
    df.drop(df[dup_flag].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if not df.empty:
        df = df.sort_values(["date", "time"], ascending=[False, False]).reset_index(drop=True)
    return df
