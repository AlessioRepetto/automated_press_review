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
from datetime import datetime, timezone

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser

from scripts.config import REQUEST_TIMEOUT, TZ_ROME, USER_AGENT


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
    # Performs a GET request with a custom User-Agent
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content


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
