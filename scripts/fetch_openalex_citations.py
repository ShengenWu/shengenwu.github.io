#!/usr/bin/env python3
"""Fetch OpenAlex author citation counts and write a static JSON file."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


OPENALEX_API = "https://api.openalex.org/authors"
DEFAULT_AUTHOR = "orcid:0009-0001-3432-3583"
DEFAULT_AUTHOR_SEARCH = "Shengen Wu"
SELECT_FIELDS = "id,display_name,orcid,cited_by_count,counts_by_year,works_count,summary_stats,updated_date"


def normalize_author_identifier(value: str) -> str:
    value = value.strip()
    if value.startswith("https://openalex.org/"):
        return value.rstrip("/").split("/")[-1]
    if value.startswith("https://orcid.org/"):
        return "orcid:" + value.rstrip("/").split("/")[-1]
    return value


def fetch_author(author_identifier: str, api_key: str) -> dict:
    author_identifier = normalize_author_identifier(author_identifier)
    query = urllib.parse.urlencode({
        "select": SELECT_FIELDS,
        "api_key": api_key,
    })
    url = f"{OPENALEX_API}/{urllib.parse.quote(author_identifier, safe=':')}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "shengenwu.github.io citation updater"})

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise LookupError(f"OpenAlex has no author record for {author_identifier}.") from exc
        raise RuntimeError(f"OpenAlex request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAlex request failed: {exc}") from exc


def search_authors(search_name: str, api_key: str, limit: int = 5) -> list[dict]:
    query = urllib.parse.urlencode({
        "search": search_name,
        "select": "id,display_name,orcid,cited_by_count,works_count,last_known_institutions",
        "per-page": limit,
        "api_key": api_key,
    })
    request = urllib.request.Request(
        f"{OPENALEX_API}?{query}",
        headers={"User-Agent": "shengenwu.github.io citation updater"},
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAlex author search failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAlex author search failed: {exc}") from exc

    return data.get("results") or []


def format_candidate(candidate: dict) -> str:
    institutions = candidate.get("last_known_institutions") or []
    institution_names = ", ".join(item.get("display_name", "") for item in institutions if item.get("display_name"))
    details = [
        f"id={candidate.get('id')}",
        f"name={candidate.get('display_name')}",
        f"orcid={candidate.get('orcid')}",
        f"works={candidate.get('works_count')}",
        f"citations={candidate.get('cited_by_count')}",
    ]
    if institution_names:
        details.append(f"institutions={institution_names}")
    return " | ".join(details)


def contiguous_counts(counts_by_year: list[dict], years: int) -> list[dict]:
    current_year = dt.date.today().year
    by_year = {int(item["year"]): item for item in counts_by_year if item.get("year")}
    start_year = current_year - years + 1
    rows = []

    for year in range(start_year, current_year + 1):
        row = by_year.get(year, {})
        rows.append({
            "year": year,
            "works_count": int(row.get("works_count") or 0),
            "cited_by_count": int(row.get("cited_by_count") or 0),
        })

    return rows


def build_output(author: dict, author_identifier: str, years: int) -> dict:
    today = dt.date.today().isoformat()
    return {
        "source": "OpenAlex",
        "updated_at": today,
        "author_identifier": normalize_author_identifier(author_identifier),
        "author_id": author.get("id"),
        "display_name": author.get("display_name"),
        "orcid": author.get("orcid"),
        "works_count": author.get("works_count"),
        "total_citations": int(author.get("cited_by_count") or 0),
        "summary_stats": author.get("summary_stats") or {},
        "counts_by_year": contiguous_counts(author.get("counts_by_year") or [], years),
        "is_sample_data": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", default=os.getenv("OPENALEX_AUTHOR_ID", DEFAULT_AUTHOR))
    parser.add_argument("--search-name", default=os.getenv("OPENALEX_AUTHOR_SEARCH_NAME", DEFAULT_AUTHOR_SEARCH))
    parser.add_argument("--output", default="assets/data/openalex-citations.json")
    parser.add_argument("--years", type=int, default=10)
    args = parser.parse_args()

    api_key = os.getenv("OPENALEX_API_KEY")
    if not api_key:
        print("OPENALEX_API_KEY is required. Create a free key at https://openalex.org/settings/api.", file=sys.stderr)
        return 2

    try:
        author = fetch_author(args.author, api_key)
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        print("", file=sys.stderr)
        print("This usually means the OpenAlex author profile is not linked to that ORCID yet.", file=sys.stderr)
        print("Find the right OpenAlex author below, then set a repository variable named OPENALEX_AUTHOR_ID to its A... ID.", file=sys.stderr)
        print("", file=sys.stderr)
        candidates = search_authors(args.search_name, api_key)
        if candidates:
            print(f"Top OpenAlex author search results for {args.search_name!r}:", file=sys.stderr)
            for candidate in candidates:
                print(f"- {format_candidate(candidate)}", file=sys.stderr)
        else:
            print(f"No OpenAlex author candidates found for {args.search_name!r}.", file=sys.stderr)
        return 3

    output = build_output(author, args.author, args.years)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} for {output.get('display_name')} ({output.get('author_id')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
