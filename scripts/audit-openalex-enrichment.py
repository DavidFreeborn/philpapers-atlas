#!/usr/bin/env python3
"""Add an OpenAlex pass to the saved Crossref enrichment pilot.

Only records without an accepted Crossref match are queried, keeping the pilot
within the unauthenticated API's daily allowance. The same conservative title,
author, and year acceptance rule is applied.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "philpapers-atlas-metadata-audit/0.1 (https://github.com/davidfreeborn/philpapers-atlas)"


def normalized(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def source_surname(value: str) -> str:
    if "," in value:
        return normalized(value.split(",", 1)[0])
    words = normalized(value).split()
    return words[-1] if words else ""


def type_compatible(source_type: str, candidate_type: str, source: str) -> bool:
    candidate = normalized(candidate_type).replace(" ", "-")
    if source_type == "Book":
        return candidate in {"book", "monograph", "reference-book", "edited-book"}
    if source_type == "Review":
        return candidate in {"article", "journal-article", "review", "peer-review"}
    if source_type == "Article":
        return candidate in {
            "article", "journal-article", "book-chapter", "proceedings-article",
            "posted-content", "preprint", "dissertation",
        }
    return False


def conservative_acceptance(row: dict, match: dict, source: str) -> bool:
    if source == "OpenAlex" and "philpapers" in normalized(match.get("journal", "")):
        return False
    source_year = match.get("sourceYear")
    candidate_year = match.get("candidateYear")
    year_ok = (
        source_year is None
        or candidate_year is None
        or abs(int(source_year) - int(candidate_year)) <= 2
    )
    author_ok = bool(match.get("authorMatch")) or (
        int(match.get("authorCount", 0)) == 0
        and float(match.get("titleSimilarity", 0)) == 1
        and source_year is not None
        and candidate_year is not None
        and int(source_year) == int(candidate_year)
    )
    candidate_type = match.get("openAlexType") if source == "OpenAlex" else match.get("crossrefType")
    return (
        float(match.get("titleSimilarity", 0)) >= 0.94
        and author_ok
        and year_ok
        and type_compatible(row["sourceType"], str(candidate_type or ""), source)
    )


def openalex_items(title: str) -> list[dict]:
    parameters = urlencode(
        {
            "filter": f'title.search:"{title}"',
            "per-page": 5,
            "select": "id,doi,title,publication_year,authorships,primary_location,type",
        }
    )
    request = Request(
        f"https://api.openalex.org/works?{parameters}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                return json.load(response).get("results", [])
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


def score(row: dict, candidate: dict) -> dict:
    title = str(candidate.get("title") or "")
    similarity = SequenceMatcher(None, normalized(row["title"]), normalized(title)).ratio()
    surname = source_surname(row["creator"])
    candidate_surnames = set()
    for authorship in candidate.get("authorships") or []:
        words = normalized((authorship.get("author") or {}).get("display_name", "")).split()
        if words:
            candidate_surnames.add(words[-1])
    author_match = bool(surname and surname in candidate_surnames)
    source_year = row.get("match", {}).get("sourceYear") if row.get("match") else None
    if source_year is None:
        match = re.search(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)", row.get("date", ""))
        source_year = int(match.group(1)) if match else None
    candidate_year = candidate.get("publication_year")
    year_match = (
        source_year is not None
        and candidate_year is not None
        and abs(int(source_year) - int(candidate_year)) <= 1
    )
    total_score = similarity * 0.84 + float(author_match) * 0.10 + float(year_match) * 0.06
    location = candidate.get("primary_location") or {}
    source = location.get("source") or {}
    result = {
        "candidateTitle": title,
        "titleSimilarity": round(similarity, 4),
        "authorMatch": author_match,
        "sourceYear": source_year,
        "candidateYear": candidate_year,
        "yearMatch": year_match,
        "score": round(total_score, 4),
        "accepted": False,
        "openAlexId": candidate.get("id", ""),
        "doi": candidate.get("doi", ""),
        "journal": source.get("display_name", "") or "",
        "publisher": source.get("host_organization_name", "") or "",
        "openAlexType": candidate.get("type", "") or "",
        "authorCount": len(candidate.get("authorships") or []),
    }
    result["accepted"] = conservative_acceptance(row, result, "OpenAlex")
    return result


def audit(row: dict) -> dict | None:
    candidates = [score(row, candidate) for candidate in openalex_items(row["title"])]
    return max(candidates, key=lambda candidate: candidate["score"], default=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "analysis" / "metadata-enrichment-pilot.json",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for row in data["results"]:
        if row.get("match"):
            row["match"]["accepted"] = conservative_acceptance(row, row["match"], "Crossref")
    unmatched = [row for row in data["results"] if not (row.get("match") and row["match"]["accepted"])]
    pending = [row for row in unmatched if "openAlexMatch" not in row]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(audit, row): row for row in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            futures[future]["openAlexMatch"] = future.result()
            print(f"[{index:03d}/{len(pending):03d}] {futures[future]['id']}", flush=True)

    combined = []
    for row in data["results"]:
        if row.get("openAlexMatch"):
            row["openAlexMatch"]["accepted"] = conservative_acceptance(row, row["openAlexMatch"], "OpenAlex")
        if row.get("match") and row["match"]["accepted"]:
            row["combinedMatch"] = {"source": "Crossref", **row["match"]}
        elif row.get("openAlexMatch") and row["openAlexMatch"]["accepted"]:
            row["combinedMatch"] = {"source": "OpenAlex", **row["openAlexMatch"]}
        else:
            row["combinedMatch"] = None
        if row["combinedMatch"]:
            combined.append(row)

    crossref_accepted = [
        row for row in data["results"]
        if row.get("match") and row["match"]["accepted"]
    ]
    data["summary"] = {
        "sample": len(data["results"]),
        "accepted": len(crossref_accepted),
        "acceptedRate": len(crossref_accepted) / len(data["results"]),
        "journalAmongAccepted": sum(bool(row["match"]["journal"]) for row in crossref_accepted),
        "publisherAmongAccepted": sum(bool(row["match"]["publisher"]) for row in crossref_accepted),
        "multiAuthorAmongAccepted": sum(row["match"]["authorCount"] > 1 for row in crossref_accepted),
        "bySourceType": {
            source_type: {
                "sample": len(rows),
                "accepted": sum(bool(row.get("match") and row["match"]["accepted"]) for row in rows),
            }
            for source_type in sorted({row["sourceType"] for row in data["results"]})
            if (rows := [row for row in data["results"] if row["sourceType"] == source_type])
        },
    }

    data["method"]["secondPass"] = "OpenAlex title search for records without an accepted Crossref match"
    data["method"]["acceptance"] = "title similarity ≥ .94; author surname required when candidate authors exist; year within two years; compatible publication type; circular PhilPapers records excluded"
    data["combinedSummary"] = {
        "sample": len(data["results"]),
        "accepted": len(combined),
        "acceptedRate": len(combined) / len(data["results"]),
        "bySource": dict(Counter(row["combinedMatch"]["source"] for row in combined)),
        "journalAmongAccepted": sum(bool(row["combinedMatch"]["journal"]) for row in combined),
        "publisherAmongAccepted": sum(bool(row["combinedMatch"]["publisher"]) for row in combined),
        "multiAuthorAmongAccepted": sum(row["combinedMatch"]["authorCount"] > 1 for row in combined),
    }
    args.input.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data["combinedSummary"], indent=2))


if __name__ == "__main__":
    main()
