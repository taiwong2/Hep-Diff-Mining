"""Tool: search_corpus — search DB for already-extracted protocols.

Searches the protocols table (via JOIN with papers on title/DOI) and
corpus_cache. If query looks like a DOI, does exact match first, then
falls back to LIKE on paper title.

Returns JSON: up to 5 hits with paper_title, doi, protocol summary.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the database of already-extracted protocols for a specific "
            "paper or protocol. Use this when a paper references another protocol "
            "and you want to check if it has already been extracted. "
            "Query can be a DOI, author name + year, or title keywords."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query: a DOI (e.g. '10.1002/hep.23354'), "
                        "author + year (e.g. 'Si-Tayeb 2010'), or title keywords."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def execute(db, args: dict) -> str:
    """Execute search_corpus tool. Returns JSON string result."""
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "Empty query", "results": []})

    results = db.search_corpus(query, limit=5)

    if not results:
        # Also check corpus_cache
        cached = _search_cache(db, query)
        if cached:
            return json.dumps({"results": cached, "source": "corpus_cache"})
        return json.dumps({"results": [], "message": "No matching protocols found."})

    # Format results for the LLM
    formatted = []
    for r in results:
        entry: dict[str, Any] = {
            "paper_title": r.get("title", ""),
            "doi": r.get("doi", ""),
            "pmc_id": r.get("pmc_id", ""),
            "protocol_arm": r.get("protocol_arm", ""),
            "extraction_confidence": r.get("extraction_confidence"),
        }

        # Summarize stages
        stages = r.get("stages", [])
        if isinstance(stages, list):
            stage_summary = []
            for s in stages:
                if isinstance(s, dict):
                    name = s.get("stage_name", "unknown")
                    duration = s.get("duration_days", "?")
                    gfs = [gf.get("name", "") for gf in s.get("growth_factors", [])
                           if isinstance(gf, dict)]
                    sms = [sm.get("name", "") for sm in s.get("small_molecules", [])
                           if isinstance(sm, dict)]
                    factors = ", ".join(gfs + sms)
                    stage_summary.append(
                        f"{name} ({duration}d): {factors}" if factors
                        else f"{name} ({duration}d)"
                    )
            entry["stages_summary"] = stage_summary

        # Cell source
        cell_source = r.get("cell_source", {})
        if isinstance(cell_source, dict):
            entry["cell_source_type"] = cell_source.get("type", "")
            entry["cell_line"] = cell_source.get("line_name", "")

        formatted.append(entry)

    return json.dumps({"results": formatted})


def _search_cache(db, query: str) -> list[dict]:
    """Fallback search in corpus_cache table."""
    results = []

    # DOI exact match
    if query.startswith("10.") or "doi.org" in query:
        doi = query.replace("https://doi.org/", "").replace("http://doi.org/", "")
        cached = db.get_cached_text(doi=doi)
        if cached:
            results.append({
                "title": cached.get("title", ""),
                "doi": cached.get("doi", ""),
                "source": "corpus_cache",
                "text_available": True,
                "text_preview": (cached.get("full_text", "")[:500] + "...")
                    if cached.get("full_text") else "",
            })

    # Title search in cache
    if not results:
        rows = db._conn.execute(
            "SELECT doi, pmc_id, title FROM corpus_cache WHERE title LIKE ? LIMIT 5",
            (f"%{query}%",),
        ).fetchall()
        for r in rows:
            results.append({
                "title": r["title"],
                "doi": r["doi"],
                "pmc_id": r["pmc_id"],
                "source": "corpus_cache",
                "text_available": True,
            })

    return results
