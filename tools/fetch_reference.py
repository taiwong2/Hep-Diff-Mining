"""Tool: fetch_reference — fetch full text of an external paper not in corpus.

1. Check corpus_cache first (already fetched?)
2. Resolve DOI → PMC ID via NCBI ID converter API
3. Fetch PMC XML via PMC client
4. Convert to text via parse_pmc_xml_to_text()
5. Cache in corpus_cache table
6. Return text (60K char budget for references)

One hop max — the tool never chases references within fetched references.
"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

REFERENCE_TEXT_BUDGET = 60_000

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_reference",
        "description": (
            "Fetch the full text of a referenced paper that is not in the corpus. "
            "Use this when the paper being extracted says 'we followed the protocol "
            "described in [reference]' and search_corpus didn't find it. "
            "Provide the DOI of the referenced paper. Returns the paper's text "
            "or an error message if the paper cannot be fetched (e.g. behind paywall)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doi": {
                    "type": "string",
                    "description": "The DOI of the paper to fetch (e.g. '10.1002/hep.23354')",
                },
                "pmid": {
                    "type": "string",
                    "description": "PMID as fallback if DOI is not available",
                },
            },
            "required": ["doi"],
        },
    },
}


def execute(db, args: dict) -> str:
    """Execute fetch_reference tool. Returns JSON string with text or error."""
    doi = args.get("doi", "").strip()
    pmid = args.get("pmid", "").strip()

    if not doi and not pmid:
        return json.dumps({"error": "No DOI or PMID provided"})

    # 1. Check corpus_cache
    cached = db.get_cached_text(doi=doi if doi else None)
    if cached and cached.get("full_text"):
        text = cached["full_text"]
        if len(text) > REFERENCE_TEXT_BUDGET:
            text = text[:REFERENCE_TEXT_BUDGET] + "\n[... truncated ...]"
        return json.dumps({
            "title": cached.get("title", ""),
            "text": text,
            "source": "cache",
        })

    # 2. Check if already in our papers table
    paper = None
    if doi:
        paper = db.get_paper(doi=doi)
    if not paper and pmid:
        # Try by pmid
        row = db._conn.execute(
            "SELECT * FROM papers WHERE pmid = ?", (pmid,)
        ).fetchone()
        if row:
            paper = dict(row)

    if paper and paper.get("parsed_text_path"):
        # Already parsed — read the text
        try:
            text = Path(paper["parsed_text_path"]).read_text()
            if len(text) > REFERENCE_TEXT_BUDGET:
                text = text[:REFERENCE_TEXT_BUDGET] + "\n[... truncated ...]"
            return json.dumps({
                "title": paper.get("title", ""),
                "text": text,
                "source": "in_corpus",
            })
        except FileNotFoundError:
            pass

    # 3. Try to resolve DOI → PMC ID and fetch
    pmc_id = _resolve_doi_to_pmc(doi) if doi else None
    if not pmc_id and pmid:
        pmc_id = _resolve_pmid_to_pmc(pmid)

    if not pmc_id:
        return json.dumps({
            "error": f"Cannot resolve DOI '{doi}' to a PMC article. "
                     "The paper may not be in PMC Open Access.",
            "doi": doi,
        })

    # 4. Fetch and parse the XML
    text, title = _fetch_and_parse_pmc(pmc_id)
    if not text:
        return json.dumps({
            "error": f"Failed to fetch/parse PMC article {pmc_id}",
            "doi": doi,
            "pmc_id": pmc_id,
        })

    # 5. Cache it
    if len(text) > REFERENCE_TEXT_BUDGET:
        text = text[:REFERENCE_TEXT_BUDGET] + "\n[... truncated ...]"

    db.cache_text(doi=doi, pmc_id=pmc_id, title=title, text=text,
                  source="fetch_reference")

    return json.dumps({
        "title": title or "",
        "text": text,
        "source": "fetched",
        "pmc_id": pmc_id,
    })


def _resolve_doi_to_pmc(doi: str) -> str | None:
    """Resolve a DOI to PMC ID using NCBI ID converter API."""
    try:
        url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        resp = requests.get(url, params={
            "ids": doi,
            "format": "json",
            "tool": "CellDifferentiationMining",
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("records", [])
        if records:
            pmcid = records[0].get("pmcid", "")
            if pmcid:
                return pmcid
    except Exception as e:
        logger.debug("DOI->PMC resolution failed for %s: %s", doi, e)
    return None


def _resolve_pmid_to_pmc(pmid: str) -> str | None:
    """Resolve a PMID to PMC ID using NCBI ID converter."""
    try:
        url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        resp = requests.get(url, params={
            "ids": pmid,
            "format": "json",
            "tool": "CellDifferentiationMining",
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("records", [])
        if records:
            pmcid = records[0].get("pmcid", "")
            if pmcid:
                return pmcid
    except Exception as e:
        logger.debug("PMID->PMC resolution failed for %s: %s", pmid, e)
    return None


def _fetch_and_parse_pmc(pmc_id: str) -> tuple[str, str]:
    """Fetch PMC XML and parse to text.

    Returns (text, title) or ("", "") on failure.
    """
    from data_layer.pmc.pmc_client import PMCClient
    from data_layer.xml_to_text import parse_pmc_xml_to_text

    try:
        client = PMCClient()
        # Fetch XML
        raw_id = pmc_id.replace("PMC", "")
        raw_xml = client.fetch_xml([raw_id])
        if not raw_xml or len(raw_xml.strip()) < 100:
            return ("", "")

        # Write to temp location for parsing
        from data_layer.pmc.fetch_pmc_xmls import shard_path, extract_articles
        articles = extract_articles(raw_xml)

        if raw_id in articles:
            xml_str = articles[raw_id]
        elif pmc_id in articles:
            xml_str = articles[pmc_id]
        else:
            # Just try the whole XML
            xml_str = raw_xml

        # Write temp file for parse_pmc_xml_to_text
        out_path = shard_path(raw_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            out_path.write_text(xml_str, encoding="utf-8")

        parsed = parse_pmc_xml_to_text(out_path)
        if parsed and parsed.full_text:
            return (parsed.full_text, parsed.title)

    except Exception as e:
        logger.warning("Failed to fetch/parse PMC %s: %s", pmc_id, e)

    return ("", "")
