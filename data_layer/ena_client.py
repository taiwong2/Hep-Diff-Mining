"""ENA/SRA metadata fetcher for repository cross-referencing.

Queries the ENA Portal API and NCBI SRA RunInfo to retrieve metadata
for BioProject (PRJEB/PRJNA) and SRA (SRP) accessions.

Usage:
    from data_layer.ena_client import fetch_ena_metadata, fetch_sra_metadata
    meta = fetch_ena_metadata("PRJEB12345")
    meta = fetch_sra_metadata("SRP123456")
"""

from __future__ import annotations

import json
import logging
import re
import time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"
ENA_FIELDS = (
    "sample_alias,experiment_title,library_strategy,library_source,"
    "instrument_model,read_count,fastq_ftp,submitted_ftp,scientific_name"
)

SRA_RUNINFO_URL = "https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi"

# ArrayExpress API (legacy, now via BioStudies)
AE_API = "https://www.ebi.ac.uk/biostudies/api/v1/studies"


def fetch_ena_metadata(accession: str) -> dict | None:
    """Fetch metadata from ENA Portal API for a BioProject/SRA accession.

    Returns dict with project info, sample list, library details, or None on failure.
    """
    url = (
        f"{ENA_API}?accession={accession}&result=read_run"
        f"&fields={ENA_FIELDS}&format=json"
    )
    req = Request(url, headers={"User-Agent": "CellDiffMining/1.0"})

    try:
        with urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("ENA fetch failed for %s: %s", accession, e)
        return None

    try:
        runs = json.loads(data)
    except json.JSONDecodeError:
        logger.warning("ENA returned non-JSON for %s", accession)
        return None

    if not runs:
        return None

    # Aggregate metadata
    sample_aliases = set()
    instruments = set()
    strategies = set()
    organisms = set()
    total_reads = 0

    for r in runs:
        alias = r.get("sample_alias", "")
        if alias:
            sample_aliases.add(alias)
        inst = r.get("instrument_model", "")
        if inst:
            instruments.add(inst)
        strat = r.get("library_strategy", "")
        if strat:
            strategies.add(strat)
        org = r.get("scientific_name", "")
        if org:
            organisms.add(org)
        try:
            total_reads += int(r.get("read_count", 0))
        except (ValueError, TypeError):
            pass

    # Classify data type from library strategy
    data_type = _classify_library_strategy(strategies)

    return {
        "accession": accession,
        "repository": "ENA",
        "sample_count": len(sample_aliases),
        "platform": ", ".join(sorted(instruments)) if instruments else None,
        "organism": ", ".join(sorted(organisms)) if organisms else None,
        "data_type": data_type,
        "has_processed_matrix": False,  # ENA has raw data only
        "sample_metadata": [
            {"sample_alias": a} for a in sorted(sample_aliases)
        ],
        "raw_response": data[:5000],  # truncate for storage
    }


def fetch_sra_metadata(accession: str) -> dict | None:
    """Fetch metadata from NCBI SRA RunInfo as a fallback.

    Works for SRP-prefixed accessions and sometimes for BioProject.
    """
    url = f"{SRA_RUNINFO_URL}?save=efetch&db=sra&rettype=runinfo&term={accession}"
    req = Request(url, headers={"User-Agent": "CellDiffMining/1.0"})

    try:
        with urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("SRA RunInfo fetch failed for %s: %s", accession, e)
        return None

    if not data.strip() or "Run" not in data.split("\n", 1)[0]:
        return None

    # Parse CSV-like output
    lines = data.strip().split("\n")
    if len(lines) < 2:
        return None

    headers = lines[0].split(",")
    samples = set()
    instruments = set()
    strategies = set()

    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split(",")
        row = dict(zip(headers, fields))
        sample = row.get("SampleName", "") or row.get("Sample", "")
        if sample:
            samples.add(sample)
        inst = row.get("Model", "")
        if inst:
            instruments.add(inst)
        strat = row.get("LibraryStrategy", "")
        if strat:
            strategies.add(strat)

    data_type = _classify_library_strategy(strategies)

    return {
        "accession": accession,
        "repository": "SRA",
        "sample_count": len(samples),
        "platform": ", ".join(sorted(instruments)) if instruments else None,
        "data_type": data_type,
        "has_processed_matrix": False,
        "raw_response": data[:5000],
    }


def fetch_arrayexpress_metadata(accession: str) -> dict | None:
    """Fetch metadata from ArrayExpress/BioStudies for E-MTAB accessions."""
    url = f"{AE_API}/{accession}"
    req = Request(url, headers={
        "User-Agent": "CellDiffMining/1.0",
        "Accept": "application/json",
    })

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("ArrayExpress fetch failed for %s: %s", accession, e)
        return None

    if not data:
        return None

    title = data.get("title", "")
    # Try to count samples from section data
    sample_count = 0
    for section in data.get("section", {}).get("subsections", []):
        if isinstance(section, dict) and section.get("type") == "Samples":
            links = section.get("links", [])
            sample_count = len(links) if isinstance(links, list) else 0

    return {
        "accession": accession,
        "repository": "ArrayExpress",
        "project_title": title,
        "sample_count": sample_count,
        "has_processed_matrix": False,
        "raw_response": json.dumps(data)[:5000],
    }


def check_geo_to_ena_link(gse_id: str) -> str | None:
    """Check if a GEO accession has a linked ENA/SRA BioProject.

    Returns the BioProject accession (PRJNA...) if found, else None.
    """
    # GEO stores SRA links; we can check via the GEO accession page
    url = (
        f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        f"?acc={gse_id}&targ=self&form=text&view=brief"
    )
    req = Request(url, headers={"User-Agent": "CellDiffMining/1.0"})

    try:
        with urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, TimeoutError):
        return None

    # Look for BioProject link in SOFT text
    for line in text.splitlines():
        if "BioProject" in line or "PRJNA" in line or "PRJEB" in line:
            match = re.search(r'(PRJNA\d+|PRJEB\d+)', line)
            if match:
                return match.group(1)

    return None


def _classify_library_strategy(strategies: set[str]) -> str:
    """Classify data type from ENA/SRA library strategy values."""
    strategies_lower = {s.lower() for s in strategies}

    if "rna-seq" in strategies_lower or "rna_seq" in strategies_lower:
        return "bulk_rnaseq"
    if any("10x" in s.lower() or "chromium" in s.lower() for s in strategies):
        return "scrna_seq"
    if "microarray" in strategies_lower or "other" in strategies_lower:
        return "microarray"
    if "atac-seq" in strategies_lower:
        return "atac_seq"
    if "chip-seq" in strategies_lower:
        return "chip_seq"

    return "unknown"
