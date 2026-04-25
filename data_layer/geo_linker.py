"""GEO accession discovery via 3-strategy waterfall.

Strategy 1a: Mine PMC XML for GEO accessions (GSE/GSM/GDS/ArrayExpress)
Strategy 1b: Mine supplement text for accessions
Strategy 2:  Two-hop elink (PubMed → GDS) via NCBI EDirect
Strategy 3:  GEO SOFT validation and metadata fetch

Orchestrator runs all strategies, deduplicates, and computes confidence.

Usage:
    from data_layer.geo_linker import discover_geo_for_paper
    result = discover_geo_for_paper(pmc_id, xml_path, pmid, supp_text_path, pmc_client)
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# Regex for GEO / ArrayExpress accessions
GEO_ACCESSION_RE = re.compile(r'\b(GSE\d{4,8}|GSM\d{4,8}|GDS\d{3,6}|E-[A-Z]{4}-\d+)\b')

# Context classification keywords
DATA_AVAIL_KEYWORDS = re.compile(
    r'deposited\s+in|uploaded\s+to|submitted\s+to|available\s+(at|from|in|under)|'
    r'data\s+availability|accession\s+(number|code|id)|gene\s+expression\s+omnibus|'
    r'GEO\s+database|ArrayExpress',
    re.IGNORECASE,
)
REFERENCED_KEYWORDS = re.compile(
    r'reanalyz|downloaded\s+from|previously\s+published|obtained\s+from|'
    r'publicly\s+available\s+data|retrieved\s+from',
    re.IGNORECASE,
)


@dataclass
class AccessionHit:
    """A single GEO accession found in text."""
    accession: str
    context: str  # own_data | referenced | ambiguous
    source: str   # xml | supplement | elink
    surrounding_text: str = ""


@dataclass
class GEOSampleMeta:
    """Per-sample metadata from SOFT."""
    gsm_id: str
    sample_title: str = ""
    source_name: str = ""
    description: str = ""
    characteristics: dict = field(default_factory=dict)
    sra_accession: str = ""


@dataclass
class GEOSeriesMeta:
    """Series-level metadata from SOFT."""
    gse_id: str
    title: str = ""
    summary: str = ""
    data_type: str = ""
    platform: str = ""
    sample_count: int = 0
    linked_pmids: list[str] = field(default_factory=list)
    submission_date: str = ""
    organism: str = ""
    samples: list[GEOSampleMeta] = field(default_factory=list)


@dataclass
class GEODiscoveryResult:
    """Aggregated result from all discovery strategies."""
    pmc_id: str
    hits: list[AccessionHit] = field(default_factory=list)
    series_meta: dict[str, GEOSeriesMeta] = field(default_factory=dict)
    strategies_used: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Strategy 1a — XML text mining
# ------------------------------------------------------------------

def mine_accessions_from_xml(xml_path: str | Path) -> list[AccessionHit]:
    """Parse PMC XML and extract GEO accessions with context classification."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return []

    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        logger.warning("XML parse error for %s: %s", xml_path, e)
        return []

    root = tree.getroot()
    hits: list[AccessionHit] = []
    seen: set[str] = set()

    def _get_text_content(elem: ET.Element) -> str:
        """Get all text within an element including tails."""
        return "".join(elem.itertext())

    def _classify_element(elem: ET.Element, ancestors: list[ET.Element]) -> str:
        """Classify context by examining element and its ancestors."""
        # Check if inside ref-list
        for anc in ancestors:
            tag = anc.tag.split("}")[-1] if "}" in anc.tag else anc.tag
            if tag in ("ref-list", "ref", "mixed-citation", "element-citation"):
                return "referenced"

        # Check if inside data-availability section
        for anc in ancestors:
            tag = anc.tag.split("}")[-1] if "}" in anc.tag else anc.tag
            if tag == "sec":
                sec_type = anc.get("sec-type", "")
                if "data-availability" in sec_type or "data_availability" in sec_type:
                    return "own_data"
                # Check section title
                title_elem = anc.find("title")
                if title_elem is not None:
                    title_text = (title_elem.text or "").lower()
                    if any(kw in title_text for kw in
                           ("data availability", "data access", "accession",
                            "data deposition")):
                        return "own_data"

        # Check surrounding text for context keywords
        context_text = _get_text_content(elem) if elem.text else ""
        # Also check parent text
        for anc in ancestors[-2:]:
            context_text += " " + _get_text_content(anc)

        if DATA_AVAIL_KEYWORDS.search(context_text):
            return "own_data"
        if REFERENCED_KEYWORDS.search(context_text):
            return "referenced"

        return "ambiguous"

    def _walk_with_ancestors(elem: ET.Element, ancestors: list[ET.Element]) -> None:
        """Walk the XML tree tracking ancestor chain."""
        # Check all text in this element
        all_text = ""
        if elem.text:
            all_text += elem.text
        if elem.tail:
            all_text += " " + elem.tail

        for match in GEO_ACCESSION_RE.finditer(all_text):
            acc = match.group(1)
            if acc not in seen:
                seen.add(acc)
                context = _classify_element(elem, ancestors)
                # Get surrounding text (100 chars each side)
                start = max(0, match.start() - 100)
                end = min(len(all_text), match.end() + 100)
                surrounding = all_text[start:end].strip()
                hits.append(AccessionHit(
                    accession=acc,
                    context=context,
                    source="xml",
                    surrounding_text=surrounding,
                ))

        for child in elem:
            _walk_with_ancestors(child, ancestors + [elem])

    _walk_with_ancestors(root, [])
    return hits


# ------------------------------------------------------------------
# Strategy 1b — Supplement text mining
# ------------------------------------------------------------------

def mine_accessions_from_supplement(supp_text_path: str | Path) -> list[AccessionHit]:
    """Scan supplement text for GEO accessions."""
    supp_path = Path(supp_text_path)
    if not supp_path.exists():
        return []

    text = supp_path.read_text()
    if not text.strip():
        return []

    hits: list[AccessionHit] = []
    seen: set[str] = set()

    for match in GEO_ACCESSION_RE.finditer(text):
        acc = match.group(1)
        if acc not in seen:
            seen.add(acc)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            surrounding = text[start:end].strip()
            hits.append(AccessionHit(
                accession=acc,
                context="own_data",  # supplements are the paper's own material
                source="supplement",
                surrounding_text=surrounding,
            ))

    return hits


# ------------------------------------------------------------------
# Strategy 2 — Two-hop elink (PubMed → GDS)
# ------------------------------------------------------------------

def elink_pubmed_to_gds(pmid: str, pmc_client) -> list[AccessionHit]:
    """Use NCBI E-utilities HTTPS API to find GDS records linked to a PubMed ID.

    Two HTTPS calls:
    1. elink.fcgi  (PubMed → GDS) → linked GDS record IDs
    2. esummary.fcgi (GDS docsums) → GSE accessions
    """
    if not pmid:
        return []

    # Step 1: elink PubMed → GDS
    try:
        linked_ids = pmc_client.elink_https("pubmed", "gds", [pmid])
    except Exception as e:
        logger.warning("elink failed for PMID %s: %s", pmid, e)
        return []

    if not linked_ids:
        return []

    # Step 2: esummary to get GSE accessions
    try:
        docsum_xml = pmc_client.esummary_https("gds", linked_ids)
    except Exception as e:
        logger.warning("esummary GDS failed for PMID %s: %s", pmid, e)
        return []

    # Step 3: parse docsummaries for GSE accessions
    hits: list[AccessionHit] = []
    seen: set[str] = set()

    try:
        doc_root = ET.fromstring(f"<root>{docsum_xml}</root>")
    except ET.ParseError:
        doc_root = None

    if doc_root is not None:
        for doc in doc_root.iter("DocumentSummary"):
            accession = doc.findtext("Accession", "")
            if accession and accession.startswith("GSE") and accession not in seen:
                seen.add(accession)
                title = doc.findtext("title", "") or doc.findtext("Title", "")
                hits.append(AccessionHit(
                    accession=accession,
                    context="own_data",
                    source="elink",
                    surrounding_text=title,
                ))

    # Also regex-scan the raw output for GSE IDs
    for match in GEO_ACCESSION_RE.finditer(docsum_xml):
        acc = match.group(1)
        if acc.startswith("GSE") and acc not in seen:
            seen.add(acc)
            hits.append(AccessionHit(
                accession=acc,
                context="own_data",
                source="elink",
            ))

    return hits


# ------------------------------------------------------------------
# Strategy 3 — GEO SOFT validation and metadata fetch
# ------------------------------------------------------------------

def _fetch_soft(gse_id: str, view: str = "brief", targ: str = "self") -> str | None:
    """Fetch SOFT format text from GEO for a series accession."""
    url = (
        f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        f"?acc={gse_id}&targ={targ}&form=text&view={view}"
    )
    req = Request(url, headers={"User-Agent": "CellDiffMining/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("SOFT fetch failed for %s: %s", gse_id, e)
        return None


def parse_soft_brief(soft_text: str) -> GEOSeriesMeta:
    """Parse brief SOFT format to extract series metadata."""
    meta = GEOSeriesMeta(gse_id="")

    for line in soft_text.splitlines():
        line = line.strip()
        if line.startswith("!Series_geo_accession"):
            meta.gse_id = line.split("=", 1)[-1].strip()
        elif line.startswith("!Series_title"):
            meta.title = line.split("=", 1)[-1].strip()
        elif line.startswith("!Series_summary"):
            summary_part = line.split("=", 1)[-1].strip()
            meta.summary = (meta.summary + " " + summary_part).strip() if meta.summary else summary_part
        elif line.startswith("!Series_type"):
            meta.data_type = line.split("=", 1)[-1].strip()
        elif line.startswith("!Series_platform_id"):
            meta.platform = line.split("=", 1)[-1].strip()
        elif line.startswith("!Series_sample_id"):
            meta.sample_count += 1
        elif line.startswith("!Series_pubmed_id"):
            pmid = line.split("=", 1)[-1].strip()
            if pmid:
                meta.linked_pmids.append(pmid)
        elif line.startswith("!Series_submission_date"):
            meta.submission_date = line.split("=", 1)[-1].strip()
        elif line.startswith("!Series_organism"):
            organism_part = line.split("=", 1)[-1].strip()
            # Multiple organism lines possible; keep first non-empty
            if organism_part and not meta.organism:
                meta.organism = organism_part

    return meta


def parse_soft_full(soft_text: str) -> list[GEOSampleMeta]:
    """Parse full SOFT format to extract per-sample metadata."""
    samples: list[GEOSampleMeta] = []
    current_sample: dict | None = None

    for line in soft_text.splitlines():
        line = line.strip()

        if line.startswith("^SAMPLE"):
            if current_sample and current_sample.get("gsm_id"):
                samples.append(GEOSampleMeta(**current_sample))
            gsm_id = line.split("=", 1)[-1].strip()
            current_sample = {"gsm_id": gsm_id, "characteristics": {}}

        elif current_sample is not None:
            if line.startswith("!Sample_title"):
                current_sample["sample_title"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_source_name"):
                current_sample["source_name"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_description"):
                current_sample["description"] = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_characteristics_ch"):
                val = line.split("=", 1)[-1].strip()
                if ":" in val:
                    k, v = val.split(":", 1)
                    current_sample["characteristics"][k.strip()] = v.strip()
                else:
                    current_sample["characteristics"][val] = val
            elif line.startswith("!Sample_relation"):
                val = line.split("=", 1)[-1].strip()
                if "SRA:" in val or "SRX" in val:
                    # Extract SRA accession
                    sra_match = re.search(r'(SRX\d+|SRR\d+)', val)
                    if sra_match:
                        current_sample["sra_accession"] = sra_match.group(1)

    # Don't forget last sample
    if current_sample and current_sample.get("gsm_id"):
        samples.append(GEOSampleMeta(**current_sample))

    return samples


def validate_and_fetch_soft(
    gse_id: str, paper_pmid: str | None = None
) -> tuple[GEOSeriesMeta | None, str]:
    """Validate a GSE accession via SOFT and determine context.

    Returns (series_meta, context) where context is 'own_data' if
    the paper's PMID appears in the SOFT linked_pmids.
    """
    brief = _fetch_soft(gse_id, view="brief")
    if not brief or "could not be found" in brief.lower():
        return None, "ambiguous"

    meta = parse_soft_brief(brief)
    if not meta.gse_id:
        return None, "ambiguous"

    # Determine context from PMID linkage
    context = "ambiguous"
    if paper_pmid and paper_pmid in meta.linked_pmids:
        context = "own_data"

    # Fetch per-sample metadata (targ=gsm returns individual sample records)
    time.sleep(0.5)  # Be polite to NCBI
    full = _fetch_soft(gse_id, view="brief", targ="gsm")
    if full:
        meta.samples = parse_soft_full(full)
        meta.sample_count = max(meta.sample_count, len(meta.samples))

    return meta, context


# ------------------------------------------------------------------
# Supplementary file check (for Phase 2 cross-referencing)
# ------------------------------------------------------------------

def check_geo_supplementary_files(gse_id: str) -> list[dict]:
    """Check GEO SOFT for supplementary file URLs and classify them.

    Returns list of {filename, url, file_type, has_count_matrix} dicts.
    Identifies count matrix files (.txt.gz, .csv.gz, .xlsx with gene names).
    """
    soft = _fetch_soft(gse_id, view="brief")
    if not soft:
        return []

    files: list[dict] = []
    count_matrix_patterns = re.compile(
        r'(count|expression|matrix|tpm|fpkm|rpkm|normalized|raw_count)',
        re.IGNORECASE,
    )
    matrix_extensions = {".txt.gz", ".csv.gz", ".tsv.gz", ".xlsx", ".csv", ".txt"}

    for line in soft.splitlines():
        if not line.startswith("!Series_supplementary_file"):
            continue

        url = line.split("=", 1)[-1].strip()
        if not url:
            continue

        # Extract filename from URL
        filename = url.rsplit("/", 1)[-1] if "/" in url else url

        # Determine file type
        lower_name = filename.lower()
        has_count_matrix = False

        if any(lower_name.endswith(ext) for ext in matrix_extensions):
            if count_matrix_patterns.search(lower_name):
                has_count_matrix = True
            # Also check for common gene expression file naming
            if any(kw in lower_name for kw in ("gene", "rna", "deg", "deseq", "edger")):
                has_count_matrix = True

        file_type = "unknown"
        if lower_name.endswith((".tar.gz", ".tar")):
            file_type = "archive"
        elif lower_name.endswith((".txt.gz", ".txt", ".tsv.gz", ".tsv")):
            file_type = "text_table"
        elif lower_name.endswith((".csv.gz", ".csv")):
            file_type = "csv"
        elif lower_name.endswith((".xlsx", ".xls")):
            file_type = "excel"
        elif lower_name.endswith((".cel.gz", ".cel")):
            file_type = "microarray_raw"

        files.append({
            "filename": filename,
            "url": url,
            "file_type": file_type,
            "has_count_matrix": has_count_matrix,
        })

    return files


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

def discover_geo_for_paper(
    pmc_id: str,
    xml_path: str | Path | None,
    pmid: str | None,
    supplement_text_path: str | Path | None,
    pmc_client=None,
) -> GEODiscoveryResult:
    """Run all 3 discovery strategies for a paper.

    Returns a GEODiscoveryResult with all hits and series metadata.
    """
    result = GEODiscoveryResult(pmc_id=pmc_id)

    # Strategy 1a: XML text mining
    if xml_path:
        xml_hits = mine_accessions_from_xml(xml_path)
        result.hits.extend(xml_hits)
        if xml_hits:
            result.strategies_used.append("xml_mining")
            logger.info("[%s] XML mining found %d accession(s)", pmc_id, len(xml_hits))

    # Strategy 1b: Supplement text mining
    if supplement_text_path:
        supp_hits = mine_accessions_from_supplement(supplement_text_path)
        result.hits.extend(supp_hits)
        if supp_hits:
            result.strategies_used.append("supplement_mining")
            logger.info("[%s] Supplement mining found %d accession(s)", pmc_id, len(supp_hits))

    # Strategy 2: elink
    if pmid and pmc_client:
        elink_hits = elink_pubmed_to_gds(pmid, pmc_client)
        result.hits.extend(elink_hits)
        if elink_hits:
            result.strategies_used.append("elink")
            logger.info("[%s] elink found %d accession(s)", pmc_id, len(elink_hits))

    # Deduplicate by accession, keeping best context
    gse_hits = _deduplicate_hits(result.hits)

    # Strategy 3: SOFT validation for GSE accessions
    for acc, hit in gse_hits.items():
        if not acc.startswith("GSE"):
            continue
        logger.info("[%s] Fetching SOFT for %s...", pmc_id, acc)
        meta, soft_context = validate_and_fetch_soft(acc, pmid)
        if meta:
            # Upgrade context if SOFT confirms ownership
            if soft_context == "own_data":
                hit.context = "own_data"
            result.series_meta[acc] = meta
            if "soft_validation" not in result.strategies_used:
                result.strategies_used.append("soft_validation")
        time.sleep(0.5)  # Rate limit NCBI requests

    # Replace hits with deduplicated version
    result.hits = list(gse_hits.values())

    return result


def _deduplicate_hits(hits: list[AccessionHit]) -> dict[str, AccessionHit]:
    """Deduplicate hits by accession, keeping best context.

    Priority: own_data > referenced > ambiguous.
    Multiple sources strengthen confidence.
    """
    context_priority = {"own_data": 3, "referenced": 2, "ambiguous": 1}
    merged: dict[str, AccessionHit] = {}

    for hit in hits:
        acc = hit.accession
        if acc not in merged:
            merged[acc] = hit
        else:
            existing = merged[acc]
            # Keep better context
            if context_priority.get(hit.context, 0) > context_priority.get(existing.context, 0):
                merged[acc] = AccessionHit(
                    accession=acc,
                    context=hit.context,
                    source=f"{existing.source}+{hit.source}",
                    surrounding_text=hit.surrounding_text or existing.surrounding_text,
                )
            else:
                merged[acc].source = f"{existing.source}+{hit.source}"

    return merged


def compute_confidence(hit: AccessionHit, strategies_used: list[str]) -> float:
    """Compute confidence score based on context and strategy agreement."""
    base = 0.5

    # Context boost
    if hit.context == "own_data":
        base = 0.8
    elif hit.context == "referenced":
        base = 0.6

    # Multi-source boost
    sources = hit.source.split("+")
    if len(sources) >= 2:
        base = min(base + 0.1, 1.0)

    # SOFT validation boost
    if "soft_validation" in strategies_used:
        base = min(base + 0.05, 1.0)

    return round(base, 2)


# ------------------------------------------------------------------
# Batch runner
# ------------------------------------------------------------------

def discover_geo_all(db, pmc_client=None, limit: int | None = None) -> int:
    """Run GEO discovery on all eligible papers. Returns count of papers with GEO."""
    papers = db.get_papers_needing_geo()
    if limit:
        papers = papers[:limit]

    if not papers:
        logger.info("No papers need GEO discovery")
        return 0

    logger.info("Running GEO discovery on %d papers", len(papers))
    found = 0

    for i, paper in enumerate(papers):
        pmc_id = paper["pmc_id"]
        paper_id = paper["id"]

        result = discover_geo_for_paper(
            pmc_id=pmc_id,
            xml_path=paper.get("xml_path"),
            pmid=paper.get("pmid"),
            supplement_text_path=paper.get("supplement_text_path"),
            pmc_client=pmc_client,
        )

        # Filter to GSE-only hits (series level)
        gse_hits = [h for h in result.hits if h.accession.startswith("GSE")]

        if gse_hits:
            for hit in gse_hits:
                confidence = compute_confidence(hit, result.strategies_used)
                meta = result.series_meta.get(hit.accession)

                accession_data = {
                    "gse_id": hit.accession,
                    "context": hit.context,
                    "confidence": confidence,
                    "discovery_strategies": result.strategies_used,
                    "soft_fetched": meta is not None,
                }

                if meta:
                    accession_data.update({
                        "data_type": meta.data_type,
                        "platform": meta.platform,
                        "sample_count": meta.sample_count,
                        "series_title": meta.title,
                        "series_summary": meta.summary[:2000] if meta.summary else None,
                        "linked_pmids": meta.linked_pmids,
                        "submission_date": meta.submission_date,
                        "organism": meta.organism,
                    })

                acc_id = db.store_geo_accession(paper_id, accession_data)

                # Store per-sample metadata if available
                if meta and meta.samples:
                    for sample in meta.samples:
                        db.store_geo_sample(acc_id, {
                            "gsm_id": sample.gsm_id,
                            "sample_title": sample.sample_title,
                            "source_name": sample.source_name,
                            "description": sample.description,
                            "characteristics": sample.characteristics,
                            "sra_accession": sample.sra_accession,
                        })

            db.update_paper(paper_id, geo_status="linked")
            found += 1
            logger.info("[%d/%d] %s: found %d GEO series", i + 1, len(papers),
                        pmc_id, len(gse_hits))
        else:
            db.update_paper(paper_id, geo_status="checked_none")
            logger.info("[%d/%d] %s: no GEO data found", i + 1, len(papers), pmc_id)

    logger.info("GEO discovery complete: %d/%d papers linked", found, len(papers))
    return found
