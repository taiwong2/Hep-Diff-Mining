"""Download supplement files from PMC for triage-passed papers.

Uses the PMC OA FTP tar.gz package approach: queries the OA Service API
for the package URL, downloads the tar.gz, extracts only supplement files
(PDFs, Word docs, Excel, CSV — not images).

Storage:
    data/PMC76/PMC7612819_supp/
      ├── supplementary_table_1.xlsx
      ├── supplementary_methods.pdf
      └── supplementary_figures.pdf

Usage:
    python -m data_layer.fetch_supplements           # full run
    python -m data_layer.fetch_supplements --limit 5  # test
"""

from __future__ import annotations

import io
import logging
import os
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# File extensions to extract from tar.gz (skip large images)
WANTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".tsv",
    ".txt", ".rtf", ".xml",
}
SKIP_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".eps", ".nxml"}

PMC_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PMC_FTP_HTTPS = "https://ftp.ncbi.nlm.nih.gov/pub/pmc"
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
RATE_LIMIT_INTERVAL = 0.5  # seconds between requests


def _shard_dir(pmc_id: str) -> Path:
    """Get the sharded data directory for a PMC ID."""
    tag = pmc_id if pmc_id.startswith("PMC") else f"PMC{pmc_id}"
    prefix = tag[:5]
    return Path("data/db") / prefix


def supp_dir_for(pmc_id: str) -> Path:
    """Get the supplement directory path for a PMC ID."""
    tag = pmc_id if pmc_id.startswith("PMC") else f"PMC{pmc_id}"
    return _shard_dir(pmc_id) / f"{tag}_supp"


def extract_supplement_filenames(xml_path: str | Path) -> list[dict[str, str]]:
    """Parse <supplementary-material> elements from a PMC XML file.

    Returns list of dicts with 'filename', 'label', 'caption' for supplement
    files that should be extracted from the tar.gz package.
    """
    xml_path = Path(xml_path)
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        logger.warning("Cannot parse XML for supplements: %s", xml_path)
        return []

    root = tree.getroot()
    supplements: list[dict[str, str]] = []

    for supp in root.iter("supplementary-material"):
        href = supp.get("{http://www.w3.org/1999/xlink}href", "")
        if not href:
            href = supp.get("href", "")

        # Check child <media> elements for the href (common in PMC XMLs)
        if not href:
            media = supp.find("media")
            if media is not None:
                href = media.get("{http://www.w3.org/1999/xlink}href", "")
                if not href:
                    href = media.get("href", "")

        if not href:
            continue

        filename = href.split("/")[-1] if "/" in href else href
        ext = Path(filename).suffix.lower()

        if ext in SKIP_EXTENSIONS:
            continue

        label_el = supp.find("label")
        label = (label_el.text or "").strip() if label_el is not None else ""

        caption_el = supp.find(".//caption")
        caption = ""
        if caption_el is not None:
            caption = "".join(caption_el.itertext()).strip()

        supplements.append({
            "filename": filename,
            "extension": ext,
            "label": label,
            "caption": caption,
        })

    return supplements


def _get_oa_package_url(pmc_id: str) -> str | None:
    """Query PMC OA API for the tar.gz package URL."""
    try:
        resp = requests.get(
            PMC_OA_API,
            params={"id": pmc_id},
            timeout=15,
            headers={"User-Agent": f"CellDifferentiationMining/1.0 ({NCBI_EMAIL})"},
        )
        resp.raise_for_status()

        # Parse XML response
        root = ET.fromstring(resp.text)
        for link in root.iter("link"):
            href = link.get("href", "")
            if href.endswith(".tar.gz"):
                # Convert ftp:// to https://
                if href.startswith("ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/"):
                    return href.replace(
                        "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc",
                        PMC_FTP_HTTPS,
                    )
                return href
    except (requests.RequestException, ET.ParseError) as e:
        logger.warning("OA API error for %s: %s", pmc_id, e)
    return None


def fetch_supplements_for_paper(
    pmc_id: str, xml_path: str | Path
) -> dict[str, Any]:
    """Download supplement files for a paper from the PMC OA tar.gz package.

    Returns dict with 'pmc_id', 'supp_dir', 'files_downloaded', 'files_skipped'.
    """
    # Identify which filenames are supplements
    supp_info = extract_supplement_filenames(xml_path)
    wanted_filenames = {s["filename"] for s in supp_info}

    if not wanted_filenames:
        return {
            "pmc_id": pmc_id,
            "supp_dir": None,
            "files_downloaded": 0,
            "files_skipped": 0,
            "files": [],
        }

    # Get package URL from OA API
    pkg_url = _get_oa_package_url(pmc_id)
    if not pkg_url:
        logger.info("[%s] Not available via OA API (not open access?)", pmc_id)
        return {
            "pmc_id": pmc_id,
            "supp_dir": None,
            "files_downloaded": 0,
            "files_skipped": 0,
            "files": [],
        }

    # Download tar.gz
    try:
        headers = {"User-Agent": f"CellDifferentiationMining/1.0 ({NCBI_EMAIL})"}
        resp = requests.get(pkg_url, timeout=120, headers=headers)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("[%s] Failed to download package: %s", pmc_id, e)
        return {
            "pmc_id": pmc_id,
            "supp_dir": None,
            "files_downloaded": 0,
            "files_skipped": 0,
            "files": [],
        }

    # Extract wanted supplement files
    dest_dir = supp_dir_for(pmc_id)
    downloaded = 0
    skipped = 0
    files: list[dict] = []

    try:
        tf = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
    except tarfile.TarError as e:
        logger.warning("[%s] Failed to open tar.gz: %s", pmc_id, e)
        return {
            "pmc_id": pmc_id,
            "supp_dir": None,
            "files_downloaded": 0,
            "files_skipped": 0,
            "files": [],
        }

    for member in tf.getmembers():
        if not member.isfile():
            continue

        basename = Path(member.name).name
        ext = Path(basename).suffix.lower()

        # Check if this is a wanted supplement file by name match
        if basename in wanted_filenames:
            dest_path = dest_dir / basename
            if dest_path.exists() and dest_path.stat().st_size > 0:
                skipped += 1
                files.append({"filename": basename, "status": "exists"})
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(member)
            if f:
                dest_path.write_bytes(f.read())
                downloaded += 1
                files.append({"filename": basename, "status": "downloaded"})
            continue

        # Also grab any supplement-like files not referenced in XML
        # but present in the package with wanted extensions
        if ext in WANTED_EXTENSIONS and ext != ".xml":
            # Skip the main article PDF (usually named with "Article" in it)
            if "Article" in basename or basename.endswith(".nxml"):
                continue
            # Only grab files that look supplementary (MOESM, supp, supplement, etc)
            name_lower = basename.lower()
            if any(kw in name_lower for kw in ("moesm", "supp", "table_s", "figure_s",
                                                 "additional", "appendix")):
                dest_path = dest_dir / basename
                if dest_path.exists() and dest_path.stat().st_size > 0:
                    skipped += 1
                    continue

                dest_dir.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(member)
                if f:
                    dest_path.write_bytes(f.read())
                    downloaded += 1
                    files.append({"filename": basename, "status": "downloaded"})

    tf.close()

    return {
        "pmc_id": pmc_id,
        "supp_dir": str(dest_dir) if downloaded > 0 or any(
            f["status"] == "exists" for f in files
        ) else None,
        "files_downloaded": downloaded,
        "files_skipped": skipped,
        "files": files,
    }


def fetch_all_supplements(
    papers: list[dict],
    limit: int | None = None,
) -> list[dict]:
    """Fetch supplements for all papers that have XML files.

    Args:
        papers: List of paper dicts from DB (need 'pmc_id', 'xml_path').
        limit: Max papers to process (for testing).

    Returns list of result dicts.
    """
    results: list[dict] = []
    to_process = papers[:limit] if limit else papers

    for i, paper in enumerate(to_process):
        pmc_id = paper["pmc_id"]
        xml_path = paper.get("xml_path")
        if not xml_path or not Path(xml_path).exists():
            continue

        # Skip if supplement dir already exists and has files
        sd = supp_dir_for(pmc_id)
        if sd.exists() and any(sd.iterdir()):
            results.append({
                "pmc_id": pmc_id,
                "supp_dir": str(sd),
                "files_downloaded": 0,
                "files_skipped": 0,
                "files": [],
                "status": "already_exists",
            })
            continue

        logger.info("[%d/%d] Fetching supplements for %s",
                    i + 1, len(to_process), pmc_id)

        time.sleep(RATE_LIMIT_INTERVAL)
        result = fetch_supplements_for_paper(pmc_id, xml_path)
        results.append(result)

    return results


if __name__ == "__main__":
    import argparse
    import sys

    _PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from data_layer.database import PipelineDB

    parser = argparse.ArgumentParser(description="Fetch PMC supplement files")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    db = PipelineDB()
    papers = db.get_papers_needing_text()
    print(f"Papers eligible for supplement fetching: {len(papers)}")

    results = fetch_all_supplements(papers, limit=args.limit)

    fetched = sum(1 for r in results if r.get("files_downloaded", 0) > 0)
    total_files = sum(r.get("files_downloaded", 0) for r in results)
    print(f"\nPapers with new supplements: {fetched}")
    print(f"Total files downloaded: {total_files}")

    # Update DB with supplement dirs
    for r in results:
        if r.get("supp_dir"):
            paper = db.get_paper(pmc_id=r["pmc_id"])
            if paper:
                db.update_paper(paper["id"], supplement_dir=r["supp_dir"])

    db.close()
