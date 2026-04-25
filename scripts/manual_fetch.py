"""Manual PDF fetch workflow for behind-paywall references.

Three subcommands:
  show        — Generate manifest of needed references + create drop folder
  ingest      — Process PDFs dropped into data/manual_pdfs/
  re-extract  — Re-run extraction on papers whose references are now cached

Usage:
    python manual_fetch.py show
    python manual_fetch.py ingest
    python manual_fetch.py re-extract [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_layer.database import PipelineDB

logger = logging.getLogger(__name__)

MANUAL_PDFS_DIR = Path("data/manual_pdfs")
MANIFEST_PATH = MANUAL_PDFS_DIR / "MANIFEST.json"


# ------------------------------------------------------------------
# show — Generate manifest of needed references
# ------------------------------------------------------------------

def cmd_show(db: PipelineDB) -> None:
    """Query protocols with needs_manual_fetch=1, deduplicate references,
    write MANIFEST.json and print summary."""

    rows = db._conn.execute(
        """SELECT p.id AS protocol_id, p.paper_id, p.base_protocol_doi,
                  p.incomplete_flags, pa.pmc_id, pa.title AS paper_title,
                  pa.xml_path
           FROM protocols p
           JOIN papers pa ON pa.id = p.paper_id
           WHERE p.needs_manual_fetch = 1"""
    ).fetchall()

    if not rows:
        print("No protocols with needs_manual_fetch flag.")
        return

    # Build a map: reference_key -> reference info
    # We key by DOI when available, otherwise by a description slug
    ref_map: dict[str, dict] = {}

    for row in rows:
        row = dict(row)
        protocol_id = row["protocol_id"]
        paper_id = row["paper_id"]
        pmc_id = row["pmc_id"] or ""
        base_doi = (row["base_protocol_doi"] or "").strip()

        # Parse incomplete_flags for behind_paywall_reference entries
        flags = []
        if row["incomplete_flags"]:
            try:
                flags = json.loads(row["incomplete_flags"])
            except (json.JSONDecodeError, TypeError):
                pass

        paywall_flags = [
            f for f in flags
            if isinstance(f, dict) and f.get("reason") == "behind_paywall_reference"
        ]

        if not paywall_flags and not base_doi:
            # No actionable reference info — skip
            continue

        # Extract DOIs mentioned in flag details text
        flag_dois = set()
        flag_descriptions = []
        xml_path = row.get("xml_path")
        for flag in paywall_flags:
            details = flag.get("details", "")
            flag_descriptions.append(details)
            # Try to extract DOI from details text
            doi_match = re.search(r'(10\.\d{4,}/[^\s,;)\]]+)', details)
            if doi_match:
                flag_dois.add(doi_match.group(1).rstrip('.'))
            else:
                # Try to resolve author+year from flag against XML ref-list
                resolved = _resolve_flag_to_doi(details, xml_path)
                if resolved:
                    flag_dois.add(resolved)

        # If base_protocol_doi is set, that's the primary reference
        if base_doi:
            flag_dois.add(base_doi)

        # If we found DOIs, create entries keyed by DOI
        if flag_dois:
            for doi in flag_dois:
                key = doi
                if key not in ref_map:
                    ref_map[key] = {
                        "doi": doi,
                        "description": "",
                        "suggested_filename": _doi_to_filename(doi),
                        "protocol_ids": [],
                        "paper_pmc_ids": [],
                        "status": "pending",
                        "ingested_at": None,
                    }
                if protocol_id not in ref_map[key]["protocol_ids"]:
                    ref_map[key]["protocol_ids"].append(protocol_id)
                if pmc_id and pmc_id not in ref_map[key]["paper_pmc_ids"]:
                    ref_map[key]["paper_pmc_ids"].append(pmc_id)
                # Use the first flag description that mentions this DOI
                if not ref_map[key]["description"] and flag_descriptions:
                    ref_map[key]["description"] = flag_descriptions[0]
        else:
            # No DOI found — create entry keyed by description slug
            desc = flag_descriptions[0] if flag_descriptions else "unknown reference"
            slug = _description_to_slug(desc)
            key = f"no_doi__{slug}"
            if key not in ref_map:
                ref_map[key] = {
                    "doi": None,
                    "description": desc,
                    "suggested_filename": f"{slug}.pdf",
                    "protocol_ids": [],
                    "paper_pmc_ids": [],
                    "status": "pending",
                    "ingested_at": None,
                }
            if protocol_id not in ref_map[key]["protocol_ids"]:
                ref_map[key]["protocol_ids"].append(protocol_id)
            if pmc_id and pmc_id not in ref_map[key]["paper_pmc_ids"]:
                ref_map[key]["paper_pmc_ids"].append(pmc_id)

    if not ref_map:
        print("No behind-paywall references found in incomplete flags.")
        return

    # Assign sequential IDs and build manifest — preserve status from prior run
    old_status: dict[str, dict] = {}
    if MANIFEST_PATH.exists():
        try:
            old_manifest = json.loads(MANIFEST_PATH.read_text())
            for old_ref in old_manifest.get("references", []):
                key_doi = old_ref.get("doi")
                key_fname = old_ref.get("suggested_filename", "").lower()
                if key_doi:
                    old_status[key_doi] = old_ref
                if key_fname:
                    old_status[key_fname] = old_ref
        except (json.JSONDecodeError, TypeError):
            pass

    references = []
    for i, (key, ref) in enumerate(sorted(ref_map.items()), start=1):
        ref["id"] = f"ref_{i:03d}"
        # Restore ingestion status from previous manifest
        prev = old_status.get(ref.get("doi") or "") or old_status.get(
            ref.get("suggested_filename", "").lower(), {}
        )
        if prev.get("status") == "ingested":
            ref["status"] = "ingested"
            ref["ingested_at"] = prev.get("ingested_at")
            ref["ingested_filename"] = prev.get("ingested_filename")
            ref["text_length"] = prev.get("text_length")
        references.append(ref)

    n_pending = sum(1 for r in references if r["status"] == "pending")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_references_needed": len(references),
        "pending": n_pending,
        "ingested": len(references) - n_pending,
        "references": references,
    }

    # Write manifest
    MANUAL_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Print summary
    pending_refs = [r for r in references if r["status"] == "pending"]
    ingested_refs = [r for r in references if r["status"] == "ingested"]

    print(f"\nManifest written to {MANIFEST_PATH}")
    print(f"Total unique references: {len(references)} "
          f"({len(pending_refs)} pending, {len(ingested_refs)} already ingested)")
    print(f"Pending protocols affected: "
          f"{sum(len(r['protocol_ids']) for r in pending_refs)}")
    print(f"\nDrop PDFs into: {MANUAL_PDFS_DIR}/")
    print()
    print(f"{'ID':<10} {'DOI':<35} {'Protocols':<10} {'Suggested Filename'}")
    print("-" * 95)
    for ref in pending_refs:
        doi_display = ref["doi"] or "(no DOI)"
        n_protos = len(ref["protocol_ids"])
        line = f"{ref['id']:<10} {doi_display:<35} {n_protos:<10} {ref['suggested_filename']}"
        if ref["doi"]:
            line += f"  https://doi.org/{ref['doi']}"
        print(line)

    if ingested_refs:
        print(f"\n({len(ingested_refs)} already-ingested references hidden)")

    # Show top pending references by protocol count
    by_count = sorted(pending_refs, key=lambda r: len(r["protocol_ids"]), reverse=True)
    print(f"\nTop pending references by protocol count:")
    for ref in by_count[:10]:
        doi_display = ref["doi"] or "(no DOI)"
        desc_short = ref["description"][:80] if ref["description"] else ""
        link = f"  https://doi.org/{ref['doi']}" if ref["doi"] else ""
        print(f"  {ref['id']} — {doi_display} ({len(ref['protocol_ids'])} protocols){link}")
        if desc_short:
            print(f"    {desc_short}")


# ------------------------------------------------------------------
# ingest — Process dropped PDFs
# ------------------------------------------------------------------

def cmd_ingest(db: PipelineDB) -> None:
    """Match PDFs in manual_pdfs/ to manifest entries, convert to text,
    store in corpus_cache."""

    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}. Run 'show' first.")
        return

    manifest = json.loads(MANIFEST_PATH.read_text())
    references = manifest.get("references", [])

    if not references:
        print("Manifest has no references.")
        return

    # Find all PDFs in the drop folder
    pdfs = list(MANUAL_PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {MANUAL_PDFS_DIR}/")
        print("Download PDFs and place them there with the suggested filenames.")
        return

    print(f"Found {len(pdfs)} PDF(s) in {MANUAL_PDFS_DIR}/")

    # Build lookup: suggested_filename -> reference
    by_filename: dict[str, dict] = {}
    for ref in references:
        by_filename[ref["suggested_filename"].lower()] = ref
        # Also index by DOI-based variations
        if ref.get("doi"):
            # Allow the raw DOI with / replaced by __
            alt = ref["doi"].replace("/", "__") + ".pdf"
            by_filename[alt.lower()] = ref

    ingested = 0
    skipped = 0

    for pdf_path in sorted(pdfs):
        fname = pdf_path.name.lower()

        # Try exact match first
        ref = by_filename.get(fname)

        # Try fuzzy: strip common prefixes, match DOI substring
        if not ref:
            ref = _fuzzy_match_pdf(pdf_path.name, references)

        if not ref:
            print(f"  SKIP {pdf_path.name} — no matching manifest entry")
            skipped += 1
            continue

        if ref["status"] == "ingested":
            print(f"  SKIP {pdf_path.name} — already ingested (ref {ref['id']})")
            skipped += 1
            continue

        print(f"  Processing {pdf_path.name} → ref {ref['id']} ({ref.get('doi', 'no DOI')})...",
              end=" ", flush=True)

        # Convert PDF to text
        from data_layer.supplement_processor import _process_pdf
        text = _process_pdf(pdf_path)

        if not text or len(text.strip()) < 100:
            print("FAILED (no usable text extracted)")
            continue

        print(f"OK ({len(text)} chars)")

        # Store in corpus_cache
        db.cache_text(
            doi=ref.get("doi"),
            pmc_id=None,
            title=None,
            text=text,
            source="manual_pdf",
        )

        # Update manifest entry
        ref["status"] = "ingested"
        ref["ingested_at"] = datetime.now(timezone.utc).isoformat()
        ref["ingested_filename"] = pdf_path.name
        ref["text_length"] = len(text)
        ingested += 1

    # Save updated manifest
    manifest["references"] = references
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nIngested: {ingested}, Skipped: {skipped}")
    if ingested > 0:
        print(f"Manifest updated at {MANIFEST_PATH}")
        print(f"\nNext step: python manual_fetch.py re-extract")


# ------------------------------------------------------------------
# re-extract — Re-run extraction on affected papers
# ------------------------------------------------------------------

def cmd_re_extract(db: PipelineDB, limit: int | None = None) -> None:
    """Re-extract protocols for papers whose behind-paywall references
    are now in corpus_cache."""

    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}. Run 'show' and 'ingest' first.")
        return

    manifest = json.loads(MANIFEST_PATH.read_text())
    references = manifest.get("references", [])

    # Find ingested references
    ingested_refs = [r for r in references if r["status"] == "ingested"]
    if not ingested_refs:
        print("No references have been ingested yet. Run 'ingest' first.")
        return

    # Collect protocol_ids and paper_ids that can be re-extracted
    protocol_ids_to_reextract: set[int] = set()
    for ref in ingested_refs:
        protocol_ids_to_reextract.update(ref["protocol_ids"])

    # Get the paper_ids for these protocols
    if not protocol_ids_to_reextract:
        print("No protocols to re-extract.")
        return

    placeholders = ",".join("?" for _ in protocol_ids_to_reextract)
    rows = db._conn.execute(
        f"SELECT DISTINCT paper_id FROM protocols WHERE id IN ({placeholders})",
        list(protocol_ids_to_reextract),
    ).fetchall()
    paper_ids = [row[0] for row in rows]

    if limit:
        paper_ids = paper_ids[:limit]

    if not paper_ids:
        print("No papers to re-extract.")
        return

    print(f"Re-extracting {len(paper_ids)} paper(s) "
          f"({len(protocol_ids_to_reextract)} protocols affected)...")

    # For each paper: delete old protocols, reset status, re-extract
    from llm.openrouter.client import OpenRouterClient
    from llm.agents.agentic_extractor import extract_paper, DEFAULT_OUTPUT

    async def _run():
        async with OpenRouterClient(max_concurrent=3) as client:
            for i, paper_id in enumerate(paper_ids):
                paper = db.get_paper(paper_id=paper_id)
                if not paper:
                    print(f"  Paper ID {paper_id} not found, skipping.")
                    continue

                pmc_id = paper["pmc_id"]
                title = paper.get("title", "")[:60]
                print(f"[{i+1}/{len(paper_ids)}] {pmc_id}: {title}...")

                # Safety check: verify parsed text exists BEFORE deleting
                text_path = paper.get("parsed_text_path")
                if not text_path or not Path(text_path).exists():
                    print(f"  SKIP — no parsed text at {text_path}")
                    continue

                # Delete old protocols for this paper
                deleted = db.delete_protocols_for_paper(paper_id)
                print(f"  Deleted {deleted} old protocol(s)")

                # Reset extraction status
                db.set_extraction_status(paper_id, "pending")

                # Re-extract
                db.set_extraction_status(paper_id, "in_progress")
                try:
                    protocols = await extract_paper(client, db, paper)

                    if protocols:
                        for protocol in protocols:
                            # Check if any remaining behind_paywall flags
                            flags = protocol.get("incomplete_flags", [])
                            has_paywall = any(
                                isinstance(f, dict) and f.get("reason") == "behind_paywall_reference"
                                for f in flags
                            )
                            protocol["needs_manual_fetch"] = has_paywall

                            proto_id = db.store_protocol(paper_id, protocol)

                            # Set needs_manual_fetch on the protocol row
                            if has_paywall:
                                db._conn.execute(
                                    "UPDATE protocols SET needs_manual_fetch = 1 WHERE id = ?",
                                    (proto_id,)
                                )
                                db._conn.commit()

                            # Write to JSONL
                            record = {
                                "pmc_id": pmc_id,
                                "doi": paper.get("doi"),
                                "title": paper.get("title"),
                                "protocol_arm": protocol.get("protocol_arm"),
                                "extraction_confidence": protocol.get("extraction_confidence"),
                                "stages_count": len(protocol.get("stages") or []),
                                "incomplete_flags": protocol.get("incomplete_flags", []),
                                "re_extracted": True,
                            }
                            _append_jsonl(DEFAULT_OUTPUT, record)

                        db.set_extraction_status(paper_id, "completed")
                        print(f"  Extracted {len(protocols)} protocol(s)")
                    else:
                        db.set_extraction_status(paper_id, "failed")
                        print(f"  Extraction failed (no protocols returned)")

                except Exception as e:
                    logger.exception("Error re-extracting %s", pmc_id)
                    db.set_extraction_status(paper_id, "failed")
                    db.log_processing(paper_id, "re_extraction", "failed",
                                      error_message=str(e))
                    print(f"  Error: {e}")

    asyncio.run(_run())

    print("\nRe-extraction complete.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Cache parsed ref-lists per xml_path to avoid re-parsing
_ref_list_cache: dict[str, list[dict]] = {}


def _resolve_flag_to_doi(details: str, xml_path: str | None) -> str | None:
    """Try to resolve an author+year flag description to a DOI via XML ref-list.

    Parses patterns like "Si-Tayeb et al. 2010", "Song and Bhatt 2015",
    "Hannan 2013", and bracket refs like "[17]" from the flag text, then
    matches against the citing paper's structured <ref-list>.
    """
    if not xml_path or not Path(xml_path).exists():
        return None

    # Parse author surname + year from flag description
    # Pattern 1: "Surname et al. YYYY" or "Surname et al., YYYY"
    # Pattern 2: "Surname and Surname YYYY"
    # Pattern 3: "Surname YYYY" (bare)
    author_year = None
    m = re.search(
        r'([A-Z][a-z][A-Za-z-]*)'          # first author surname
        r'(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z][A-Za-z-]*))?'  # optional et al./and
        r'[\s,()]*'                          # separator
        r'(\d{4})',                           # year
        details,
    )
    if m:
        author_year = (m.group(1), m.group(2))

    # Parse bracket reference number: [17], ref. 17, reference 17
    bracket_num = None
    bm = re.search(r'\[(\d+)\]|ref\.?\s*(\d+)|reference\s+(\d+)', details, re.IGNORECASE)
    if bm:
        bracket_num = next(g for g in bm.groups() if g is not None)

    if not author_year and not bracket_num:
        return None

    # Load ref-list (cached)
    if xml_path not in _ref_list_cache:
        from data_layer.xml_to_text import extract_ref_list
        _ref_list_cache[xml_path] = extract_ref_list(xml_path)

    ref_list = _ref_list_cache[xml_path]
    if not ref_list:
        return None

    # Match by author surname + year
    if author_year:
        surname_query, year_query = author_year
        matches = []
        for r in ref_list:
            r_surname = r.get("first_author_surname") or ""
            r_year = r.get("year") or ""
            if (r_year == year_query
                    and r_surname
                    and surname_query.lower() in r_surname.lower()):
                matches.append(r)
        if len(matches) == 1 and matches[0].get("doi"):
            return matches[0]["doi"]

    # Fallback: match by bracket label number
    if bracket_num:
        for r in ref_list:
            if r.get("label") == bracket_num and r.get("doi"):
                return r["doi"]

    return None


def _doi_to_filename(doi: str) -> str:
    """Convert a DOI to a safe filename: 10.1002/hep.23354 -> 10.1002__hep.23354.pdf"""
    return doi.replace("/", "__") + ".pdf"


def _description_to_slug(desc: str) -> str:
    """Extract a human-readable slug from a flag description.

    Tries to find author names and years like 'Si-Tayeb et al. 2010'.
    Falls back to a truncated alphanumeric slug.
    """
    # Try author + year pattern
    match = re.search(r'([A-Z][a-z-]+(?:\s+et\s+al\.?)?)\s*[\[(]?\s*(\d{4})', desc)
    if match:
        author = match.group(1).strip().rstrip(".")
        year = match.group(2)
        slug = re.sub(r'[^a-z0-9]+', '_', f"{author}_{year}".lower()).strip("_")
        return slug

    # Fallback: first 40 chars, alphanumeric only
    slug = re.sub(r'[^a-z0-9]+', '_', desc[:40].lower()).strip("_")
    return slug or "unknown_ref"


def _fuzzy_match_pdf(filename: str, references: list[dict]) -> dict | None:
    """Try to fuzzy-match a PDF filename to a manifest reference.

    Checks if the filename contains a DOI slug or description slug.
    """
    fname_lower = filename.lower().replace(".pdf", "")

    for ref in references:
        if ref.get("doi"):
            doi_slug = ref["doi"].replace("/", "__").lower()
            # Check if the DOI slug appears in the filename
            if doi_slug in fname_lower:
                return ref
            # Also check without the prefix (e.g., just 'hep.23354')
            doi_suffix = ref["doi"].split("/", 1)[-1].lower() if "/" in ref["doi"] else ""
            if doi_suffix and doi_suffix in fname_lower:
                return ref

        # Check suggested filename (without extension)
        suggested = ref["suggested_filename"].lower().replace(".pdf", "")
        if suggested in fname_lower or fname_lower in suggested:
            return ref

    return None


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual PDF fetch workflow for behind-paywall references",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # show
    subparsers.add_parser("show", help="Generate manifest of needed references")

    # ingest
    subparsers.add_parser("ingest", help="Process PDFs dropped into manual_pdfs/")

    # re-extract
    re_extract_parser = subparsers.add_parser(
        "re-extract", help="Re-run extraction on papers with newly cached references"
    )
    re_extract_parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of papers to re-extract"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db = PipelineDB()
    try:
        if args.command == "show":
            cmd_show(db)
        elif args.command == "ingest":
            cmd_ingest(db)
        elif args.command == "re-extract":
            cmd_re_extract(db, limit=args.limit)
    finally:
        db.close()


if __name__ == "__main__":
    main()
