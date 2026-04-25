"""3-tier accession grounding: verify, validate, and re-discover GEO accessions.

Tier 1 — Text verification: check if each DB accession appears in paper XML/supplement text.
Tier 2 — API validation: validate accessions via GEO SOFT API and check PMID linkage.
Tier 3 — Re-discovery: re-scan XML and supplement text for papers that lost accessions.

Usage:
    python3 ground_accessions.py                   # full run
    python3 ground_accessions.py --dry-run         # report only, no DB changes
    python3 ground_accessions.py --single PMC...   # single paper
    python3 ground_accessions.py --tier 1          # only text verification
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from data_layer.database import PipelineDB
from data_layer.geo_linker import (
    GEO_ACCESSION_RE,
    compute_confidence,
    mine_accessions_from_xml,
    mine_accessions_from_supplement,
    validate_and_fetch_soft,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_JSONL = Path("data/logs/accession_grounding_results.jsonl")


def _append_jsonl(record: dict) -> None:
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------
# Tier 1 — Text verification
# ------------------------------------------------------------------

def tier1_text_verification(
    accessions: list[dict], dry_run: bool = False
) -> dict:
    """Check if each accession appears in the paper's XML or supplement text.

    Returns summary dict with counts.
    """
    verified = 0
    hallucinated = 0
    no_xml = 0
    details: list[dict] = []

    for acc in accessions:
        gse_id = acc["gse_id"]
        xml_path = acc.get("xml_path")
        supp_text_path = acc.get("supplement_text_path")
        acc_id = acc["id"]
        pmc_id = acc.get("pmc_id", "?")

        found_in_xml = False
        found_in_supp = False

        if xml_path and Path(xml_path).exists():
            xml_text = Path(xml_path).read_text(errors="replace")
            if gse_id in xml_text:
                found_in_xml = True
        else:
            no_xml += 1

        if not found_in_xml and supp_text_path and Path(supp_text_path).exists():
            supp_text = Path(supp_text_path).read_text(errors="replace")
            if gse_id in supp_text:
                found_in_supp = True

        if found_in_xml or found_in_supp:
            verified += 1
            status = "verified"
            source = "xml" if found_in_xml else "supplement"
            logger.debug("[%s] %s: found in %s", pmc_id, gse_id, source)
        else:
            hallucinated += 1
            status = "not_in_text"
            source = None
            logger.info("[%s] %s: NOT found in text — likely hallucinated", pmc_id, gse_id)

        detail = {
            "acc_id": acc_id,
            "gse_id": gse_id,
            "pmc_id": pmc_id,
            "paper_id": acc["paper_id"],
            "tier1_status": status,
            "found_source": source,
        }
        details.append(detail)
        _append_jsonl({"tier": 1, **detail})

    summary = {
        "verified": verified,
        "hallucinated": hallucinated,
        "no_xml": no_xml,
        "total": len(accessions),
    }
    logger.info("Tier 1 summary: %s", summary)
    return summary


# ------------------------------------------------------------------
# Tier 2 — API validation
# ------------------------------------------------------------------

def tier2_api_validation(
    accessions: list[dict], dry_run: bool = False
) -> dict:
    """Validate accessions via GEO SOFT API. Check PMID linkage.

    Only runs on accessions that passed Tier 1 (found in text).
    Returns summary dict.
    """
    valid = 0
    invalid = 0
    own_data_confirmed = 0
    details: list[dict] = []

    for acc in accessions:
        gse_id = acc["gse_id"]
        pmid = acc.get("pmid")
        acc_id = acc["id"]
        pmc_id = acc.get("pmc_id", "?")

        logger.info("[%s] Tier 2: validating %s via SOFT...", pmc_id, gse_id)
        meta, context = validate_and_fetch_soft(gse_id, pmid)

        if meta is None:
            invalid += 1
            status = "invalid_gse"
            logger.info("[%s] %s: does NOT exist in GEO", pmc_id, gse_id)
        else:
            valid += 1
            status = "valid"
            if context == "own_data":
                own_data_confirmed += 1
            logger.info(
                "[%s] %s: valid (context=%s, samples=%d)",
                pmc_id, gse_id, context, meta.sample_count,
            )

        detail = {
            "acc_id": acc_id,
            "gse_id": gse_id,
            "pmc_id": pmc_id,
            "paper_id": acc["paper_id"],
            "tier2_status": status,
            "context": context if meta else None,
            "sample_count": meta.sample_count if meta else None,
        }
        details.append(detail)
        _append_jsonl({"tier": 2, **detail})

        time.sleep(0.5)  # Rate limit

    summary = {
        "valid": valid,
        "invalid": invalid,
        "own_data_confirmed": own_data_confirmed,
        "total": len(accessions),
    }
    logger.info("Tier 2 summary: %s", summary)
    return summary


# ------------------------------------------------------------------
# Tier 3 — Re-discovery
# ------------------------------------------------------------------

def tier3_rediscovery(
    papers_lost_accessions: list[dict], db: PipelineDB, dry_run: bool = False
) -> dict:
    """Re-scan XML and supplement text for papers that lost all accessions.

    Returns summary dict.
    """
    rediscovered = 0
    new_accessions_total = 0
    details: list[dict] = []

    for paper in papers_lost_accessions:
        paper_id = paper["id"]
        pmc_id = paper["pmc_id"]
        xml_path = paper.get("xml_path")
        supp_text_path = paper.get("supplement_text_path")

        new_hits: list[str] = []

        # Mine from XML
        if xml_path:
            xml_hits = mine_accessions_from_xml(xml_path)
            for hit in xml_hits:
                if hit.accession.startswith("GSE"):
                    new_hits.append(hit.accession)

        # Mine from supplement text
        if supp_text_path:
            supp_hits = mine_accessions_from_supplement(supp_text_path)
            for hit in supp_hits:
                if hit.accession.startswith("GSE"):
                    new_hits.append(hit.accession)

        # Deduplicate
        new_hits = list(dict.fromkeys(new_hits))

        # Check which are already in DB for this paper
        existing = {a["gse_id"] for a in db.get_geo_accessions(paper_id)}
        truly_new = [h for h in new_hits if h not in existing]

        if truly_new:
            rediscovered += 1
            new_accessions_total += len(truly_new)
            logger.info("[%s] Re-discovered %d accession(s): %s",
                        pmc_id, len(truly_new), truly_new)

            if not dry_run:
                for gse_id in truly_new:
                    # Validate via SOFT before storing
                    meta, context = validate_and_fetch_soft(gse_id, paper.get("pmid"))
                    if meta:
                        acc_data = {
                            "gse_id": gse_id,
                            "context": context,
                            "confidence": 0.7,
                            "discovery_strategies": ["tier3_rediscovery"],
                            "soft_fetched": True,
                            "data_type": meta.data_type,
                            "platform": meta.platform,
                            "sample_count": meta.sample_count,
                            "series_title": meta.title,
                            "series_summary": meta.summary[:2000] if meta.summary else None,
                            "linked_pmids": meta.linked_pmids,
                            "submission_date": meta.submission_date,
                        }
                        acc_id = db.store_geo_accession(paper_id, acc_data)

                        # Store samples
                        if meta.samples:
                            for sample in meta.samples:
                                db.store_geo_sample(acc_id, {
                                    "gsm_id": sample.gsm_id,
                                    "sample_title": sample.sample_title,
                                    "source_name": sample.source_name,
                                    "description": sample.description,
                                    "characteristics": sample.characteristics,
                                    "sra_accession": sample.sra_accession,
                                })
                    time.sleep(0.5)
        else:
            logger.debug("[%s] No new accessions found", pmc_id)

        detail = {
            "paper_id": paper_id,
            "pmc_id": pmc_id,
            "new_accessions": truly_new,
        }
        details.append(detail)
        _append_jsonl({"tier": 3, **detail})

    summary = {
        "papers_scanned": len(papers_lost_accessions),
        "papers_rediscovered": rediscovered,
        "new_accessions": new_accessions_total,
    }
    logger.info("Tier 3 summary: %s", summary)
    return summary


# ------------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------------

def run_grounding(
    db: PipelineDB,
    dry_run: bool = False,
    single_pmc: str | None = None,
    max_tier: int = 3,
) -> dict:
    """Run accession grounding pipeline.

    Returns combined summary dict.
    """
    # Get all accessions with paper info
    all_accessions = db.get_all_geo_accessions_with_paper()

    if single_pmc:
        all_accessions = [a for a in all_accessions if a.get("pmc_id") == single_pmc]
        if not all_accessions:
            logger.info("No accessions found for %s", single_pmc)
            return {}

    logger.info("Grounding %d accessions (dry_run=%s, max_tier=%d)",
                len(all_accessions), dry_run, max_tier)

    summary: dict = {}

    # ------ Tier 1 ------
    t1 = tier1_text_verification(all_accessions, dry_run=dry_run)
    summary["tier1"] = t1

    # Partition accessions by Tier 1 result
    # Re-read the JSONL to get per-accession status
    t1_results: dict[int, str] = {}
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("tier") == 1:
                    t1_results[rec["acc_id"]] = rec["tier1_status"]

    hallucinated_ids = [
        a for a in all_accessions
        if t1_results.get(a["id"]) == "not_in_text"
    ]
    verified_ids = [
        a for a in all_accessions
        if t1_results.get(a["id"]) == "verified"
    ]

    # Remove hallucinated accessions
    if hallucinated_ids and not dry_run:
        logger.info("Removing %d hallucinated accessions from DB", len(hallucinated_ids))
        for acc in hallucinated_ids:
            db.mark_accession_grounded(acc["id"], "hallucinated")
            db.remove_geo_accession(acc["id"])
    elif hallucinated_ids:
        logger.info("DRY RUN: would remove %d hallucinated accessions", len(hallucinated_ids))

    if max_tier < 2:
        return summary

    # ------ Tier 2 ------
    if verified_ids:
        t2 = tier2_api_validation(verified_ids, dry_run=dry_run)
        summary["tier2"] = t2

        # Read Tier 2 results
        t2_results: dict[int, str] = {}
        with open(OUTPUT_JSONL) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("tier") == 2:
                    t2_results[rec["acc_id"]] = rec["tier2_status"]

        invalid_ids = [
            a for a in verified_ids
            if t2_results.get(a["id"]) == "invalid_gse"
        ]

        if invalid_ids and not dry_run:
            logger.info("Removing %d invalid accessions from DB", len(invalid_ids))
            for acc in invalid_ids:
                db.mark_accession_grounded(acc["id"], "invalid_gse")
                db.remove_geo_accession(acc["id"])
        elif invalid_ids:
            logger.info("DRY RUN: would remove %d invalid accessions", len(invalid_ids))

        # Mark valid accessions as grounded
        if not dry_run:
            for acc in verified_ids:
                if t2_results.get(acc["id"]) == "valid":
                    db.mark_accession_grounded(acc["id"], "grounded")

    if max_tier < 3:
        return summary

    # ------ Tier 3 ------
    # Find papers that lost all accessions (had some before, now have none)
    removed_paper_ids = set()
    for acc in hallucinated_ids:
        removed_paper_ids.add(acc["paper_id"])
    if not dry_run:
        # Also include papers with invalid accessions
        t2_results_final: dict[int, str] = {}
        if OUTPUT_JSONL.exists():
            with open(OUTPUT_JSONL) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("tier") == 2:
                        t2_results_final[rec["acc_id"]] = rec["tier2_status"]
        for acc in verified_ids:
            if t2_results_final.get(acc["id"]) == "invalid_gse":
                removed_paper_ids.add(acc["paper_id"])

    # Check which papers now have zero accessions
    papers_needing_rediscovery = []
    for paper_id in removed_paper_ids:
        remaining = db.get_geo_accessions(paper_id)
        if not remaining:
            paper = db.get_paper(paper_id=paper_id)
            if paper:
                papers_needing_rediscovery.append(paper)

    if single_pmc:
        papers_needing_rediscovery = [
            p for p in papers_needing_rediscovery if p["pmc_id"] == single_pmc
        ]

    if papers_needing_rediscovery:
        t3 = tier3_rediscovery(papers_needing_rediscovery, db, dry_run=dry_run)
        summary["tier3"] = t3
    else:
        logger.info("Tier 3: no papers need re-discovery")
        summary["tier3"] = {"papers_scanned": 0, "papers_rediscovered": 0, "new_accessions": 0}

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground GEO accessions: verify, validate, re-discover")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only, no DB changes")
    parser.add_argument("--single", type=str, default=None, metavar="PMC_ID",
                        help="Process a single paper by PMC ID")
    parser.add_argument("--tier", type=int, default=3, choices=[1, 2, 3],
                        help="Maximum tier to run (default: 3)")
    args = parser.parse_args()

    db = PipelineDB()

    try:
        summary = run_grounding(
            db,
            dry_run=args.dry_run,
            single_pmc=args.single,
            max_tier=args.tier,
        )
        print("\n=== Accession Grounding Summary ===")
        for tier_name, tier_data in summary.items():
            print(f"\n{tier_name}:")
            for k, v in tier_data.items():
                print(f"  {k}: {v}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
