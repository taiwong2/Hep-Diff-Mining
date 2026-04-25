"""Phase 2: Repository cross-referencing.

For every accession discovered in Phase 1 (LLM extraction) and existing GEO
discovery, pulls metadata from external APIs (GEO, ENA, SRA, ArrayExpress)
and classifies data availability.

Usage:
    python3 rnaseq_crossref.py                # cross-ref all eligible
    python3 rnaseq_crossref.py --limit 5      # test on 5 papers
    python3 rnaseq_crossref.py --dry-run      # show eligible without running
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from data_layer.database import PipelineDB
from data_layer.ena_client import (
    fetch_ena_metadata,
    fetch_sra_metadata,
    fetch_arrayexpress_metadata,
)
from data_layer.geo_linker import (
    check_geo_supplementary_files,
    validate_and_fetch_soft,
)

logger = logging.getLogger(__name__)

OUTPUT_JSONL = Path("data/logs/rnaseq_crossref_results.jsonl")


def _collect_accessions(db: PipelineDB, paper_id: int) -> list[dict]:
    """Collect all known accessions for a paper from multiple sources."""
    accessions: dict[str, dict] = {}

    # From LLM extraction (rnaseq_metadata)
    rm = db.get_rnaseq_metadata(paper_id)
    if rm and rm.get("accessions"):
        for acc in rm["accessions"]:
            key = acc.get("accession", "")
            if key:
                accessions[key] = {
                    "accession": key,
                    "repository": acc.get("repository", _guess_repository(key)),
                    "context": acc.get("context", "ambiguous"),
                    "source": "llm_extraction",
                }

    # From GEO discovery (geo_accessions table)
    geo_accs = db.get_geo_accessions(paper_id)
    for ga in geo_accs:
        gse_id = ga["gse_id"]
        if gse_id not in accessions:
            accessions[gse_id] = {
                "accession": gse_id,
                "repository": "GEO",
                "context": ga.get("context", "ambiguous"),
                "source": "geo_discovery",
            }

    return list(accessions.values())


def _guess_repository(accession: str) -> str:
    """Guess repository from accession pattern."""
    acc = accession.upper()
    if acc.startswith("GSE") or acc.startswith("GSM"):
        return "GEO"
    if acc.startswith("PRJNA") or acc.startswith("SRP"):
        return "SRA"
    if acc.startswith("PRJEB") or acc.startswith("ERP"):
        return "ENA"
    if acc.startswith("E-MTAB") or acc.startswith("E-"):
        return "ArrayExpress"
    if acc.startswith("DRA"):
        return "DDBJ"
    return "unknown"


def classify_availability(accessions: list[dict], geo_supp_files: dict) -> str:
    """Classify data availability based on cross-referencing results."""
    has_geo = any(a["repository"] == "GEO" for a in accessions)
    has_matrix = any(
        any(f.get("has_count_matrix") for f in geo_supp_files.get(a["accession"], []))
        for a in accessions
    )
    if has_geo and has_matrix:
        return "geo_with_matrix"
    if has_geo:
        return "geo_raw_only"
    has_ena = any(a["repository"] in ("ENA", "SRA") for a in accessions)
    if has_ena:
        return "ena_only"
    has_ae = any(a["repository"] == "ArrayExpress" for a in accessions)
    if has_ae:
        return "arrayexpress_only"
    return "no_deposit"


def crossref_paper(db: PipelineDB, paper: dict) -> dict:
    """Cross-reference all accessions for a single paper."""
    paper_id = paper["id"]
    pmc_id = paper["pmc_id"]
    pmid = paper.get("pmid")

    accessions = _collect_accessions(db, paper_id)
    if not accessions:
        return {"pmc_id": pmc_id, "n_accessions": 0, "classification": "no_deposit"}

    geo_supp_files: dict[str, list[dict]] = {}
    results = []

    for acc_info in accessions:
        accession = acc_info["accession"]
        repo = acc_info["repository"]

        logger.info("[%s] Cross-referencing %s (%s)...", pmc_id, accession, repo)

        meta = None
        if repo == "GEO":
            # Check supplementary files
            supp_files = check_geo_supplementary_files(accession)
            geo_supp_files[accession] = supp_files

            # Get series metadata if not already fetched via geo_linker
            existing_geo = db.get_geo_accessions(paper_id)
            already_fetched = any(
                g["gse_id"] == accession and g.get("soft_fetched")
                for g in existing_geo
            )

            if not already_fetched:
                series_meta, context = validate_and_fetch_soft(accession, pmid)
                if series_meta:
                    meta = {
                        "accession": accession,
                        "repository": "GEO",
                        "project_title": series_meta.title,
                        "organism": series_meta.organism or None,
                        "data_type": series_meta.data_type,
                        "platform": series_meta.platform,
                        "sample_count": series_meta.sample_count,
                        "has_processed_matrix": any(
                            f.get("has_count_matrix") for f in supp_files
                        ),
                        "supplementary_files": supp_files,
                        "fetch_status": "fetched",
                    }
                time.sleep(0.5)
            else:
                # Use existing data
                for g in existing_geo:
                    if g["gse_id"] == accession:
                        meta = {
                            "accession": accession,
                            "repository": "GEO",
                            "project_title": g.get("series_title"),
                            "organism": g.get("organism"),
                            "data_type": g.get("data_type"),
                            "platform": g.get("platform"),
                            "sample_count": g.get("sample_count"),
                            "has_processed_matrix": any(
                                f.get("has_count_matrix") for f in supp_files
                            ),
                            "supplementary_files": supp_files,
                            "fetch_status": "fetched",
                        }
                        break

        elif repo in ("ENA", "SRA"):
            meta = fetch_ena_metadata(accession)
            if not meta:
                meta = fetch_sra_metadata(accession)
            if meta:
                meta["fetch_status"] = "fetched"
            else:
                meta = {
                    "accession": accession,
                    "repository": repo,
                    "fetch_status": "not_found",
                }
            time.sleep(0.3)

        elif repo == "ArrayExpress":
            meta = fetch_arrayexpress_metadata(accession)
            if meta:
                meta["fetch_status"] = "fetched"
            else:
                meta = {
                    "accession": accession,
                    "repository": "ArrayExpress",
                    "fetch_status": "not_found",
                }
            time.sleep(0.3)

        else:
            meta = {
                "accession": accession,
                "repository": repo,
                "fetch_status": "unsupported_repository",
            }

        if meta:
            # Populate sample_metadata from geo_samples when available
            if repo == "GEO" and not meta.get("sample_metadata"):
                for g in existing_geo:
                    if g["gse_id"] == accession:
                        samples = db.get_geo_samples(g["id"])
                        if samples:
                            meta["sample_metadata"] = [
                                {"gsm_id": s["gsm_id"],
                                 "title": s.get("sample_title"),
                                 "source": s.get("source_name")}
                                for s in samples
                            ]
                        break

            db.store_repository_metadata(paper_id, meta)
            results.append(meta)

    # Classify availability
    classification = classify_availability(accessions, geo_supp_files)

    # Update rnaseq_metadata.data_availability
    rm = db.get_rnaseq_metadata(paper_id)
    if rm:
        da = rm.get("data_availability") or {}
        if isinstance(da, str):
            try:
                da = json.loads(da)
            except json.JSONDecodeError:
                da = {}
        da["classification"] = classification
        da["n_accessions"] = len(accessions)
        db.store_rnaseq_metadata(paper_id, {
            **rm,
            "data_availability": da,
        })

    return {
        "pmc_id": pmc_id,
        "n_accessions": len(accessions),
        "classification": classification,
        "repositories": [r.get("repository") for r in results],
    }


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run(
    db: PipelineDB,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    papers = db.get_papers_needing_crossref()
    if limit:
        papers = papers[:limit]

    if not papers:
        print("No papers eligible for repository cross-referencing.")
        return

    print(f"Found {len(papers)} paper(s) eligible for cross-referencing.")

    if dry_run:
        for p in papers:
            rm = db.get_rnaseq_metadata(p["id"])
            n_acc = len(rm.get("accessions") or []) if rm else 0
            print(f"  {p['pmc_id']}: {n_acc} accession(s) — {(p.get('title') or '')[:60]}")
        return

    classifications: dict[str, int] = {}

    for i, paper in enumerate(papers):
        pmc_id = paper["pmc_id"]
        print(
            f"[{i+1}/{len(papers)}] {pmc_id}: {(paper.get('title') or '')[:55]}...",
            end=" ",
            flush=True,
        )

        try:
            result = crossref_paper(db, paper)
            cls = result["classification"]
            classifications[cls] = classifications.get(cls, 0) + 1
            print(f"{result['n_accessions']} accession(s), {cls}")
            _append_jsonl(OUTPUT_JSONL, result)
        except Exception as e:
            logger.exception("Error cross-referencing %s", pmc_id)
            print(f"error: {e}")
            _append_jsonl(OUTPUT_JSONL, {"pmc_id": pmc_id, "error": str(e)})

    print(f"\nDone. Classification summary:")
    for cls, count in sorted(classifications.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    # Backfill organism and sample_metadata on existing repository_metadata rows
    _backfill_repository_metadata(db)


def _backfill_repository_metadata(db: PipelineDB) -> None:
    """Backfill organism and sample_metadata on repository_metadata from geo_accessions/geo_samples."""
    rows = db._conn.execute(
        """SELECT rm.id, rm.paper_id, rm.accession, rm.organism, rm.sample_metadata
           FROM repository_metadata rm
           WHERE rm.repository = 'GEO'"""
    ).fetchall()

    updated = 0
    for row in rows:
        rm_id = row["id"]
        paper_id = row["paper_id"]
        accession = row["accession"]
        needs_organism = not row["organism"]
        needs_samples = not row["sample_metadata"]

        if not needs_organism and not needs_samples:
            continue

        geo_accs = db.get_geo_accessions(paper_id)
        for g in geo_accs:
            if g["gse_id"] != accession:
                continue

            updates = {}
            if needs_organism and g.get("organism"):
                updates["organism"] = g["organism"]

            if needs_samples:
                samples = db.get_geo_samples(g["id"])
                if samples:
                    sample_meta = json.dumps([
                        {"gsm_id": s["gsm_id"],
                         "title": s.get("sample_title"),
                         "source": s.get("source_name")}
                        for s in samples
                    ])
                    updates["sample_metadata"] = sample_meta

            if updates:
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                vals = list(updates.values()) + [rm_id]
                db._conn.execute(
                    f"UPDATE repository_metadata SET {set_clause} WHERE id = ?", vals
                )
                updated += 1
            break

    if updated:
        db._conn.commit()
        print(f"  Backfilled organism/sample_metadata for {updated} repository_metadata rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Cross-reference RNA-seq accessions with repositories",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show eligible papers without running")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db = PipelineDB()
    try:
        run(db, limit=args.limit, dry_run=args.dry_run)
    finally:
        db.close()


if __name__ == "__main__":
    main()
