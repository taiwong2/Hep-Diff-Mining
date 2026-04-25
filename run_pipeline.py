"""End-to-end pipeline orchestrator for hepatocyte differentiation mining.

Steps:
  0.  Re-fetch PMC XMLs (existing fetch_pmc_xmls.py)
  1.  Bootstrap DB from triage_results.jsonl
  2.  Convert PMC XMLs to markdown, store paths in DB
  3.  Fetch supplement files from PMC OA
  4.  Process supplements (PDF/Word→text, Excel/CSV→tables) + disk backfill
  5.  Build reference graph, set processing_priority
  6.  GEO discovery (text mining + elink + SOFT validation)
  6.5 Accession grounding (verify, validate, re-discover GEO accessions)
  7.  Extract protocols (agentic for articles, lighter for reviews)
  8.  Grounding cleanup (validate terms against source text)
  9.  GEO sample-to-stage mapping (with Tier 2 LLM fallback)
  10. Print statistics
  11. RNA-seq metadata extraction (LLM-based)
  12. Repository cross-referencing (ENA/SRA/GEO API) + organism/sample backfill
  13. Expression data retrieval + integration + comparison/day backfill
  14. Expression integration (protocol × gene matrix with cross-study normalization)
  15. Export (final multi-sheet Excel workbook)

Each step is resumable — checks DB state before processing.

Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --from-step 2      # resume from step 2
    python run_pipeline.py --only-step 7      # run only step 7
    python run_pipeline.py --limit 5          # limit extraction to 5 papers
    python run_pipeline.py --skip-fetch       # skip PMC XML re-fetch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
TRIAGE_JSONL = PROJECT_ROOT / "data" / "triage" / "triage_results.jsonl"
PARSED_TEXT_DIR = PROJECT_ROOT / "data" / "db" / "parsed_texts"
SUPPLEMENT_TEXT_DIR = PROJECT_ROOT / "data" / "db" / "supplement_texts"


def step0_fetch_xmls() -> None:
    """Re-fetch PMC XMLs using existing script."""
    print("\n" + "=" * 60)
    print("Step 0: Fetching PMC XMLs")
    print("=" * 60)

    from data_layer.pmc.fetch_pmc_xmls import main as fetch_main
    fetch_main()


def step1_bootstrap_db(db) -> None:
    """Import triage results into database."""
    print("\n" + "=" * 60)
    print("Step 1: Bootstrapping database from triage results")
    print("=" * 60)

    existing = db._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    if existing > 0:
        print(f"  Database already has {existing} papers")
        # Still import any new ones
        imported = db.import_from_triage_jsonl(TRIAGE_JSONL)
        print(f"  Imported {imported} new papers")
    else:
        imported = db.import_from_triage_jsonl(TRIAGE_JSONL)
        print(f"  Imported {imported} papers from triage results")

    stats = db.get_stats()
    print(f"  Total papers: {stats['total_papers']}")
    print(f"  By category: {json.dumps(stats['by_category'], indent=4)}")


def step2_convert_xml_to_text(db) -> None:
    """Convert PMC XMLs to structured markdown."""
    print("\n" + "=" * 60)
    print("Step 2: Converting PMC XMLs to markdown")
    print("=" * 60)

    from data_layer.xml_to_text import parse_pmc_xml_to_text

    papers = db.get_papers_needing_text()
    print(f"  Papers needing text conversion: {len(papers)}")

    PARSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    converted = 0
    failed = 0
    for i, paper in enumerate(papers):
        pmc_id = paper["pmc_id"]
        xml_path = paper.get("xml_path")

        if not xml_path or not Path(xml_path).exists():
            continue

        out_path = PARSED_TEXT_DIR / f"{pmc_id}.md"
        if out_path.exists():
            # Already converted, just update DB
            db.update_paper(paper["id"], parsed_text_path=str(out_path))
            converted += 1
            continue

        parsed = parse_pmc_xml_to_text(xml_path)
        if parsed and parsed.full_text:
            out_path.write_text(parsed.full_text)
            db.update_paper(
                paper["id"],
                parsed_text_path=str(out_path),
                abstract=parsed.abstract if parsed.abstract else paper.get("abstract"),
            )
            converted += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(papers)} ({converted} ok, {failed} failed)")

    print(f"  Converted: {converted}, Failed: {failed}")


def step3_fetch_supplements(db) -> None:
    """Download supplement files from PMC."""
    print("\n" + "=" * 60)
    print("Step 3: Fetching supplement files")
    print("=" * 60)

    from data_layer.fetch_supplements import fetch_all_supplements

    papers = db._conn.execute(
        """SELECT p.* FROM papers p
           WHERE p.triage_category IN ('primary_protocol', 'disease_model', 'methods_tool', 'review')
           AND p.xml_path IS NOT NULL
           AND p.supplement_dir IS NULL"""
    ).fetchall()
    papers = [dict(p) for p in papers]

    print(f"  Papers to check for supplements: {len(papers)}")
    results = fetch_all_supplements(papers)

    fetched = 0
    no_supps = 0
    for r in results:
        paper = db.get_paper(pmc_id=r["pmc_id"])
        if not paper:
            continue
        if r.get("supp_dir"):
            db.update_paper(paper["id"], supplement_dir=r["supp_dir"])
            fetched += 1
        else:
            # Mark as checked so we don't re-process
            db.update_paper(paper["id"], supplement_dir="none")
            no_supps += 1

    print(f"  Papers with supplements: {fetched}")
    print(f"  Papers with no supplements: {no_supps}")


def step4_process_supplements(db, skip_pdf: bool = False) -> None:
    """Convert supplement files to text/tables."""
    print("\n" + "=" * 60)
    print(f"Step 4: Processing supplement files{' (skipping PDFs)' if skip_pdf else ''}")
    print("=" * 60)

    from data_layer.supplement_processor import process_supplements

    DATA_ROOT = PROJECT_ROOT / "data" / "db"

    # ------------------------------------------------------------------
    # Backfill supplement_dir from disk (existing _supp/ directories)
    # ------------------------------------------------------------------
    all_papers = db._conn.execute(
        """SELECT id, pmc_id, supplement_dir FROM papers
           WHERE xml_path IS NOT NULL"""
    ).fetchall()
    backfill_dir = 0
    for p in all_papers:
        pmc_id = p["pmc_id"]
        shard = pmc_id[:5]
        supp_dir = DATA_ROOT / shard / f"{pmc_id}_supp"
        if supp_dir.exists() and p["supplement_dir"] in (None, "none"):
            db.update_paper(p["id"], supplement_dir=str(supp_dir))
            backfill_dir += 1
    if backfill_dir:
        print(f"  Backfilled supplement_dir for {backfill_dir} papers from disk")

    # ------------------------------------------------------------------
    # Backfill supplement_text_path from disk (existing _supp.md files)
    # ------------------------------------------------------------------
    all_papers2 = db._conn.execute(
        """SELECT id, pmc_id, supplement_text_path FROM papers"""
    ).fetchall()
    backfill_text = 0
    for p in all_papers2:
        pmc_id = p["pmc_id"]
        supp_text = SUPPLEMENT_TEXT_DIR / f"{pmc_id}_supp.md"
        if supp_text.exists() and not p["supplement_text_path"]:
            db.update_paper(p["id"], supplement_text_path=str(supp_text))
            backfill_text += 1
    if backfill_text:
        print(f"  Backfilled supplement_text_path for {backfill_text} papers from disk")

    # ------------------------------------------------------------------
    # Process remaining supplements
    # ------------------------------------------------------------------
    papers = db._conn.execute(
        """SELECT p.* FROM papers p
           WHERE p.supplement_dir IS NOT NULL
           AND p.supplement_dir != 'none'
           AND p.supplement_text_path IS NULL"""
    ).fetchall()
    papers = [dict(p) for p in papers]

    print(f"  Papers with unprocessed supplements: {len(papers)}")

    SUPPLEMENT_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for i, paper in enumerate(papers):
        supp_dir = paper.get("supplement_dir")
        if not supp_dir or not Path(supp_dir).exists():
            skipped += 1
            continue

        text = process_supplements(supp_dir, skip_pdf=skip_pdf)
        if text and text.strip():
            pmc_id = paper["pmc_id"]
            out_path = SUPPLEMENT_TEXT_DIR / f"{pmc_id}_supp.md"
            out_path.write_text(text)
            db.update_paper(paper["id"], supplement_text_path=str(out_path))
            processed += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(papers)} ({processed} processed)")

    print(f"  Supplements processed: {processed}")
    print(f"  Skipped (no dir): {skipped}")


def step5_reference_graph(db) -> None:
    """Build citation DAG and set processing order."""
    print("\n" + "=" * 60)
    print("Step 5: Building reference graph")
    print("=" * 60)

    from data_layer.reference_graph import build_reference_graph

    order = build_reference_graph(db)
    print(f"  Processing order set for {len(order)} papers")

    # Show first 10 in order
    if order:
        print("  First 10 papers to process:")
        for pid in order[:10]:
            paper = db.get_paper(paper_id=pid)
            if paper:
                print(f"    {paper['pmc_id']}: {(paper.get('title') or '')[:60]}")


def step6_geo_discovery(db) -> None:
    """Run GEO accession discovery on all eligible papers."""
    print("\n" + "=" * 60)
    print("Step 6: GEO Discovery")
    print("=" * 60)

    from data_layer.geo_linker import discover_geo_all

    papers = db.get_papers_needing_geo()
    print(f"  Papers eligible for GEO discovery: {len(papers)}")

    if not papers:
        return

    # Build PMC client for elink strategy
    pmc_client = None
    try:
        from data_layer.pmc.pmc_client import PMCClient
        pmc_client = PMCClient()
    except Exception as e:
        logger.warning("PMCClient unavailable, elink strategy skipped: %s", e)

    found = discover_geo_all(db, pmc_client=pmc_client)
    print(f"  Papers linked to GEO: {found}")


def step6b_accession_grounding(db) -> None:
    """Verify, validate, and re-discover GEO accessions."""
    print("\n" + "=" * 60)
    print("Step 6.5: Accession Grounding")
    print("=" * 60)

    from steps.ground_accessions import run_grounding

    summary = run_grounding(db)
    for tier_name, tier_data in summary.items():
        print(f"  {tier_name}: {tier_data}")


def step7_extract_protocols(db, limit: int | None = None) -> None:
    """Run agentic extraction on articles and lighter extraction on reviews."""
    print("\n" + "=" * 60)
    print("Step 7: Extracting protocols")
    print("=" * 60)

    from llm.agents.agentic_extractor import run_extraction
    from llm.agents.review_extractor import run_review_extraction

    # Article extraction
    article_count = db._conn.execute(
        """SELECT COUNT(*) FROM papers
           WHERE triage_category IN ('primary_protocol', 'disease_model', 'methods_tool')
           AND extraction_status = 'pending'
           AND parsed_text_path IS NOT NULL"""
    ).fetchone()[0]

    print(f"\n  Articles ready for extraction: {article_count}")
    if article_count > 0:
        asyncio.run(run_extraction(db, limit=limit))

    # Review extraction
    review_count = db._conn.execute(
        """SELECT COUNT(*) FROM papers
           WHERE triage_category = 'review'
           AND extraction_status = 'pending'
           AND parsed_text_path IS NOT NULL"""
    ).fetchone()[0]

    print(f"\n  Reviews ready for extraction: {review_count}")
    if review_count > 0:
        asyncio.run(run_review_extraction(db, limit=limit))


def step8_grounding(db) -> None:
    """Validate extracted terms against source text, remove hallucinations."""
    print("\n" + "=" * 60)
    print("Step 8: Grounding Cleanup")
    print("=" * 60)

    from steps.grounding_cleanup import run_cleanup

    run_cleanup(db)


def step9_geo_sample_mapping(db) -> None:
    """Run GEO sample-to-stage mapping on papers with GEO data."""
    print("\n" + "=" * 60)
    print("Step 9: GEO Sample-to-Stage Mapping")
    print("=" * 60)

    from data_layer.geo_sample_mapper import map_all_papers
    from llm.openrouter.client import OpenRouterClient

    papers = db.get_papers_needing_geo_mapping()
    print(f"  Papers eligible for sample mapping: {len(papers)}")

    if not papers:
        return

    client = OpenRouterClient()
    total = map_all_papers(db, client=client)
    print(f"  Sample-to-stage mappings created: {total}")


def step10_statistics(db) -> None:
    """Print pipeline statistics."""
    print("\n" + "=" * 60)
    print("Step 10: Pipeline Statistics")
    print("=" * 60)

    stats = db.get_stats()
    print(f"\n  Total papers in DB: {stats['total_papers']}")
    print(f"\n  By triage category:")
    for cat, count in sorted(stats["by_category"].items(),
                              key=lambda x: -x[1]):
        print(f"    {cat:25s}: {count:4d}")

    print(f"\n  Extraction status:")
    for status, count in sorted(stats["extraction_status"].items(),
                                 key=lambda x: -x[1]):
        print(f"    {status:15s}: {count:4d}")

    print(f"\n  Papers with parsed text: {stats['papers_with_text']}")
    print(f"  Protocols extracted: {stats['protocols_extracted']}")
    print(f"  Corpus cache entries: {stats['corpus_cache_entries']}")

    print(f"\n  GEO enrichment:")
    print(f"    Papers linked to GEO: {stats['geo_papers_linked']}")
    print(f"    GEO series (own data): {stats['geo_series_own_data']}")
    print(f"    GEO samples total: {stats['geo_samples_total']}")
    print(f"    Samples mapped to stages: {stats['geo_mapped_samples']}")

    print(f"\n  RNA-seq / Expression:")
    print(f"    Papers checked for RNA-seq: {stats['rnaseq_checked']}")
    print(f"    Papers with RNA-seq: {stats['rnaseq_papers']}")
    print(f"    Expression values: {stats['expression_values_count']}")
    print(f"    Distinct genes: {stats['expression_genes']}")
    print(f"    Papers with expression data: {stats['expression_papers']}")


def step11_rnaseq_extraction(db, limit: int | None = None) -> None:
    """Extract RNA-seq metadata from papers via LLM."""
    print("\n" + "=" * 60)
    print("Step 11: RNA-seq Metadata Extraction")
    print("=" * 60)

    from steps.rnaseq_extract import run as rnaseq_run

    asyncio.run(rnaseq_run(db, limit=limit))


def step12_rnaseq_crossref(db, limit: int | None = None) -> None:
    """Cross-reference RNA-seq accessions with external repositories."""
    print("\n" + "=" * 60)
    print("Step 12: Repository Cross-Referencing")
    print("=" * 60)

    from steps.rnaseq_crossref import run as crossref_run

    crossref_run(db, limit=limit)


def step13_expression_retrieval(db, limit: int | None = None) -> None:
    """Retrieve and integrate expression data."""
    print("\n" + "=" * 60)
    print("Step 13: Expression Data Retrieval & Integration")
    print("=" * 60)

    from steps.rnaseq_retrieve import run as retrieve_run

    retrieve_run(db, limit=limit)


def step14_expression_integration(db) -> None:
    """Build protocol × gene expression matrix with cross-study normalization."""
    print("\n" + "=" * 60)
    print("Step 14: Expression Integration")
    print("=" * 60)

    from steps.rnaseq_integrate import run as integrate_run

    integrate_run(db)


def step15_export(db) -> None:
    """Export final multi-sheet Excel workbook."""
    print("\n" + "=" * 60)
    print("Step 15: Export")
    print("=" * 60)

    from steps.export_results import export, DEFAULT_OUTPUT

    export(DEFAULT_OUTPUT, db)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the hepatocyte differentiation mining pipeline",
    )
    parser.add_argument("--from-step", type=float, default=0,
                        help="Start from this step number (0-15, use 6.5 for accession grounding)")
    parser.add_argument("--only-step", type=float, default=None,
                        help="Run only this step")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit extraction to N papers")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip step 0 (PMC XML re-fetch)")
    parser.add_argument("--skip-pdf", action="store_true",
                        help="Skip PDF OCR in step 4 (process only docx/xlsx/csv)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from data_layer.database import PipelineDB
    db = PipelineDB()

    steps = {
        0:   ("Fetch PMC XMLs", lambda: step0_fetch_xmls()),
        1:   ("Bootstrap DB", lambda: step1_bootstrap_db(db)),
        2:   ("XML → Markdown", lambda: step2_convert_xml_to_text(db)),
        3:   ("Fetch Supplements", lambda: step3_fetch_supplements(db)),
        4:   ("Process Supplements", lambda: step4_process_supplements(db, skip_pdf=args.skip_pdf)),
        5:   ("Reference Graph", lambda: step5_reference_graph(db)),
        6:   ("GEO Discovery", lambda: step6_geo_discovery(db)),
        6.5: ("Accession Grounding", lambda: step6b_accession_grounding(db)),
        7:   ("Extract Protocols", lambda: step7_extract_protocols(db, args.limit)),
        8:   ("Grounding Cleanup", lambda: step8_grounding(db)),
        9:   ("GEO Sample Mapping", lambda: step9_geo_sample_mapping(db)),
        10:  ("Statistics", lambda: step10_statistics(db)),
        11:  ("RNA-seq Extraction", lambda: step11_rnaseq_extraction(db, args.limit)),
        12:  ("Repository Cross-Ref", lambda: step12_rnaseq_crossref(db, args.limit)),
        13:  ("Expression Retrieval", lambda: step13_expression_retrieval(db, args.limit)),
        14:  ("Expression Integration", lambda: step14_expression_integration(db)),
        15:  ("Export", lambda: step15_export(db)),
    }

    step_order = sorted(steps.keys())

    if args.only_step is not None:
        if args.only_step in steps:
            name, func = steps[args.only_step]
            print(f"\nRunning only step {args.only_step}: {name}")
            func()
        else:
            valid = ", ".join(str(s) for s in step_order)
            print(f"Invalid step: {args.only_step}. Valid: {valid}")
    else:
        start = args.from_step
        if args.skip_fetch and start == 0:
            start = 1

        for step_num in step_order:
            if step_num < start:
                continue
            if step_num == 0 and args.skip_fetch:
                continue
            name, func = steps[step_num]
            t0 = time.time()
            try:
                func()
            except Exception as e:
                logger.exception("Step %g (%s) failed", step_num, name)
                print(f"\n  ERROR in step {step_num}: {e}")
                print("  Use --from-step to resume from this step")
                break
            elapsed = time.time() - t0
            print(f"  Step {step_num} completed in {elapsed:.1f}s")

    db.close()
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
