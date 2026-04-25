"""Pass 3-only supplement enrichment for already-extracted protocols.

Papers that already have extraction_status='completed' but now have
supplement text available (after running pipeline steps 3-4) get their
protocols enriched with quantitative data from supplements.

Usage:
    python3 supplement_enrich.py                  # enrich all eligible
    python3 supplement_enrich.py --limit 5        # test on 5 papers
    python3 supplement_enrich.py --dry-run        # show eligible without running
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_layer.database import PipelineDB
from data_layer.grounding import ground_protocol
from llm.agents.agentic_extractor import run_pass3, merge_pass3
from llm.openrouter.client import OpenRouterClient

logger = logging.getLogger(__name__)

OUTPUT_JSONL = Path("data/logs/supplement_enrich_results.jsonl")


def get_eligible_papers(db: PipelineDB) -> list[dict]:
    """Papers with supplement text that already completed extraction."""
    rows = db._conn.execute(
        """SELECT p.* FROM papers p
           WHERE p.supplement_text_path IS NOT NULL
           AND p.extraction_status = 'completed'
           AND p.triage_category IN ('primary_protocol', 'disease_model', 'methods_tool')
           AND p.id NOT IN (
               SELECT DISTINCT paper_id FROM protocols WHERE pass_number >= 3
           )
           ORDER BY p.id"""
    ).fetchall()
    return [dict(r) for r in rows]


async def enrich_paper(
    client: OpenRouterClient,
    db: PipelineDB,
    paper: dict,
) -> int:
    """Run Pass 3 on all protocols for a paper. Returns count of enriched protocols."""
    paper_id = paper["id"]
    pmc_id = paper["pmc_id"]
    title = paper.get("title", "")

    supp_path = paper.get("supplement_text_path")
    if not supp_path or not Path(supp_path).exists():
        return 0

    supplement_text = Path(supp_path).read_text()
    if not supplement_text.strip():
        return 0

    # Load paper text for grounding
    text_path = paper.get("parsed_text_path")
    paper_text = Path(text_path).read_text() if text_path and Path(text_path).exists() else ""

    protocols = db.get_protocols_for_paper(paper_id)
    if not protocols:
        return 0

    enriched = 0
    for proto in protocols:
        proto_id = proto["id"]
        logger.info("[%s] Pass 3 enrichment for protocol %d (%s)...",
                    pmc_id, proto_id, proto.get("protocol_arm", "?"))

        pass3_result = await run_pass3(client, supplement_text, proto, title)
        if not pass3_result or pass3_result.get("no_additional_data"):
            logger.info("[%s] Protocol %d: no additional data from supplements",
                        pmc_id, proto_id)
            continue

        merged = merge_pass3(dict(proto), pass3_result)

        # Ground merged protocol against paper + supplement text
        merged, removals = ground_protocol(merged, paper_text, supplement_text)
        if removals:
            removed_terms = [r["term"] for r in removals]
            logger.info("[%s] Protocol %d: grounding removed %d item(s): %s",
                        pmc_id, proto_id, len(removals), ", ".join(removed_terms))
            notes = merged.get("extraction_notes", "") or ""
            note = f"Grounding removed: {', '.join(removed_terms)}"
            merged["extraction_notes"] = f"{notes} | {note}".strip(" | ")

        db.update_protocol(proto_id, {
            "stages": merged.get("stages", []),
            "endpoint_assessment": merged.get("endpoint_assessment"),
            "extraction_notes": merged.get("extraction_notes"),
            "extraction_confidence": merged.get("extraction_confidence",
                                                 proto.get("extraction_confidence")),
            "pass_number": 3,
        })
        enriched += 1

        _append_jsonl(OUTPUT_JSONL, {
            "pmc_id": pmc_id,
            "protocol_id": proto_id,
            "protocol_arm": proto.get("protocol_arm"),
            "enriched": True,
            "pass3_notes": pass3_result.get("extraction_notes", ""),
        })

    return enriched


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


async def run(db: PipelineDB, limit: int | None = None, dry_run: bool = False) -> None:
    papers = get_eligible_papers(db)
    if limit:
        papers = papers[:limit]

    if not papers:
        print("No papers eligible for supplement enrichment.")
        return

    print(f"Found {len(papers)} paper(s) eligible for supplement enrichment.")

    if dry_run:
        for p in papers:
            n_protos = len(db.get_protocols_for_paper(p["id"]))
            print(f"  {p['pmc_id']}: {n_protos} protocol(s) — {(p.get('title') or '')[:60]}")
        return

    total_enriched = 0
    async with OpenRouterClient(max_concurrent=3) as client:
        for i, paper in enumerate(papers):
            pmc_id = paper["pmc_id"]
            print(f"[{i+1}/{len(papers)}] {pmc_id}: {(paper.get('title') or '')[:60]}...",
                  end=" ", flush=True)
            try:
                n = await enrich_paper(client, db, paper)
                total_enriched += n
                print(f"enriched {n} protocol(s)")
            except Exception as e:
                logger.exception("Error enriching %s", pmc_id)
                print(f"error: {e}")

    print(f"\nDone. Enriched {total_enriched} protocol(s) across {len(papers)} paper(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pass 3-only supplement enrichment for completed extractions",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show eligible papers without running extraction")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db = PipelineDB()
    try:
        asyncio.run(run(db, limit=args.limit, dry_run=args.dry_run))
    finally:
        db.close()


if __name__ == "__main__":
    main()
