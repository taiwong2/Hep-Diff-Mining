"""Phase 1: RNA-seq metadata extraction via LLM.

For every paper with completed extraction, uses an LLM to identify RNA-seq
data and extract structured metadata (technology, accessions, DEG stats,
pipeline info) from the paper text and supplement text.

Usage:
    python3 rnaseq_extract.py                     # extract all eligible
    python3 rnaseq_extract.py --limit 10           # test on 10 papers
    python3 rnaseq_extract.py --dry-run            # show eligible without running
    python3 rnaseq_extract.py --single PMC7870774  # test on a single paper
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from data_layer.database import PipelineDB
from llm.openrouter.client import OpenRouterClient

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "llm" / "agents" / "prompts" / "rnaseq_extraction.txt"
OUTPUT_JSONL = Path("data/logs/rnaseq_extraction_results.jsonl")

# Text budget
MAIN_TEXT_BUDGET = 120_000
SUPP_TEXT_BUDGET = 40_000


def _load_prompt() -> str:
    return PROMPT_PATH.read_text()


async def extract_rnaseq_for_paper(
    client: OpenRouterClient,
    db: PipelineDB,
    paper: dict,
    system_prompt: str,
) -> dict | None:
    """LLM extraction of RNA-seq metadata from paper text."""
    paper_id = paper["id"]
    pmc_id = paper["pmc_id"]

    # Load paper text (fall back to standard path if parsed_text_path not set)
    text_path = paper.get("parsed_text_path")
    if not text_path or not Path(text_path).exists():
        fallback = Path("data/db/parsed_texts") / f"{pmc_id}.md"
        if fallback.exists():
            text_path = str(fallback)
        else:
            return None

    paper_text = Path(text_path).read_text()
    if not paper_text.strip():
        return None

    # Truncate to budget
    if len(paper_text) > MAIN_TEXT_BUDGET:
        paper_text = paper_text[:MAIN_TEXT_BUDGET]

    # Load supplement text if available (fall back to standard path)
    supp_text = ""
    supp_path = paper.get("supplement_text_path")
    if not supp_path or not Path(supp_path).exists():
        supp_fallback = Path("data/db/supplement_texts") / f"{pmc_id}_supp.md"
        if supp_fallback.exists():
            supp_path = str(supp_fallback)
    if supp_path and Path(supp_path).exists():
        supp_text = Path(supp_path).read_text()
        if len(supp_text) > SUPP_TEXT_BUDGET:
            supp_text = supp_text[:SUPP_TEXT_BUDGET]

    # Check for known GEO accessions as hints
    geo_accs = db.get_geo_accessions(paper_id)
    geo_hint = ""
    if geo_accs:
        gse_ids = [a["gse_id"] for a in geo_accs]
        geo_hint = f"\n\nKnown GEO accessions for this paper: {', '.join(gse_ids)}"

    # Build user message
    user_content = f"Paper text:\n{paper_text}"
    if geo_hint:
        user_content += geo_hint
    if supp_text:
        user_content += f"\n\nSupplementary text:\n{supp_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    resp = await client.complete(
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=4096,
    )

    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("[%s] Failed to parse LLM response as JSON", pmc_id)
        return None


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


async def run(
    db: PipelineDB,
    limit: int | None = None,
    dry_run: bool = False,
    single: str | None = None,
) -> None:
    if single:
        paper = db.get_paper(pmc_id=single)
        if not paper:
            print(f"Paper {single} not found in database.")
            return
        papers = [paper]
    else:
        papers = db.get_papers_needing_rnaseq()
        if limit:
            papers = papers[:limit]

    if not papers:
        print("No papers eligible for RNA-seq extraction.")
        return

    print(f"Found {len(papers)} paper(s) eligible for RNA-seq extraction.")

    if dry_run:
        for p in papers:
            has_supp = "yes" if p.get("supplement_text_path") else "no"
            geo_accs = db.get_geo_accessions(p["id"])
            geo_str = f"{len(geo_accs)} GEO" if geo_accs else "no GEO"
            print(f"  {p['pmc_id']}: supp={has_supp}, {geo_str} — {(p.get('title') or '')[:60]}")
        return

    system_prompt = _load_prompt()
    total_rnaseq = 0
    total_none = 0

    async with OpenRouterClient(max_concurrent=5) as client:
        for i, paper in enumerate(papers):
            pmc_id = paper["pmc_id"]
            paper_id = paper["id"]
            print(
                f"[{i+1}/{len(papers)}] {pmc_id}: {(paper.get('title') or '')[:55]}...",
                end=" ",
                flush=True,
            )

            try:
                result = await extract_rnaseq_for_paper(client, db, paper, system_prompt)
            except Exception as e:
                logger.exception("Error extracting RNA-seq for %s", pmc_id)
                print(f"error: {e}")
                _append_jsonl(OUTPUT_JSONL, {
                    "pmc_id": pmc_id, "error": str(e),
                })
                continue

            if result is None:
                db.update_paper(paper_id, rnaseq_status="checked_none")
                total_none += 1
                print("no text / parse error")
                continue

            has_rnaseq = result.get("has_rnaseq", False)

            if has_rnaseq:
                db.store_rnaseq_metadata(paper_id, result)
                db.update_paper(paper_id, rnaseq_status="has_rnaseq")
                total_rnaseq += 1

                n_acc = len(result.get("accessions") or [])
                rtype = result.get("rnaseq_type", "?")
                print(f"RNA-seq ({rtype}), {n_acc} accession(s)")
            else:
                db.store_rnaseq_metadata(paper_id, result)
                db.update_paper(paper_id, rnaseq_status="checked_none")
                total_none += 1
                print("no RNA-seq")

            _append_jsonl(OUTPUT_JSONL, {
                "pmc_id": pmc_id,
                "has_rnaseq": has_rnaseq,
                "rnaseq_type": result.get("rnaseq_type"),
                "n_accessions": len(result.get("accessions") or []),
            })

    print(f"\nDone. {total_rnaseq} paper(s) with RNA-seq, {total_none} without.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Extract RNA-seq metadata via LLM",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show eligible papers without running extraction")
    parser.add_argument("--single", type=str, default=None,
                        help="Process a single paper by PMC ID")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db = PipelineDB()
    try:
        asyncio.run(run(db, limit=args.limit, dry_run=args.dry_run, single=args.single))
    finally:
        db.close()


if __name__ == "__main__":
    main()
