"""Lighter extraction for review papers.

Single-pass, no tools. Extracts:
- Protocol references mentioned (with DOIs/PMIDs)
- Comparison tables (if the review compares protocols)
- Reference list of papers describing differentiation protocols

Usage:
    python -m llm.agents.review_extractor
    python -m llm.agents.review_extractor --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from llm.openrouter.client import OpenRouterClient, APIError
from data_layer.database import PipelineDB

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "openai/gpt-4o-mini"
PROMPT_PATH = Path(__file__).parent / "prompts" / "review_extraction.txt"
DEFAULT_OUTPUT = Path("data/results/review_extraction_results.jsonl")


def _load_prompt() -> str:
    return PROMPT_PATH.read_text().strip()


async def extract_review(
    client: OpenRouterClient,
    paper_text: str,
    title: str,
) -> dict | None:
    """Single-pass review extraction."""
    system_prompt = _load_prompt()
    user_content = f"Paper title: {title}\n\n---\n\n{paper_text}"

    try:
        resp = await client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model=EXTRACTION_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp["choices"][0]["message"]["content"]
        return json.loads(content)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Review extraction parse error for '%s': %s", title[:50], e)
        return None
    except APIError as e:
        logger.warning("Review extraction API error: %s", e)
        return None


async def run_review_extraction(
    db: PipelineDB,
    output_file: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
) -> None:
    """Run review extraction on all eligible review papers."""
    papers = db.get_review_papers_for_extraction()
    if limit:
        papers = papers[:limit]

    if not papers:
        print("No review papers ready for extraction.")
        return

    print(f"Extracting from {len(papers)} review papers...")
    system_prompt = _load_prompt()

    async with OpenRouterClient(max_concurrent=5) as client:
        for i, paper in enumerate(papers):
            pmc_id = paper["pmc_id"]
            paper_id = paper["id"]
            title = paper.get("title", "")

            text_path = paper.get("parsed_text_path")
            if not text_path or not Path(text_path).exists():
                logger.info("Skipping %s: no parsed text", pmc_id)
                continue

            paper_text = Path(text_path).read_text()
            print(f"[{i+1}/{len(papers)}] {pmc_id}: {title[:60]}...", end=" ", flush=True)

            db.set_extraction_status(paper_id, "in_progress")
            start = time.time()

            result = await extract_review(client, paper_text, title)
            duration = time.time() - start

            if result:
                # Store as a protocol record with review-specific fields
                protocol_record = {
                    "protocol_arm": "review",
                    "is_optimized": False,
                    "cell_source": None,
                    "culture_system": None,
                    "stages": [],
                    "endpoint_assessment": None,
                    "modifications": None,
                    "step_sources": None,
                    "base_protocol_doi": None,
                    "extraction_confidence": result.get("extraction_confidence", 0),
                    "extraction_notes": json.dumps({
                        "protocol_references": result.get("protocol_references", []),
                        "protocol_comparisons": result.get("protocol_comparisons", []),
                        "methodological_insights": result.get("methodological_insights", []),
                        "differentiation_papers": result.get("differentiation_papers_in_references", []),
                    }),
                    "incomplete_flags": [],
                    "pass_number": 1,
                }
                db.store_protocol(paper_id, protocol_record)
                db.set_extraction_status(paper_id, "completed")
                db.log_processing(paper_id, "review_extraction", "completed",
                                  duration_secs=duration)

                # Write to JSONL
                record = {
                    "pmc_id": pmc_id,
                    "doi": paper.get("doi"),
                    "title": title,
                    "protocol_refs_found": len(result.get("protocol_references", [])),
                    "comparisons_found": len(result.get("protocol_comparisons", [])),
                    "diff_papers_in_refs": len(result.get("differentiation_papers_in_references", [])),
                    "confidence": result.get("extraction_confidence"),
                }
                _append_jsonl(output_file, record)
                print(f"done ({record['protocol_refs_found']} refs, "
                      f"{record['comparisons_found']} comparisons)")
            else:
                db.set_extraction_status(paper_id, "failed")
                db.log_processing(paper_id, "review_extraction", "failed",
                                  duration_secs=duration)
                print("failed")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract protocol references from review papers",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db = PipelineDB()
    asyncio.run(run_review_extraction(db, output_file=args.output, limit=args.limit))
    db.close()


if __name__ == "__main__":
    main()
