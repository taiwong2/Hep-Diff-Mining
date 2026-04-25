"""Triage classifier: parse PMC XML abstracts and classify papers using DeepSeek v3.

Usage:
    python -m llm.agents.triage_classifier                     # full run
    python -m llm.agents.triage_classifier --limit 10          # test with N papers
    python -m llm.agents.triage_classifier --single FILE.xml   # debug single file
    python -m llm.agents.triage_classifier --summary-only      # print summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from llm.openrouter.client import OpenRouterClient, APIError

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    "primary_protocol",
    "disease_model",
    "methods_tool",
    "review",
    "tangential",
    "not_relevant",
}

PROMPT_PATH = Path(__file__).parent / "prompts" / "triage_system.txt"
DEFAULT_DATA_DIR = Path("data/db")
DEFAULT_OUTPUT = Path("data/triage/triage_results.jsonl")
DEFAULT_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PaperMetadata:
    pmc_id: str
    pmid: str | None
    doi: str | None
    title: str
    abstract: str
    article_type: str | None
    file_path: str


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _extract_abstract_text(abstract_elem: ET.Element) -> str:
    """Extract text from an <abstract> element, handling sectioned and plain formats."""
    parts: list[str] = []
    sections = abstract_elem.findall("sec")
    if sections:
        for sec in sections:
            title_el = sec.find("title")
            if title_el is not None:
                title_text = "".join(title_el.itertext()).strip()
                if title_text:
                    parts.append(title_text)
            for p in sec.findall("p"):
                p_text = "".join(p.itertext()).strip()
                if p_text:
                    parts.append(p_text)
    else:
        for p in abstract_elem.findall("p"):
            p_text = "".join(p.itertext()).strip()
            if p_text:
                parts.append(p_text)

    if not parts:
        # Fallback: grab all text content
        fallback = "".join(abstract_elem.itertext()).strip()
        if fallback:
            parts.append(fallback)

    return "\n".join(parts)


def parse_pmc_xml(path: str | Path) -> PaperMetadata | None:
    """Parse a PMC XML file and extract metadata. Returns None if unparseable."""
    path = Path(path)
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        logger.warning("Malformed XML, skipping: %s", path)
        return None

    root = tree.getroot()

    # Article type from root <article> element
    article_type = None
    article_el = root if root.tag == "article" else root.find(".//article")
    if article_el is not None:
        article_type = article_el.get("article-type")

    # Find article-meta
    meta = root.find(".//article-meta")
    if meta is None:
        logger.warning("No article-meta found: %s", path)
        return None

    # Extract IDs
    pmc_id = pmid = doi = None
    for aid in meta.findall("article-id"):
        id_type = aid.get("pub-id-type", "")
        text = (aid.text or "").strip()
        if id_type == "pmcid" and not pmc_id:
            pmc_id = text
        elif id_type == "pmid" and not pmid:
            pmid = text
        elif id_type == "doi" and not doi:
            doi = text

    if not pmc_id:
        # Derive from filename
        pmc_id = path.stem

    # Title
    title_el = meta.find(".//article-title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""
    if not title:
        logger.warning("No title found: %s", path)
        return None

    # Abstract — take the first non-empty abstract
    abstract = ""
    for ab in meta.findall("abstract"):
        abstract = _extract_abstract_text(ab)
        if abstract:
            break

    if not abstract:
        logger.debug("No abstract found: %s", path)
        # Still return metadata so caller can record it with an error
        return PaperMetadata(
            pmc_id=pmc_id, pmid=pmid, doi=doi,
            title=title, abstract="", article_type=article_type,
            file_path=str(path),
        )

    return PaperMetadata(
        pmc_id=pmc_id, pmid=pmid, doi=doi,
        title=title, abstract=abstract, article_type=article_type,
        file_path=str(path),
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_xml_files(data_dir: Path) -> list[Path]:
    """Walk sharded directories under data_dir and collect .xml file paths."""
    xml_files: list[Path] = []
    for dirpath, _, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.endswith(".xml"):
                xml_files.append(Path(dirpath) / fn)
    xml_files.sort()
    return xml_files


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

def load_completed_ids(output_file: Path) -> set[str]:
    """Read JSONL output and return set of already-processed PMC IDs."""
    done: set[str] = set()
    if not output_file.exists():
        return done
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                err = rec.get("error")
                if "pmc_id" in rec and (not err or err == "no_abstract"):
                    done.add(rec["pmc_id"])
            except json.JSONDecodeError:
                continue
    return done


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    return PROMPT_PATH.read_text().strip()


def build_user_message(meta: PaperMetadata) -> str:
    """Format paper metadata as a user message for the LLM."""
    parts = [f"Title: {meta.title}"]
    if meta.article_type:
        parts.append(f"Article type: {meta.article_type}")
    parts.append(f"Abstract:\n{meta.abstract}")
    return "\n\n".join(parts)


def parse_llm_response(raw: dict, pmc_id: str) -> dict[str, Any]:
    """Extract and validate the classification result from an LLM response."""
    try:
        choice = raw["choices"][0]
        content = choice["message"]["content"] or ""
        if not content.strip():
            finish = choice.get("finish_reason", "unknown")
            logger.warning(
                "Empty LLM content for %s (finish_reason=%s)", pmc_id, finish,
            )
            return {"error": f"empty_response: finish_reason={finish}"}
        result = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse LLM response for %s: %s\nRaw: %s", pmc_id, exc, json.dumps(raw)[:500])
        return {"error": f"parse_failed: {exc}"}

    category = result.get("category", "")
    if category not in VALID_CATEGORIES:
        logger.warning("Invalid category '%s' for %s", category, pmc_id)
        return {"error": f"invalid_category: {category}", **result}

    return {
        "category": result.get("category"),
        "confidence": result.get("confidence"),
        "reasoning": result.get("reasoning"),
        "base_protocols": result.get("base_protocols", []),
        "supplement_signal": result.get("supplement_signal", False),
        "key_cell_types": result.get("key_cell_types", []),
        "disease_context": result.get("disease_context"),
        "error": None,
    }


async def classify_batch(
    client: OpenRouterClient,
    papers: list[PaperMetadata],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Classify a batch of papers via the LLM. Returns list of result dicts."""
    message_batches = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(p)},
        ]
        for p in papers
    ]

    raw_results = await client.complete_batch(
        message_batches,
        response_format={"type": "json_object"},
    )

    results: list[dict[str, Any]] = []
    for paper, raw in zip(papers, raw_results):
        if isinstance(raw, (APIError, Exception)):
            logger.warning("API error for %s: %s", paper.pmc_id, raw)
            results.append({"error": f"api_error: {raw}"})
        else:
            results.append(parse_llm_response(raw, paper.pmc_id))
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _make_record(meta: PaperMetadata, classification: dict[str, Any]) -> dict:
    return {
        "pmc_id": meta.pmc_id,
        "doi": meta.doi,
        "pmid": meta.pmid,
        "title": meta.title,
        "category": classification.get("category"),
        "confidence": classification.get("confidence"),
        "reasoning": classification.get("reasoning"),
        "base_protocols": classification.get("base_protocols", []),
        "supplement_signal": classification.get("supplement_signal", False),
        "key_cell_types": classification.get("key_cell_types", []),
        "disease_context": classification.get("disease_context"),
        "error": classification.get("error"),
    }


def append_records(output_file: Path, records: list[dict]) -> None:
    """Append records to JSONL file, flushing after write."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(output_file: Path) -> None:
    """Print category distribution from JSONL output."""
    if not output_file.exists():
        print(f"No output file found at {output_file}")
        return

    counts: dict[str, int] = {}
    errors = 0
    total = 0
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                total += 1
                cat = rec.get("category")
                if rec.get("error"):
                    errors += 1
                if cat:
                    counts[cat] = counts.get(cat, 0) + 1
            except json.JSONDecodeError:
                continue

    print(f"\nTriage results summary ({output_file}):")
    print(f"  Total papers: {total}")
    for cat in sorted(counts, key=lambda c: -counts[c]):
        pct = counts[cat] / total * 100 if total else 0
        print(f"  {cat:20s}: {counts[cat]:5d}  ({pct:.1f}%)")
    if errors:
        print(f"  {'errors':20s}: {errors:5d}  ({errors / total * 100:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_triage(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_file: Path = DEFAULT_OUTPUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    single: str | None = None,
) -> None:
    """Run the triage classification pipeline."""
    system_prompt = _load_system_prompt()

    # Single-file mode
    if single:
        meta = parse_pmc_xml(single)
        if meta is None:
            print(f"Failed to parse {single}")
            return
        if not meta.abstract:
            print(f"No abstract in {single}")
            rec = _make_record(meta, {"category": "not_relevant", "error": "no_abstract"})
            print(json.dumps(rec, indent=2))
            return

        async with OpenRouterClient() as client:
            results = await classify_batch(client, [meta], system_prompt)
        rec = _make_record(meta, results[0])
        print(json.dumps(rec, indent=2))
        return

    # Discover files
    print(f"Discovering XML files in {data_dir}...")
    xml_files = discover_xml_files(data_dir)
    print(f"Found {len(xml_files)} XML files")

    # Load completed IDs for resumability
    completed = load_completed_ids(output_file)
    if completed:
        print(f"Resuming: {len(completed)} papers already processed")

    # Parse metadata
    print("Parsing XML metadata...")
    papers: list[PaperMetadata] = []
    skipped_parse = 0
    skipped_done = 0
    no_abstract: list[PaperMetadata] = []

    for fp in xml_files:
        meta = parse_pmc_xml(fp)
        if meta is None:
            skipped_parse += 1
            continue
        if meta.pmc_id in completed:
            skipped_done += 1
            continue
        if not meta.abstract:
            no_abstract.append(meta)
            continue
        papers.append(meta)

    print(f"  Parseable with abstract: {len(papers)}")
    print(f"  No abstract: {len(no_abstract)}")
    print(f"  Already processed: {skipped_done}")
    print(f"  Unparseable: {skipped_parse}")

    # Write no-abstract papers immediately
    if no_abstract:
        no_abs_records = [
            _make_record(m, {"category": "not_relevant", "error": "no_abstract"})
            for m in no_abstract
        ]
        append_records(output_file, no_abs_records)
        print(f"  Wrote {len(no_abs_records)} no-abstract records")

    if limit is not None:
        papers = papers[:limit]
        print(f"  Limited to {len(papers)} papers")

    if not papers:
        print("No papers to classify.")
        print_summary(output_file)
        return

    # Process in batches
    total_batches = (len(papers) + batch_size - 1) // batch_size
    print(f"\nClassifying {len(papers)} papers in {total_batches} batches...")

    failed: list[PaperMetadata] = []

    async with OpenRouterClient() as client:
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(papers))
            batch = papers[start:end]

            print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch)} papers)...", end=" ", flush=True)
            results = await classify_batch(client, batch, system_prompt)

            ok_records: list[dict] = []
            for meta, res in zip(batch, results):
                if res.get("error"):
                    failed.append(meta)
                else:
                    ok_records.append(_make_record(meta, res))
            append_records(output_file, ok_records)
            print("done")

        # Retry failures one-at-a-time
        if failed:
            print(f"\nRetrying {len(failed)} failed papers individually...")
            still_failed = 0
            for meta in failed:
                res = (await classify_batch(client, [meta], system_prompt))[0]
                if res.get("error"):
                    still_failed += 1
                    logger.warning("Retry failed for %s: %s", meta.pmc_id, res["error"])
                    append_records(output_file, [_make_record(meta, res)])
                else:
                    append_records(output_file, [_make_record(meta, res)])
            if still_failed:
                print(f"  {still_failed} papers still failed after retry")
            else:
                print("  All retries succeeded")

    print_summary(output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triage-classify PMC papers by abstract using DeepSeek v3",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Root directory containing sharded PMC XML files",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="JSONL output path",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Papers per LLM batch",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N papers (for testing)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print summary of existing results and exit",
    )
    parser.add_argument(
        "--single", type=str, default=None,
        help="Classify a single XML file and print result",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.summary_only:
        print_summary(args.output)
        return

    asyncio.run(run_triage(
        data_dir=args.data_dir,
        output_file=args.output,
        batch_size=args.batch_size,
        limit=args.limit,
        single=args.single,
    ))


if __name__ == "__main__":
    main()
