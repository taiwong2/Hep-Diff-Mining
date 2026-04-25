"""Batch retroactive grounding cleanup for existing extracted protocols.

Validates all growth factors, small molecules, and markers against source text
and removes hallucinated items not found in the paper.

Usage:
    python3 grounding_cleanup.py --dry-run        # report only, no DB changes
    python3 grounding_cleanup.py                   # apply removals to DB
    python3 grounding_cleanup.py --limit 10        # process first 10 protocols
    python3 grounding_cleanup.py --single PMC10114490  # single paper
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from data_layer.database import PipelineDB
from data_layer.grounding import ground_protocol

logger = logging.getLogger(__name__)

OUTPUT_JSONL = Path("data/logs/grounding_results.jsonl")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_cleanup(
    db: PipelineDB,
    dry_run: bool = False,
    limit: int | None = None,
    single: str | None = None,
) -> None:
    """Run grounding cleanup on all protocols."""

    # Get protocols with paper info
    if single:
        paper = db.get_paper(pmc_id=single)
        if not paper:
            print(f"Paper {single} not found")
            return
        protocols = db.get_protocols_for_paper(paper["id"])
        papers_by_id = {paper["id"]: paper}
    else:
        rows = db._conn.execute(
            """SELECT pr.id AS proto_id, pr.paper_id, pr.protocol_arm,
                      pr.stages, pr.endpoint_assessment,
                      pr.extraction_notes, pr.extraction_confidence,
                      pr.cell_source, pr.culture_system, pr.modifications,
                      pr.step_sources, pr.base_protocol_doi,
                      pr.incomplete_flags, pr.is_optimized, pr.pass_number,
                      p.pmc_id, p.title, p.parsed_text_path, p.supplement_text_path
               FROM protocols pr
               JOIN papers p ON p.id = pr.paper_id
               WHERE p.parsed_text_path IS NOT NULL
               ORDER BY pr.id"""
        ).fetchall()

        if limit:
            rows = rows[:limit]

        protocols = []
        papers_by_id: dict[int, dict] = {}
        for row in rows:
            d = dict(row)
            # Deserialize JSON fields
            for k in ("stages", "endpoint_assessment", "cell_source", "culture_system",
                       "modifications", "step_sources", "incomplete_flags"):
                if d.get(k) and isinstance(d[k], str):
                    try:
                        d[k] = json.loads(d[k])
                    except (json.JSONDecodeError, TypeError):
                        pass
            protocols.append(d)
            if d["paper_id"] not in papers_by_id:
                papers_by_id[d["paper_id"]] = {
                    "pmc_id": d["pmc_id"],
                    "title": d["title"],
                    "parsed_text_path": d["parsed_text_path"],
                    "supplement_text_path": d["supplement_text_path"],
                }

    if not protocols:
        print("No protocols found.")
        return

    print(f"Checking {len(protocols)} protocol(s) from {len(papers_by_id)} paper(s)...")
    if dry_run:
        print("DRY RUN — no changes will be made\n")

    # Cache loaded texts to avoid re-reading
    text_cache: dict[str, str] = {}

    total_checked = 0
    total_with_removals = 0
    total_removed = 0
    removal_counter: Counter = Counter()  # category -> count
    term_counter: Counter = Counter()  # (category, term) -> count

    for proto in protocols:
        proto_id = proto.get("proto_id") or proto.get("id")
        paper_id = proto.get("paper_id")

        if single:
            paper_info = papers_by_id[paper_id]
        else:
            paper_info = papers_by_id.get(paper_id, {})

        pmc_id = paper_info.get("pmc_id") or proto.get("pmc_id", "?")
        arm = proto.get("protocol_arm", "?")
        text_path = paper_info.get("parsed_text_path")

        if not text_path or not Path(text_path).exists():
            continue

        # Load paper text (cached)
        if text_path not in text_cache:
            text_cache[text_path] = Path(text_path).read_text()
        paper_text = text_cache[text_path]

        # Load supplement text (cached)
        supp_path = paper_info.get("supplement_text_path")
        supp_text = None
        if supp_path and Path(supp_path).exists():
            if supp_path not in text_cache:
                text_cache[supp_path] = Path(supp_path).read_text()
            supp_text = text_cache[supp_path]

        cleaned, removals = ground_protocol(proto, paper_text, supp_text)
        total_checked += 1

        if removals:
            total_with_removals += 1
            total_removed += len(removals)

            for r in removals:
                removal_counter[r["category"]] += 1
                term_counter[(r["category"], r["term"])] += 1

            print(f"  {pmc_id}/{arm}: {len(removals)} removed — "
                  + ", ".join(f"{r['term']}({r['category'][:2]})" for r in removals))

            if not dry_run:
                # Update DB
                updates: dict = {}
                if cleaned.get("stages") is not None:
                    updates["stages"] = cleaned["stages"]
                if cleaned.get("endpoint_assessment") is not None:
                    updates["endpoint_assessment"] = cleaned["endpoint_assessment"]

                # Append grounding note to extraction_notes
                existing_notes = proto.get("extraction_notes", "") or ""
                removed_terms = [f"{r['term']}" for r in removals]
                grounding_note = f"Grounding removed: {', '.join(removed_terms)}"
                updates["extraction_notes"] = (
                    f"{existing_notes} | {grounding_note}".strip(" | ")
                )

                db.update_protocol(proto_id, updates)

                _append_jsonl(OUTPUT_JSONL, {
                    "pmc_id": pmc_id,
                    "protocol_id": proto_id,
                    "protocol_arm": arm,
                    "removals": removals,
                })

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Grounding {'Report (DRY RUN)' if dry_run else 'Results'}")
    print(f"{'=' * 60}")
    print(f"Protocols checked:        {total_checked}")
    print(f"Protocols with removals:  {total_with_removals}")
    print(f"Total items removed:      {total_removed}")

    if removal_counter:
        print(f"\nBy category:")
        for cat, count in removal_counter.most_common():
            print(f"  {cat}: {count}")

        print(f"\nTop removed terms:")
        for (cat, term), count in term_counter.most_common(20):
            print(f"  {term} ({cat}): {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch grounding cleanup: remove hallucinated terms from protocols",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be removed without writing to DB")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max protocols to process")
    parser.add_argument("--single", type=str, default=None, metavar="PMC_ID",
                        help="Process single paper by PMC ID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db = PipelineDB()
    try:
        run_cleanup(db, dry_run=args.dry_run, limit=args.limit, single=args.single)
    finally:
        db.close()


if __name__ == "__main__":
    main()
