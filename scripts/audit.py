"""Interactive quality audit for extracted protocols.

Stratified sampling by confidence bucket and triage category.
Displays protocol details alongside source text for manual validation.

Usage:
    python3 audit.py                    # default: 5 per bucket × 4 categories
    python3 audit.py --sample 3         # 3 per bucket per category
    python3 audit.py --summary          # show summary of existing audit results
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_layer.database import PipelineDB

AUDIT_OUTPUT = Path("data/audit/audit_results.jsonl")
PARSED_TEXT_DIR = Path("data/db/parsed_texts")

CONFIDENCE_BUCKETS = {
    "high":  (0.85, 1.01),
    "good":  (0.70, 0.85),
    "fair":  (0.50, 0.70),
    "low":   (0.00, 0.50),
}

VALID_RATINGS = {"correct", "mostly_correct", "partially_correct", "incorrect", "skip"}
RATING_SCORES = {"correct": 1.0, "mostly_correct": 0.75, "partially_correct": 0.5, "incorrect": 0.0}


def sample_protocols(db: PipelineDB, per_bucket: int = 5) -> list[dict]:
    """Stratified sampling: up to per_bucket protocols per confidence bucket per category."""
    rows = db._conn.execute(
        """SELECT pr.*, p.pmc_id, p.doi, p.title AS paper_title,
                  p.triage_category, p.parsed_text_path
           FROM protocols pr
           JOIN papers p ON p.id = pr.paper_id
           WHERE p.triage_category IN ('primary_protocol', 'disease_model', 'methods_tool', 'review')
           ORDER BY pr.id"""
    ).fetchall()

    # Group by (category, bucket)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        d = dict(row)
        conf = d.get("extraction_confidence") or 0
        bucket = "low"
        for name, (lo, hi) in CONFIDENCE_BUCKETS.items():
            if lo <= conf < hi:
                bucket = name
                break
        cat = d.get("triage_category", "unknown")
        groups[(cat, bucket)].append(d)

    sampled = []
    for (cat, bucket), pool in sorted(groups.items()):
        n = min(per_bucket, len(pool))
        sampled.extend(random.sample(pool, n))

    random.shuffle(sampled)
    return sampled


def display_protocol(proto: dict, index: int, total: int) -> None:
    """Display a protocol for audit review."""
    print(f"\n{'='*70}")
    print(f"Protocol {index}/{total}")
    print(f"{'='*70}")

    # Paper metadata
    print(f"PMC ID:    {proto.get('pmc_id', '?')}")
    print(f"DOI:       {proto.get('doi', '?')}")
    print(f"Title:     {proto.get('paper_title', '?')}")
    print(f"Category:  {proto.get('triage_category', '?')}")
    print(f"Arm:       {proto.get('protocol_arm', '?')}")
    print(f"Confidence: {proto.get('extraction_confidence', '?')}")
    print(f"Pass:      {proto.get('pass_number', '?')}")

    # Cell source
    cell_source = proto.get("cell_source")
    if isinstance(cell_source, str):
        try:
            cell_source = json.loads(cell_source)
        except (json.JSONDecodeError, TypeError):
            pass
    if cell_source:
        print(f"\nCell Source: {json.dumps(cell_source, indent=2) if isinstance(cell_source, dict) else cell_source}")

    # Culture system
    culture = proto.get("culture_system")
    if isinstance(culture, str):
        try:
            culture = json.loads(culture)
        except (json.JSONDecodeError, TypeError):
            pass
    if culture:
        print(f"\nCulture System: {json.dumps(culture, indent=2) if isinstance(culture, dict) else culture}")

    # Stages
    stages = proto.get("stages")
    if isinstance(stages, str):
        try:
            stages = json.loads(stages)
        except (json.JSONDecodeError, TypeError):
            pass
    if stages and isinstance(stages, list):
        print(f"\nStages ({len(stages)}):")
        for s in stages:
            if not isinstance(s, dict):
                continue
            name = s.get("stage_name", "?")
            dur = s.get("duration_days", "?")
            factors = s.get("growth_factors", [])
            factor_names = [f.get("name", "?") if isinstance(f, dict) else str(f)
                           for f in (factors or [])]
            small_mols = s.get("small_molecules", [])
            sm_names = [m.get("name", "?") if isinstance(m, dict) else str(m)
                       for m in (small_mols or [])]
            print(f"  {name} ({dur} days)")
            if factor_names:
                print(f"    Growth factors: {', '.join(factor_names)}")
            if sm_names:
                print(f"    Small molecules: {', '.join(sm_names)}")

    # Endpoint assessment
    endpoint = proto.get("endpoint_assessment")
    if isinstance(endpoint, str):
        try:
            endpoint = json.loads(endpoint)
        except (json.JSONDecodeError, TypeError):
            pass
    if endpoint and isinstance(endpoint, dict):
        markers = endpoint.get("markers", [])
        assays = endpoint.get("functional_assays", [])
        if markers:
            print(f"\nEndpoint Markers ({len(markers)}):")
            for m in markers[:10]:
                if isinstance(m, dict):
                    name = m.get("marker_name", "?")
                    val = m.get("value", "")
                    vtype = m.get("value_type", "")
                    print(f"  {name}: {val} ({vtype})")
        if assays:
            print(f"\nFunctional Assays ({len(assays)}):")
            for a in assays[:5]:
                if isinstance(a, dict):
                    print(f"  {a.get('assay_name', '?')}: {a.get('value', '')} {a.get('unit', '')}")

    # Incomplete flags
    flags = proto.get("incomplete_flags")
    if isinstance(flags, str):
        try:
            flags = json.loads(flags)
        except (json.JSONDecodeError, TypeError):
            pass
    if flags:
        print(f"\nIncomplete Flags ({len(flags)}):")
        for f in flags:
            if isinstance(f, dict):
                print(f"  [{f.get('reason', '?')}] {f.get('field', '?')}: {f.get('details', '')[:100]}")

    # Source text snippet
    text_path = proto.get("parsed_text_path")
    if text_path and Path(text_path).exists():
        text = Path(text_path).read_text()
        # Find Methods section
        methods_start = -1
        for marker in ["## Methods", "## Materials and Methods", "## METHODS",
                       "## Experimental Procedures", "## Materials and methods"]:
            idx = text.find(marker)
            if idx >= 0:
                methods_start = idx
                break
        if methods_start >= 0:
            snippet = text[methods_start:methods_start + 2000]
        else:
            snippet = text[:2000]
        print(f"\n--- Source Text (first 2000 chars of Methods) ---")
        print(snippet)
        print(f"--- End Source Text ---")


def get_rating() -> tuple[str, str]:
    """Prompt user for a rating and optional notes."""
    print(f"\nRate this extraction:")
    print(f"  [c] correct  [m] mostly_correct  [p] partially_correct  [i] incorrect  [s] skip  [q] quit")

    while True:
        choice = input("Rating> ").strip().lower()
        if choice == "q":
            return "quit", ""
        rating_map = {"c": "correct", "m": "mostly_correct", "p": "partially_correct",
                      "i": "incorrect", "s": "skip"}
        if choice in rating_map:
            rating = rating_map[choice]
            notes = ""
            if rating != "skip":
                notes = input("Notes (optional, press Enter to skip)> ").strip()
            return rating, notes
        print("Invalid choice. Use c/m/p/i/s/q.")


def run_audit(db: PipelineDB, per_bucket: int = 5) -> None:
    """Run interactive audit session."""
    samples = sample_protocols(db, per_bucket)
    if not samples:
        print("No protocols found for audit.")
        return

    print(f"\nAudit session: {len(samples)} protocols sampled")
    print(f"Results will be saved to {AUDIT_OUTPUT}")

    completed = 0
    for i, proto in enumerate(samples, 1):
        display_protocol(proto, i, len(samples))
        rating, notes = get_rating()

        if rating == "quit":
            print(f"\nAudit stopped. Completed {completed}/{len(samples)}.")
            break

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol_id": proto["id"],
            "paper_id": proto["paper_id"],
            "pmc_id": proto.get("pmc_id"),
            "triage_category": proto.get("triage_category"),
            "extraction_confidence": proto.get("extraction_confidence"),
            "rating": rating,
            "notes": notes,
        }

        AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_OUTPUT, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed += 1

    if completed > 0:
        print_summary()


def print_summary() -> None:
    """Print summary of audit results."""
    if not AUDIT_OUTPUT.exists():
        print("No audit results found.")
        return

    results = []
    with open(AUDIT_OUTPUT) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not results:
        print("No audit results found.")
        return

    # Filter out skips for accuracy calculation
    rated = [r for r in results if r["rating"] != "skip"]

    print(f"\n{'='*50}")
    print(f"Audit Summary ({len(results)} total, {len(rated)} rated)")
    print(f"{'='*50}")

    # Overall accuracy
    if rated:
        scores = [RATING_SCORES.get(r["rating"], 0) for r in rated]
        avg = sum(scores) / len(scores)
        print(f"\nOverall accuracy score: {avg:.2f}")

    # By confidence bucket
    print(f"\nAccuracy by confidence bucket:")
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for r in rated:
        conf = r.get("extraction_confidence") or 0
        bucket = "low"
        for name, (lo, hi) in CONFIDENCE_BUCKETS.items():
            if lo <= conf < hi:
                bucket = name
                break
        by_bucket[bucket].append(RATING_SCORES.get(r["rating"], 0))

    for bucket in ["high", "good", "fair", "low"]:
        scores = by_bucket.get(bucket, [])
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {bucket:6s}: {avg:.2f} (n={len(scores)})")
        else:
            print(f"  {bucket:6s}: no data")

    # By triage category
    print(f"\nAccuracy by triage category:")
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rated:
        cat = r.get("triage_category", "unknown")
        by_cat[cat].append(RATING_SCORES.get(r["rating"], 0))

    for cat in sorted(by_cat.keys()):
        scores = by_cat[cat]
        avg = sum(scores) / len(scores)
        print(f"  {cat:25s}: {avg:.2f} (n={len(scores)})")

    # Rating distribution
    print(f"\nRating distribution:")
    dist: dict[str, int] = defaultdict(int)
    for r in results:
        dist[r["rating"]] += 1
    for rating in ["correct", "mostly_correct", "partially_correct", "incorrect", "skip"]:
        if rating in dist:
            print(f"  {rating:20s}: {dist[rating]}")

    # Correlation: confidence vs rating
    if len(rated) >= 5:
        print(f"\nConfidence vs Rating (mean confidence per rating):")
        conf_by_rating: dict[str, list[float]] = defaultdict(list)
        for r in rated:
            conf = r.get("extraction_confidence") or 0
            conf_by_rating[r["rating"]].append(conf)
        for rating in ["correct", "mostly_correct", "partially_correct", "incorrect"]:
            confs = conf_by_rating.get(rating, [])
            if confs:
                avg_conf = sum(confs) / len(confs)
                print(f"  {rating:20s}: avg confidence = {avg_conf:.3f} (n={len(confs)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive protocol quality audit")
    parser.add_argument("--sample", type=int, default=5,
                        help="Protocols per confidence bucket per category (default: 5)")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary of existing audit results (no new auditing)")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    db = PipelineDB()
    try:
        run_audit(db, per_bucket=args.sample)
    finally:
        db.close()


if __name__ == "__main__":
    main()
