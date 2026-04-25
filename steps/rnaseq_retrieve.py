"""Phase 3: Processed data retrieval.

Downloads expression data from GEO matrices, parses supplement expression
tables, and extracts target gene panel values. Stores results in the
expression_values table.

Usage:
    python3 rnaseq_retrieve.py                              # all sources
    python3 rnaseq_retrieve.py --source geo --limit 3       # GEO matrices only
    python3 rnaseq_retrieve.py --source supplement --limit 3 # supplement tables only
    python3 rnaseq_retrieve.py --dry-run                    # show eligible papers
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import re

from data_layer.database import PipelineDB
from data_layer.gene_panel import ALL_TARGET_GENES, GENE_ALIASES, resolve_alias
from data_layer.geo_matrix_fetcher import (
    download_series_matrix,
    download_supplementary_file,
    parse_series_matrix,
    parse_series_matrix_metadata,
    parse_count_table,
)
from data_layer.geo_sample_mapper import _extract_day, _extract_stage_alias
from data_layer.supplement_expression_parser import (
    find_expression_files,
    parse_deg_table,
    parse_expression_table,
)

logger = logging.getLogger(__name__)

OUTPUT_JSONL = Path("data/logs/expression_extraction_results.jsonl")
GEO_MATRIX_DIR = Path("data/expression_data/geo_matrices")
MARKERS_OUTPUT = Path("data/expression_data/extracted_markers.json")


def _get_papers_by_source(db: PipelineDB, source: str) -> dict[str, list[dict]]:
    """Group papers by data source for expression retrieval."""
    papers_by_source: dict[str, list[dict]] = {
        "geo_with_matrix": [],
        "geo_raw_only": [],
        "supplement": [],
        "text_only": [],
    }

    # Papers with RNA-seq data
    rows = db._conn.execute(
        """SELECT p.*, rm.data_availability, rm.accessions, rm.normalization
           FROM papers p
           JOIN rnaseq_metadata rm ON rm.paper_id = p.id
           WHERE rm.has_rnaseq = 1
           AND p.id NOT IN (
               SELECT DISTINCT paper_id FROM expression_values
           )
           ORDER BY p.id"""
    ).fetchall()

    for row in rows:
        paper = dict(row)
        da = paper.get("data_availability")
        if da and isinstance(da, str):
            try:
                da = json.loads(da)
            except json.JSONDecodeError:
                da = {}

        classification = (da or {}).get("classification", "")

        if classification == "geo_with_matrix":
            papers_by_source["geo_with_matrix"].append(paper)
        elif classification in ("geo_raw_only",):
            papers_by_source["geo_raw_only"].append(paper)
        else:
            # Check if paper has supplement files with expression data
            supp_dir = paper.get("supplement_dir")
            if supp_dir and supp_dir != "none" and Path(supp_dir).exists():
                papers_by_source["supplement"].append(paper)
            else:
                papers_by_source["text_only"].append(paper)

    # Filter by requested source
    if source == "geo":
        papers_by_source = {
            "geo_with_matrix": papers_by_source["geo_with_matrix"],
            "geo_raw_only": papers_by_source["geo_raw_only"],
        }
    elif source == "supplement":
        papers_by_source = {"supplement": papers_by_source["supplement"]}

    return papers_by_source


def _build_sample_lookup(
    db: PipelineDB, paper_id: int, protocol_id: int | None,
    matrix_paths: list[Path] | None = None,
) -> dict:
    """Build a unified lookup: any sample key (GSM ID or sample title) → stage info.

    Keys include GSM IDs, sample titles from geo_samples, and sample titles
    parsed from series matrix metadata so that series matrix columns (GSM IDs),
    supplementary file columns (sample labels), and other naming conventions
    can all be resolved to stage/day info.
    """
    stage_map: dict[str, dict] = {}
    if not protocol_id:
        return stage_map

    mappings = db.get_sample_stage_mappings(protocol_id)
    # Index by GSM ID
    gsm_to_stage = {m["gsm_id"]: m for m in mappings}
    stage_map.update(gsm_to_stage)

    # Also index by sample_title and source_name from geo_samples
    geo_accs = db.get_geo_accessions(paper_id)
    for acc in geo_accs:
        samples = db.get_geo_samples(acc["id"])
        for s in samples:
            gsm_id = s["gsm_id"]
            if gsm_id not in gsm_to_stage:
                continue
            info = gsm_to_stage[gsm_id]
            title = (s.get("sample_title") or "").strip()
            if title:
                stage_map[title] = info
            # Also add source_name as a key
            source = (s.get("source_name") or "").strip()
            if source and source not in stage_map:
                stage_map[source] = info

    # Parse series matrix metadata for additional GSM → title mappings
    # This captures the exact column titles used in matrix files, which
    # often match supplementary count file column headers
    for mp in (matrix_paths or []):
        if not mp or not mp.exists():
            continue
        gsm_title_map = parse_series_matrix_metadata(mp)
        for gsm_id, title in gsm_title_map.items():
            if gsm_id in gsm_to_stage and title not in stage_map:
                stage_map[title] = gsm_to_stage[gsm_id]

    return stage_map


def retrieve_geo_expression(db: PipelineDB, paper: dict) -> int:
    """Download GEO matrix and extract target gene values. Returns count of values stored."""
    paper_id = paper["id"]
    pmc_id = paper["pmc_id"]

    # Get GEO accessions
    geo_accs = db.get_geo_accessions(paper_id)
    own_data_accs = [a for a in geo_accs if a.get("context") == "own_data"]
    if not own_data_accs:
        own_data_accs = geo_accs

    # Also check repository_metadata for accessions with processed matrices
    repo_metas = db.get_repository_metadata(paper_id)

    # Get protocol mapping if available
    protocols = db.get_protocols_for_paper(paper_id)
    protocol_id = protocols[0]["id"] if protocols else None

    # Get normalization info from rnaseq_metadata
    rm_meta = db.get_rnaseq_metadata(paper_id)
    unit = (rm_meta or {}).get("normalization") or "unknown"

    values_stored = 0

    # Download all series matrices first so metadata is available for lookup
    matrix_paths: list[Path] = []
    for acc in own_data_accs:
        mp = download_series_matrix(acc["gse_id"], GEO_MATRIX_DIR)
        matrix_paths.append(mp)

    # Build unified sample lookup (GSM IDs + sample titles + matrix metadata → stage info)
    valid_paths = [p for p in matrix_paths if p]
    sample_lookup = _build_sample_lookup(db, paper_id, protocol_id, valid_paths)

    for acc, matrix_path in zip(own_data_accs, matrix_paths):
        gse_id = acc["gse_id"]

        if not matrix_path:
            continue

        gene_data = parse_series_matrix(matrix_path, ALL_TARGET_GENES)
        if not gene_data:
            logger.info("[%s] No target genes found in %s matrix", pmc_id, gse_id)
            continue

        batch = []
        for gene, sample_values in gene_data.items():
            canonical = resolve_alias(gene)
            for sample_key, value in sample_values.items():
                if value is None:
                    continue

                stage_info = sample_lookup.get(sample_key, {})
                batch.append({
                    "paper_id": paper_id,
                    "protocol_id": protocol_id,
                    "gene_symbol": canonical,
                    "gene_alias": gene if gene != canonical else None,
                    "value": value,
                    "unit": unit,
                    "condition_label": stage_info.get("condition_label") or sample_key,
                    "time_point_day": stage_info.get("time_point_day"),
                    "source_type": "geo_matrix",
                    "source_detail": f"{gse_id} series matrix",
                    "confidence": 0.95,
                })

        if batch:
            values_stored += db.store_expression_values_batch(batch)

    # Also check for supplementary count matrices from repository_metadata
    for rm in repo_metas:
        if not rm.get("has_processed_matrix"):
            continue
        supp_files = rm.get("supplementary_files") or []
        for sf in supp_files:
            if not sf.get("has_count_matrix"):
                continue
            url = sf.get("url", "")
            if not url:
                continue
            filename = sf.get("filename", "")
            dl_path = download_supplementary_file(url, GEO_MATRIX_DIR, filename)
            if dl_path:
                gene_data = parse_count_table(dl_path, ALL_TARGET_GENES)
                batch = []
                for gene, sample_values in gene_data.items():
                    canonical = resolve_alias(gene)
                    for sample_label, value in sample_values.items():
                        if value is None:
                            continue
                        stage_info = sample_lookup.get(sample_label, {})
                        batch.append({
                            "paper_id": paper_id,
                            "protocol_id": protocol_id,
                            "gene_symbol": canonical,
                            "gene_alias": gene if gene != canonical else None,
                            "value": value,
                            "unit": "unknown",
                            "condition_label": stage_info.get("condition_label") or sample_label,
                            "time_point_day": stage_info.get("time_point_day"),
                            "source_type": "geo_matrix",
                            "source_detail": f"{rm['accession']} supplementary: {filename}",
                            "confidence": 0.9,
                        })
                if batch:
                    values_stored += db.store_expression_values_batch(batch)

    # Fallback: scan disk for downloaded supplementary files matching this paper's
    # GSE IDs that weren't reached via repository_metadata URLs.
    if values_stored == 0:
        already_tried = {p.name for p in matrix_paths if p}
        already_tried |= {
            sf.get("filename", "")
            for rm in repo_metas
            for sf in (rm.get("supplementary_files") or [])
            if sf.get("filename")
        }
        for acc in own_data_accs:
            gse_id = acc["gse_id"]
            for disk_file in sorted(GEO_MATRIX_DIR.glob(f"{gse_id}_*")):
                if disk_file.name in already_tried:
                    continue
                if disk_file.name.endswith("_series_matrix.txt"):
                    continue  # already handled above
                gene_data = parse_count_table(disk_file, ALL_TARGET_GENES)
                if not gene_data:
                    continue
                batch = []
                for gene, sample_values in gene_data.items():
                    canonical = resolve_alias(gene)
                    for sample_label, value in sample_values.items():
                        if value is None:
                            continue
                        stage_info = sample_lookup.get(sample_label, {})
                        batch.append({
                            "paper_id": paper_id,
                            "protocol_id": protocol_id,
                            "gene_symbol": canonical,
                            "gene_alias": gene if gene != canonical else None,
                            "value": value,
                            "unit": "unknown",
                            "condition_label": stage_info.get("condition_label") or sample_label,
                            "time_point_day": stage_info.get("time_point_day"),
                            "source_type": "geo_matrix",
                            "source_detail": f"{gse_id} disk fallback: {disk_file.name}",
                            "confidence": 0.85,
                        })
                if batch:
                    values_stored += db.store_expression_values_batch(batch)

        # If still nothing, try ALL accessions (including ambiguous context)
        if values_stored == 0:
            tried_gses = {a["gse_id"] for a in own_data_accs}
            for acc in geo_accs:
                if acc["gse_id"] in tried_gses:
                    continue
                gse_id = acc["gse_id"]
                for disk_file in sorted(GEO_MATRIX_DIR.glob(f"{gse_id}_*")):
                    if disk_file.name in already_tried:
                        continue
                    if disk_file.name.endswith("_series_matrix.txt"):
                        continue
                    gene_data = parse_count_table(disk_file, ALL_TARGET_GENES)
                    if not gene_data:
                        continue
                    batch = []
                    for gene, sample_values in gene_data.items():
                        canonical = resolve_alias(gene)
                        for sample_label, value in sample_values.items():
                            if value is None:
                                continue
                            stage_info = sample_lookup.get(sample_label, {})
                            batch.append({
                                "paper_id": paper_id,
                                "protocol_id": protocol_id,
                                "gene_symbol": canonical,
                                "gene_alias": gene if gene != canonical else None,
                                "value": value,
                                "unit": "unknown",
                                "condition_label": stage_info.get("condition_label") or sample_label,
                                "time_point_day": stage_info.get("time_point_day"),
                                "source_type": "geo_matrix",
                                "source_detail": f"{gse_id} disk fallback (all accs): {disk_file.name}",
                                "confidence": 0.8,
                            })
                    if batch:
                        values_stored += db.store_expression_values_batch(batch)

    return values_stored


def retrieve_supplement_expression(db: PipelineDB, paper: dict) -> int:
    """Parse supplement files for expression data. Returns count of values stored."""
    paper_id = paper["id"]
    pmc_id = paper["pmc_id"]
    supp_dir = paper.get("supplement_dir")

    if not supp_dir or supp_dir == "none" or not Path(supp_dir).exists():
        return 0

    expr_files = find_expression_files(Path(supp_dir))
    if not expr_files:
        return 0

    protocols = db.get_protocols_for_paper(paper_id)
    protocol_id = protocols[0]["id"] if protocols else None

    values_stored = 0

    for file_path, classification in expr_files:
        logger.info("[%s] Processing supplement %s (%s)", pmc_id, file_path.name, classification)

        if classification == "deg_list":
            degs = parse_deg_table(file_path, ALL_TARGET_GENES)
            comparison = _extract_comparison_from_filename(file_path.name)
            batch = []
            for deg in degs:
                gene = resolve_alias(deg["gene_symbol"])
                fc = deg.get("log2fc")
                padj = deg.get("padj")
                if fc is not None:
                    batch.append({
                        "paper_id": paper_id,
                        "protocol_id": protocol_id,
                        "gene_symbol": gene,
                        "value": fc,
                        "unit": "log2FC",
                        "comparison": comparison,
                        "padj": padj,
                        "source_type": "supplement_table",
                        "source_detail": f"Supplement DEG: {file_path.name}",
                        "confidence": 0.85,
                    })
            if batch:
                values_stored += db.store_expression_values_batch(batch)

        elif classification == "count_matrix":
            gene_data = parse_expression_table(file_path, ALL_TARGET_GENES)
            batch = []
            for gene, sample_values in gene_data.items():
                canonical = resolve_alias(gene)
                for condition, value in sample_values.items():
                    if value is None:
                        continue
                    batch.append({
                        "paper_id": paper_id,
                        "protocol_id": protocol_id,
                        "gene_symbol": canonical,
                        "gene_alias": gene if gene != canonical else None,
                        "value": value,
                        "unit": "unknown",
                        "condition_label": condition,
                        "source_type": "supplement_table",
                        "source_detail": f"Supplement expression: {file_path.name}",
                        "confidence": 0.8,
                    })
            if batch:
                values_stored += db.store_expression_values_batch(batch)

    return values_stored


def _extract_comparison_from_filename(name: str) -> str | None:
    """Extract a comparison label from a DEG filename.

    Matches patterns like 'DEG_HLC_vs_HEP.xlsx', 'diffexp_d20_vs_d0.csv'.
    """
    m = re.search(r'(\w+)[-_]vs[-_](\w+)', name, re.IGNORECASE)
    if m:
        return f"{m.group(1)} vs {m.group(2)}"
    return None


def _backfill_comparison_from_source_detail(db: PipelineDB) -> int:
    """Backfill comparison from source_detail filenames for existing DEG rows."""
    rows = db._conn.execute(
        """SELECT DISTINCT source_detail
           FROM expression_values
           WHERE comparison IS NULL
           AND source_type = 'supplement_table'
           AND source_detail LIKE '%DEG%'"""
    ).fetchall()

    updated = 0
    for row in rows:
        detail = row[0]
        if not detail:
            continue
        # Extract filename from source_detail like "Supplement DEG: filename.xlsx"
        parts = detail.rsplit(": ", 1)
        if len(parts) == 2:
            comparison = _extract_comparison_from_filename(parts[1])
            if comparison:
                cur = db._conn.execute(
                    """UPDATE expression_values
                       SET comparison = ?
                       WHERE source_detail = ? AND comparison IS NULL""",
                    (comparison, detail),
                )
                updated += cur.rowcount
    db._conn.commit()
    return updated


def _backfill_day_from_labels(db: PipelineDB) -> int:
    """Extract time_point_day from condition_label text for values still missing it.

    Uses the same regex patterns as geo_sample_mapper._extract_day() to pull
    day numbers from labels like 'WT D6 1' → 6, 'Day0_Rep1' → 0, etc.
    Returns count of values updated.
    """
    # Get distinct condition_labels that lack time_point_day
    rows = db._conn.execute(
        """SELECT DISTINCT condition_label
           FROM expression_values
           WHERE time_point_day IS NULL AND condition_label != ''"""
    ).fetchall()

    label_to_day: dict[str, int] = {}
    for row in rows:
        label = row[0]
        day = _extract_day(label)
        if day is not None:
            label_to_day[label] = day

    if not label_to_day:
        return 0

    updated = 0
    for label, day in label_to_day.items():
        cur = db._conn.execute(
            """UPDATE expression_values
               SET time_point_day = ?
               WHERE condition_label = ? AND time_point_day IS NULL""",
            (day, label),
        )
        updated += cur.rowcount
    db._conn.commit()

    logger.info("Backfilled time_point_day for %d values (%d distinct labels)",
                updated, len(label_to_day))
    return updated


def _backfill_day_from_mappings(db: PipelineDB) -> int:
    """Backfill time_point_day from geo_sample_stage_mappings.

    Joins expression_values with geo_sample_stage_mappings via protocol_id
    and condition_label to populate time_point_day where available.
    """
    cur = db._conn.execute("""
        UPDATE expression_values SET time_point_day = (
            SELECT m.time_point_day FROM geo_sample_stage_mappings m
            WHERE m.protocol_id = expression_values.protocol_id
            AND m.condition_label = expression_values.condition_label
            AND m.time_point_day IS NOT NULL
            LIMIT 1
        )
        WHERE time_point_day IS NULL
        AND protocol_id IS NOT NULL
        AND condition_label IS NOT NULL
        AND condition_label != ''
    """)
    db._conn.commit()
    updated = cur.rowcount
    if updated:
        logger.info("Backfilled time_point_day from mappings for %d values", updated)
    return updated


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _write_markers_json(db: PipelineDB) -> None:
    """Write extracted markers summary to JSON."""
    MARKERS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = db._conn.execute(
        """SELECT ev.paper_id, p.pmc_id, ev.gene_symbol, ev.value, ev.unit,
                  ev.condition_label, ev.source_type, ev.confidence
           FROM expression_values ev
           JOIN papers p ON p.id = ev.paper_id
           ORDER BY p.pmc_id, ev.gene_symbol"""
    ).fetchall()

    markers: dict[str, list[dict]] = {}
    for row in rows:
        r = dict(row)
        pmc_id = r.pop("pmc_id")
        if pmc_id not in markers:
            markers[pmc_id] = []
        markers[pmc_id].append(r)

    MARKERS_OUTPUT.write_text(json.dumps(markers, indent=2, ensure_ascii=False))
    logger.info("Wrote %d papers to %s", len(markers), MARKERS_OUTPUT)


def run(
    db: PipelineDB,
    limit: int | None = None,
    dry_run: bool = False,
    source: str = "all",
) -> None:
    papers_by_source = _get_papers_by_source(db, source)

    total = sum(len(v) for v in papers_by_source.values())
    if total == 0:
        print("No papers eligible for expression data retrieval.")
        return

    print(f"Found {total} paper(s) eligible for expression retrieval:")
    for src, papers in papers_by_source.items():
        if papers:
            print(f"  {src}: {len(papers)}")

    if dry_run:
        for src, papers in papers_by_source.items():
            if not papers:
                continue
            print(f"\n--- {src} ---")
            for p in papers[:20]:
                print(f"  {p['pmc_id']}: {(p.get('title') or '')[:60]}")
            if len(papers) > 20:
                print(f"  ... and {len(papers) - 20} more")
        return

    total_values = 0
    papers_processed = 0

    # Process GEO matrices first
    for src in ("geo_with_matrix", "geo_raw_only"):
        papers = papers_by_source.get(src, [])
        if limit:
            papers = papers[:limit]
        for i, paper in enumerate(papers):
            pmc_id = paper["pmc_id"]
            print(
                f"[GEO {i+1}/{len(papers)}] {pmc_id}...",
                end=" ", flush=True,
            )
            try:
                n = retrieve_geo_expression(db, paper)
                total_values += n
                papers_processed += 1
                print(f"{n} values")
                _append_jsonl(OUTPUT_JSONL, {
                    "pmc_id": pmc_id, "source": "geo", "values_extracted": n,
                })
            except Exception as e:
                logger.exception("Error retrieving GEO data for %s", pmc_id)
                print(f"error: {e}")

    # Process supplement files
    papers = papers_by_source.get("supplement", [])
    if limit:
        papers = papers[:limit]
    for i, paper in enumerate(papers):
        pmc_id = paper["pmc_id"]
        print(
            f"[Supp {i+1}/{len(papers)}] {pmc_id}...",
            end=" ", flush=True,
        )
        try:
            n = retrieve_supplement_expression(db, paper)
            total_values += n
            papers_processed += 1
            print(f"{n} values")
            _append_jsonl(OUTPUT_JSONL, {
                "pmc_id": pmc_id, "source": "supplement", "values_extracted": n,
            })
        except Exception as e:
            logger.exception("Error retrieving supplement data for %s", pmc_id)
            print(f"error: {e}")

    print(f"\nDone. Extracted {total_values} expression values from {papers_processed} papers.")

    # Backfill comparison from source_detail filenames
    comp_backfilled = _backfill_comparison_from_source_detail(db)
    if comp_backfilled:
        print(f"Backfilled comparison for {comp_backfilled} values from source_detail")

    # Backfill time_point_day from condition_label text patterns
    backfilled = _backfill_day_from_labels(db)
    if backfilled:
        print(f"Backfilled time_point_day for {backfilled} values from label text")

    # Backfill time_point_day from geo_sample_stage_mappings
    mapping_backfilled = _backfill_day_from_mappings(db)
    if mapping_backfilled:
        print(f"Backfilled time_point_day for {mapping_backfilled} values from stage mappings")

    # Report coverage
    stats = db._conn.execute(
        """SELECT COUNT(*) as total,
                  SUM(CASE WHEN time_point_day IS NOT NULL THEN 1 ELSE 0 END) as has_day
           FROM expression_values"""
    ).fetchone()
    if stats and stats[0] > 0:
        pct = 100.0 * stats[1] / stats[0]
        print(f"time_point_day coverage: {stats[1]}/{stats[0]} ({pct:.1f}%)")

    # Write markers summary
    if total_values > 0:
        _write_markers_json(db)
        print(f"Markers written to {MARKERS_OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Retrieve expression data from GEO and supplements",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process per source")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show eligible papers without running")
    parser.add_argument("--source", choices=["geo", "supplement", "all"],
                        default="all", help="Data source to retrieve from")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db = PipelineDB()
    try:
        run(db, limit=args.limit, dry_run=args.dry_run, source=args.source)
    finally:
        db.close()


if __name__ == "__main__":
    main()
