"""Export protocol + expression data to Excel.

Creates a multi-sheet workbook combining protocol metadata (stages, growth
factors, medium) with per-day expression trajectories from GEO RNA-seq.

Usage:
    python3 export_results.py                       # default output
    python3 export_results.py -o my_export.xlsx     # custom path
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_layer.database import PipelineDB
from data_layer.gene_panel import TARGET_GENES, ALL_TARGET_GENES, resolve_alias

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/exports/protocol_expression_export.xlsx")


def _fmt_reagent(r) -> str:
    """Format a reagent dict as 'Name (conc unit)'."""
    if r is None:
        return ""
    if isinstance(r, str):
        return r
    if not isinstance(r, dict):
        return str(r)
    name = r.get("name", "")
    conc = r.get("concentration")
    unit = r.get("unit", "")
    if conc is not None:
        return f"{name} ({conc} {unit})".strip()
    return name


def _fmt_reagent_list(items) -> str:
    """Join a list of reagent dicts into a semicolon-separated string."""
    if not items:
        return ""
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except (json.JSONDecodeError, TypeError):
            return items
    return "; ".join(filter(None, (_fmt_reagent(r) for r in items)))


def _fmt_markers(items) -> str:
    if not items:
        return ""
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except (json.JSONDecodeError, TypeError):
            return items
    parts = []
    for m in items:
        if isinstance(m, str):
            parts.append(m)
        elif isinstance(m, dict):
            marker = m.get("marker", m.get("name", ""))
            val = m.get("value")
            parts.append(f"{marker}: {val}" if val else marker)
    return "; ".join(parts)


def _fmt_assays(items) -> str:
    """Format functional_assays list (may contain dicts or strings)."""
    if not items:
        return ""
    parts = []
    for a in items:
        if isinstance(a, str):
            parts.append(a)
        elif isinstance(a, dict):
            name = a.get("assay_name", a.get("name", ""))
            val = a.get("value")
            unit = a.get("unit", "")
            if val is not None:
                parts.append(f"{name}: {val} {unit}".strip())
            else:
                parts.append(name)
    return "; ".join(parts)


def build_protocols_sheet(db: PipelineDB) -> pd.DataFrame:
    """Sheet 1: One row per protocol with overview metadata."""
    rows = db._conn.execute(
        """SELECT pr.id AS protocol_id, p.pmc_id, p.doi, p.title,
                  pr.protocol_arm, pr.cell_source, pr.culture_system,
                  pr.stages, pr.endpoint_assessment,
                  pr.extraction_confidence, pr.is_optimized, pr.pass_number
           FROM protocols pr
           JOIN papers p ON p.id = pr.paper_id
           ORDER BY p.pmc_id, pr.id"""
    ).fetchall()

    records = []
    for r in rows:
        r = dict(r)
        cs = json.loads(r["cell_source"]) if r["cell_source"] else {}
        cs = cs if isinstance(cs, dict) else {}
        cu = json.loads(r["culture_system"]) if r["culture_system"] else {}
        cu = cu if isinstance(cu, dict) else {}
        stages = json.loads(r["stages"]) if r["stages"] else []
        stages = stages if isinstance(stages, list) else []
        ep = json.loads(r["endpoint_assessment"]) if r["endpoint_assessment"] else {}
        ep = ep if isinstance(ep, dict) else {}

        records.append({
            "Protocol ID": r["protocol_id"],
            "PMC ID": r["pmc_id"],
            "DOI": r.get("doi"),
            "Title": r["title"],
            "Arm": r["protocol_arm"],
            "Cell Type": cs.get("type"),
            "Cell Line": cs.get("line_name"),
            "Organism": cs.get("organism"),
            "Culture Format": cu.get("format"),
            "Substrate": cu.get("substrate"),
            "O2": cu.get("oxygen_condition"),
            "Num Stages": len(stages),
            "Confidence": r["extraction_confidence"],
            "Pass": r["pass_number"],
            "Optimized": "Yes" if r["is_optimized"] else "No",
            "Endpoint Markers": _fmt_markers(ep.get("markers")),
            "Functional Assays": _fmt_assays(ep.get("functional_assays")),
        })

    return pd.DataFrame(records)


def build_stages_sheet(db: PipelineDB) -> pd.DataFrame:
    """Sheet 2: One row per stage with growth factors, small molecules, medium."""
    rows = db._conn.execute(
        """SELECT pr.id AS protocol_id, p.pmc_id, pr.stages
           FROM protocols pr
           JOIN papers p ON p.id = pr.paper_id
           ORDER BY p.pmc_id, pr.id"""
    ).fetchall()

    records = []
    for r in rows:
        r = dict(r)
        stages = json.loads(r["stages"]) if r["stages"] else []
        stages = stages if isinstance(stages, list) else []
        for i, stage in enumerate(stages):
            records.append({
                "Protocol ID": r["protocol_id"],
                "PMC ID": r["pmc_id"],
                "Stage #": i + 1,
                "Stage Name": stage.get("stage_name", stage.get("name", "")),
                "Duration (days)": stage.get("duration_days", stage.get("duration")),
                "Base Medium": stage.get("base_medium", ""),
                "Growth Factors": _fmt_reagent_list(stage.get("growth_factors")),
                "Small Molecules": _fmt_reagent_list(stage.get("small_molecules")),
                "Supplements": _fmt_reagent_list(stage.get("supplements")),
                "Seeding Density": stage.get("seeding_density"),
                "Medium Change Freq": stage.get("medium_change_frequency"),
                "Stage Markers": _fmt_markers(stage.get("markers")),
            })

    return pd.DataFrame(records)


def build_expression_trajectories_sheet(db: PipelineDB) -> pd.DataFrame:
    """Sheet 3: Per-day expression values for protocols with temporal data.

    Columns: Protocol ID, PMC ID, Day, Gene1, Gene2, ...
    Only includes protocols that have >=2 distinct time points.
    """
    # Get protocols with multi-day data
    multi_day_protos = db._conn.execute(
        """SELECT protocol_id, COUNT(DISTINCT time_point_day) AS n_days
           FROM expression_values
           WHERE time_point_day IS NOT NULL
           GROUP BY protocol_id
           HAVING n_days >= 2"""
    ).fetchall()
    proto_ids = [r[0] for r in multi_day_protos]

    if not proto_ids:
        return pd.DataFrame()

    placeholders = ",".join("?" * len(proto_ids))
    rows = db._conn.execute(
        f"""SELECT ev.protocol_id, p.pmc_id, ev.time_point_day,
                   ev.gene_symbol, AVG(ev.value) AS avg_value
            FROM expression_values ev
            JOIN papers p ON p.id = ev.paper_id
            WHERE ev.protocol_id IN ({placeholders})
            AND ev.time_point_day IS NOT NULL
            GROUP BY ev.protocol_id, ev.time_point_day, ev.gene_symbol
            ORDER BY ev.protocol_id, ev.time_point_day, ev.gene_symbol""",
        proto_ids,
    ).fetchall()

    # Pivot: (protocol_id, day) → gene values
    gene_cols = sorted(ALL_TARGET_GENES)
    records: dict[tuple, dict] = {}
    for r in rows:
        r = dict(r)
        gene = resolve_alias(r["gene_symbol"])
        if gene not in ALL_TARGET_GENES:
            continue
        key = (r["protocol_id"], r["time_point_day"])
        if key not in records:
            records[key] = {
                "Protocol ID": r["protocol_id"],
                "PMC ID": r["pmc_id"],
                "Day": r["time_point_day"],
            }
        records[key][gene] = r["avg_value"]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(list(records.values()))
    # Reorder columns: metadata first, then genes
    meta_cols = ["Protocol ID", "PMC ID", "Day"]
    gene_present = [g for g in gene_cols if g in df.columns]
    df = df[meta_cols + gene_present]
    return df.sort_values(["Protocol ID", "Day"]).reset_index(drop=True)


def build_protocol_matrix_sheet(db: PipelineDB) -> pd.DataFrame:
    """Sheet 4: Protocol × gene matrix (best value per protocol per gene)."""
    rows = db.get_expression_matrix_data()
    if not rows:
        return pd.DataFrame()

    # Pick best value per (protocol, gene)
    best: dict[tuple, dict] = {}
    for r in rows:
        proto_id = r.get("protocol_id") or f"paper_{r['paper_id']}"
        gene = resolve_alias(r["gene_symbol"])
        if gene not in ALL_TARGET_GENES:
            continue
        key = (proto_id, gene)
        conf = r.get("confidence") or 0
        if key not in best or conf > (best[key].get("confidence") or 0):
            best[key] = r

    proto_keys = sorted(set(k[0] for k in best))
    gene_cols = sorted(ALL_TARGET_GENES)

    records = []
    for pk in proto_keys:
        row: dict = {"Protocol ID": pk, "PMC ID": "", "Arm": ""}
        for g in gene_cols:
            entry = best.get((pk, g))
            if entry:
                row["PMC ID"] = entry.get("pmc_id", "")
                row["Arm"] = entry.get("protocol_arm", "")
                row[g] = entry.get("value")
            else:
                row[g] = np.nan
        records.append(row)

    return pd.DataFrame(records)


def build_stage_expression_sheet(db: PipelineDB) -> pd.DataFrame:
    """Combined sheet: protocol stage info + expression values at each day.

    One row per (protocol, day). Includes the protocol's stage details
    (medium, growth factors, small molecules) alongside gene expression
    values at that time point, so you can see what reagents were used and
    what the expression looked like on the same row.
    """
    # Get all protocols with stages and expression data
    proto_rows = db._conn.execute(
        """SELECT pr.id AS protocol_id, p.pmc_id, p.doi, p.title,
                  pr.protocol_arm, pr.cell_source, pr.culture_system,
                  pr.stages, pr.extraction_confidence
           FROM protocols pr
           JOIN papers p ON p.id = pr.paper_id
           WHERE pr.id IN (SELECT DISTINCT protocol_id FROM expression_values
                           WHERE protocol_id IS NOT NULL)
           ORDER BY p.pmc_id, pr.id"""
    ).fetchall()

    if not proto_rows:
        return pd.DataFrame()

    # Build stage day-ranges for each protocol
    # stage_map[protocol_id] = [(start_day, end_day, stage_dict), ...]
    proto_info: dict[int, dict] = {}
    stage_map: dict[int, list[tuple[int, int, dict]]] = {}
    for r in proto_rows:
        r = dict(r)
        pid = r["protocol_id"]
        proto_info[pid] = r
        stages = json.loads(r["stages"]) if r["stages"] else []
        stages = stages if isinstance(stages, list) else []
        ranges = []
        cumulative_day = 0
        for stage in stages:
            dur = stage.get("duration_days") or stage.get("duration") or 0
            try:
                dur = int(dur)
            except (ValueError, TypeError):
                dur = 0
            start = cumulative_day
            end = cumulative_day + dur
            ranges.append((start, end, stage))
            cumulative_day = end
        stage_map[pid] = ranges

    def _find_stage(pid: int, day: int) -> dict | None:
        for start, end, stage in stage_map.get(pid, []):
            if start <= day <= end:
                return stage
        return None

    # Get expression data with time points
    expr_rows = db._conn.execute(
        """SELECT ev.protocol_id, p.pmc_id, ev.time_point_day,
                  ev.gene_symbol, AVG(ev.value) AS avg_value, ev.unit,
                  ev.condition_label
           FROM expression_values ev
           JOIN papers p ON p.id = ev.paper_id
           WHERE ev.protocol_id IS NOT NULL AND ev.time_point_day IS NOT NULL
           GROUP BY ev.protocol_id, ev.time_point_day, ev.gene_symbol
           ORDER BY ev.protocol_id, ev.time_point_day"""
    ).fetchall()

    # Pivot into (protocol_id, day) -> row
    gene_cols = sorted(ALL_TARGET_GENES)
    records: dict[tuple, dict] = {}
    for r in expr_rows:
        r = dict(r)
        pid = r["protocol_id"]
        day = r["time_point_day"]
        gene = resolve_alias(r["gene_symbol"])
        if gene not in ALL_TARGET_GENES:
            continue
        key = (pid, day)
        if key not in records:
            info = proto_info.get(pid, {})
            cs = json.loads(info.get("cell_source") or "{}") if info.get("cell_source") else {}
            if not isinstance(cs, dict):
                cs = {}
            cu = json.loads(info.get("culture_system") or "{}") if info.get("culture_system") else {}
            if not isinstance(cu, dict):
                cu = {}
            active_stage = _find_stage(pid, day)
            records[key] = {
                "Protocol ID": pid,
                "PMC ID": info.get("pmc_id", r["pmc_id"]),
                "DOI": info.get("doi", ""),
                "Arm": info.get("protocol_arm", ""),
                "Cell Type": cs.get("type", ""),
                "Cell Line": cs.get("line_name", ""),
                "Culture Format": cu.get("format", ""),
                "Day": day,
                "Condition": r.get("condition_label", ""),
                "Stage Name": (active_stage.get("stage_name", active_stage.get("name", "")) if active_stage else ""),
                "Base Medium": (active_stage.get("base_medium", "") if active_stage else ""),
                "Growth Factors": (_fmt_reagent_list(active_stage.get("growth_factors")) if active_stage else ""),
                "Small Molecules": (_fmt_reagent_list(active_stage.get("small_molecules")) if active_stage else ""),
                "Supplements": (_fmt_reagent_list(active_stage.get("supplements")) if active_stage else ""),
                "Unit": r.get("unit", ""),
            }
        records[key][gene] = r["avg_value"]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(list(records.values()))
    meta_cols = [
        "Protocol ID", "PMC ID", "DOI", "Arm", "Cell Type", "Cell Line",
        "Culture Format", "Day", "Condition", "Stage Name", "Base Medium",
        "Growth Factors", "Small Molecules", "Supplements", "Unit",
    ]
    gene_present = [g for g in gene_cols if g in df.columns]
    df = df[[c for c in meta_cols if c in df.columns] + gene_present]
    return df.sort_values(["Protocol ID", "Day"]).reset_index(drop=True)


def build_all_expression_sheet(db: PipelineDB) -> pd.DataFrame:
    """Raw expression values dump — one row per measurement."""
    rows = db._conn.execute(
        """SELECT p.pmc_id, p.doi, pr.protocol_arm,
                  ev.gene_symbol, ev.value, ev.unit,
                  ev.condition_label, ev.time_point_day, ev.comparison,
                  ev.padj, ev.source_type, ev.source_detail, ev.confidence
           FROM expression_values ev
           JOIN papers p ON p.id = ev.paper_id
           LEFT JOIN protocols pr ON pr.id = ev.protocol_id
           ORDER BY p.pmc_id, ev.time_point_day, ev.gene_symbol"""
    ).fetchall()

    return pd.DataFrame(
        [dict(r) for r in rows],
        columns=[
            "pmc_id", "doi", "protocol_arm", "gene_symbol", "value", "unit",
            "condition_label", "time_point_day", "comparison", "padj",
            "source_type", "source_detail", "confidence",
        ],
    )


def build_rnaseq_metadata_sheet(db: PipelineDB) -> pd.DataFrame:
    """RNA-seq Metadata sheet: one row per paper with RNA-seq info."""
    rows = db._conn.execute(
        """SELECT p.pmc_id, p.doi, p.title,
                  rm.has_rnaseq, rm.technology, rm.library_prep,
                  rm.read_type, rm.genome_build, rm.alignment_tool,
                  rm.quantification_tool, rm.normalization,
                  rm.deg_summary, rm.accessions, rm.data_availability
           FROM rnaseq_metadata rm
           JOIN papers p ON p.id = rm.paper_id
           ORDER BY p.pmc_id"""
    ).fetchall()

    records = []
    for r in rows:
        r = dict(r)
        deg = json.loads(r["deg_summary"]) if r["deg_summary"] else {}
        accessions = json.loads(r["accessions"]) if r["accessions"] else []
        data_avail = json.loads(r["data_availability"]) if r["data_availability"] else {}

        records.append({
            "PMC ID": r["pmc_id"],
            "DOI": r["doi"],
            "Title": r["title"],
            "Has RNA-seq": "Yes" if r["has_rnaseq"] else "No",
            "Technology": r.get("technology"),
            "Library Prep": r.get("library_prep"),
            "Read Type": r.get("read_type"),
            "Genome Build": r.get("genome_build"),
            "Alignment Tool": r.get("alignment_tool"),
            "Quantification Tool": r.get("quantification_tool"),
            "Normalization": r.get("normalization"),
            "DEG Up Count": sum(
                c.get("n_up") or 0 for c in deg.get("comparisons") or []
            ) or None,
            "DEG Down Count": sum(
                c.get("n_down") or 0 for c in deg.get("comparisons") or []
            ) or None,
            "Accessions": "; ".join(filter(None, (
                a.get("accession") if isinstance(a, dict) else str(a)
                for a in accessions
            ))) if accessions else "",
            "Data Availability": data_avail.get("repository", "") if data_avail else "",
        })

    return pd.DataFrame(records)


def build_repository_sheet(db: PipelineDB) -> pd.DataFrame:
    """Repository Cross-Ref sheet: one row per accession with repository details."""
    rows = db._conn.execute(
        """SELECT p.pmc_id, p.doi,
                  rm.accession, rm.repository, rm.project_title,
                  rm.organism, rm.data_type, rm.platform,
                  rm.sample_count, rm.has_processed_matrix,
                  rm.fetch_status
           FROM repository_metadata rm
           JOIN papers p ON p.id = rm.paper_id
           ORDER BY p.pmc_id, rm.accession"""
    ).fetchall()

    records = []
    for r in rows:
        r = dict(r)
        records.append({
            "PMC ID": r["pmc_id"],
            "DOI": r["doi"],
            "Accession": r["accession"],
            "Repository": r["repository"],
            "Project Title": r.get("project_title"),
            "Organism": r.get("organism"),
            "Data Type": r.get("data_type"),
            "Platform": r.get("platform"),
            "Sample Count": r.get("sample_count"),
            "Has Processed Matrix": "Yes" if r.get("has_processed_matrix") else "No",
            "Fetch Status": r.get("fetch_status"),
        })

    return pd.DataFrame(records)


def build_stage_matrix_sheet(db: PipelineDB) -> pd.DataFrame:
    """Stage-level expression matrix (from stage_expression_matrix.tsv)."""
    path = Path("data/integrated/stage_expression_matrix.tsv")
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return pd.DataFrame()


def export(output_path: Path, db: PipelineDB) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building Protocols Overview...")
    protocols_df = build_protocols_sheet(db)
    print(f"  {len(protocols_df)} protocols")

    print("Building Stages Detail...")
    stages_df = build_stages_sheet(db)
    print(f"  {len(stages_df)} stages")

    print("Building Stage + Expression (by day)...")
    stage_expr_df = build_stage_expression_sheet(db)
    print(f"  {len(stage_expr_df)} rows ({stage_expr_df['Protocol ID'].nunique() if len(stage_expr_df) else 0} protocols)")

    print("Building Expression Trajectories...")
    trajectories_df = build_expression_trajectories_sheet(db)
    print(f"  {len(trajectories_df)} rows ({trajectories_df['Protocol ID'].nunique() if len(trajectories_df) else 0} protocols with multi-day data)")

    print("Building Protocol Expression Matrix...")
    matrix_df = build_protocol_matrix_sheet(db)
    print(f"  {len(matrix_df)} protocols x {len([c for c in matrix_df.columns if c in ALL_TARGET_GENES]) if len(matrix_df) else 0} genes")

    print("Building All Expression Values...")
    all_expr_df = build_all_expression_sheet(db)
    print(f"  {len(all_expr_df):,} rows")

    print("Building Stage Expression Matrix...")
    stage_df = build_stage_matrix_sheet(db)
    print(f"  {len(stage_df)} stage rows")

    print("Building RNA-seq Metadata...")
    rnaseq_meta_df = build_rnaseq_metadata_sheet(db)
    print(f"  {len(rnaseq_meta_df)} papers")

    print("Building Repository Cross-Ref...")
    repo_df = build_repository_sheet(db)
    print(f"  {len(repo_df)} accessions")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        protocols_df.to_excel(writer, sheet_name="Protocols", index=False)
        stages_df.to_excel(writer, sheet_name="Stages Detail", index=False)
        if len(stage_expr_df):
            stage_expr_df.to_excel(writer, sheet_name="Stage + Expression by Day", index=False)
        if len(trajectories_df):
            trajectories_df.to_excel(writer, sheet_name="Expression Trajectories", index=False)
        matrix_df.to_excel(writer, sheet_name="Protocol x Gene Matrix", index=False)
        if len(all_expr_df):
            all_expr_df.to_excel(writer, sheet_name="All Expression Values", index=False)
        if len(stage_df):
            stage_df.to_excel(writer, sheet_name="Stage Expression Matrix", index=False)
        if len(rnaseq_meta_df):
            rnaseq_meta_df.to_excel(writer, sheet_name="RNA-seq Metadata", index=False)
        if len(repo_df):
            repo_df.to_excel(writer, sheet_name="Repository Cross-Ref", index=False)

    print(f"\nExported to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export protocols + expression data to Excel")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    db = PipelineDB()
    try:
        export(args.output, db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
