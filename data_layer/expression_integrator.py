"""Cross-study expression integration and normalization.

Builds a protocol x gene expression matrix from expression_values,
applies cross-study normalization, and tracks provenance.

Usage:
    from data_layer.expression_integrator import build_protocol_expression_matrix
    df = build_protocol_expression_matrix(db)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_layer.gene_panel import TARGET_GENES, ALL_TARGET_GENES, resolve_alias

logger = logging.getLogger(__name__)

# Flat list of all target genes (stable ordering for matrix columns)
TARGET_GENES_FLAT = sorted(ALL_TARGET_GENES)


def build_protocol_expression_matrix(db) -> pd.DataFrame:
    """Build a protocol x gene matrix from expression_values.

    Rows: one per protocol (protocol_id)
    Columns: target genes from gene_panel
    Values: best available expression value (highest confidence source)

    Returns a DataFrame with protocol metadata columns + gene columns.
    """
    rows = db.get_expression_matrix_data()
    if not rows:
        return pd.DataFrame()

    # Group values by (protocol_id or paper_id, gene)
    # Pick highest-confidence value per gene per protocol
    best_values: dict[tuple, dict] = {}  # (protocol_key, gene) -> record

    for r in rows:
        proto_id = r.get("protocol_id")
        paper_id = r["paper_id"]
        pmc_id = r.get("pmc_id", "")
        arm = r.get("protocol_arm", "")

        key = proto_id if proto_id else f"paper_{paper_id}"
        gene = resolve_alias(r["gene_symbol"])
        if gene not in ALL_TARGET_GENES:
            continue

        lookup = (key, gene)
        confidence = r.get("confidence") or 0.0
        existing = best_values.get(lookup)

        if not existing or confidence > (existing.get("confidence") or 0):
            best_values[lookup] = {
                "protocol_id": proto_id,
                "paper_id": paper_id,
                "pmc_id": pmc_id,
                "protocol_arm": arm,
                "gene_symbol": gene,
                "value": r.get("value"),
                "unit": r.get("unit"),
                "source_type": r.get("source_type"),
                "source_detail": r.get("source_detail"),
                "confidence": confidence,
            }

    if not best_values:
        return pd.DataFrame()

    # Collect unique protocol keys
    protocol_keys = sorted(set(k[0] for k in best_values))

    # Build matrix
    matrix_rows = []
    for pk in protocol_keys:
        row_data: dict = {
            "protocol_id": None,
            "paper_id": None,
            "pmc_id": "",
            "protocol_arm": "",
        }
        for gene in TARGET_GENES_FLAT:
            entry = best_values.get((pk, gene))
            if entry:
                row_data["protocol_id"] = entry["protocol_id"]
                row_data["paper_id"] = entry["paper_id"]
                row_data["pmc_id"] = entry["pmc_id"]
                row_data["protocol_arm"] = entry["protocol_arm"]
                row_data[gene] = entry["value"]
            else:
                row_data[gene] = np.nan
        matrix_rows.append(row_data)

    df = pd.DataFrame(matrix_rows)
    return df


def build_stage_expression_matrix(db) -> pd.DataFrame:
    """Build a stage-level expression matrix using GEO sample-stage mappings.

    Rows: (protocol_id, stage_name)
    Columns: target genes
    """
    rows = db.get_stage_expression_data()
    if not rows:
        return pd.DataFrame()

    # Group by (protocol_id, stage_name, gene)
    stage_values: dict[tuple, list[float]] = {}

    for r in rows:
        proto_id = r.get("protocol_id")
        stage = r.get("stage_name") or "unknown"
        gene = resolve_alias(r["gene_symbol"])
        if gene not in ALL_TARGET_GENES:
            continue

        value = r.get("value")
        if value is None:
            continue

        key = (proto_id, stage, gene)
        if key not in stage_values:
            stage_values[key] = []
        stage_values[key].append(value)

    if not stage_values:
        return pd.DataFrame()

    # Aggregate (mean of replicates)
    unique_stages = sorted(set((k[0], k[1]) for k in stage_values))

    matrix_rows = []
    for proto_id, stage in unique_stages:
        row_data: dict = {"protocol_id": proto_id, "stage_name": stage}
        for gene in TARGET_GENES_FLAT:
            vals = stage_values.get((proto_id, stage, gene))
            if vals:
                row_data[gene] = np.mean(vals)
            else:
                row_data[gene] = np.nan
        matrix_rows.append(row_data)

    return pd.DataFrame(matrix_rows)


def normalize_across_studies(
    df: pd.DataFrame, method: str = "rank"
) -> pd.DataFrame:
    """Normalize expression values across studies.

    Methods:
        rank: rank-based normalization within each paper, scaled to [0, 1]
        quantile: quantile normalization across all studies
        within_study_relative: divide by max within each paper
    """
    gene_cols = [c for c in df.columns if c in ALL_TARGET_GENES]
    if not gene_cols:
        return df

    result = df.copy()

    if method == "rank":
        if "pmc_id" in result.columns:
            for gene in gene_cols:
                result[f"{gene}_norm"] = result.groupby("pmc_id")[gene].rank(pct=True)
        else:
            for gene in gene_cols:
                result[f"{gene}_norm"] = result[gene].rank(pct=True)

    elif method == "within_study_relative":
        if "pmc_id" in result.columns:
            for gene in gene_cols:
                max_vals = result.groupby("pmc_id")[gene].transform("max")
                result[f"{gene}_norm"] = result[gene] / max_vals.replace(0, np.nan)
        else:
            for gene in gene_cols:
                max_val = result[gene].max()
                result[f"{gene}_norm"] = result[gene] / max_val if max_val else np.nan

    elif method == "quantile":
        # Quantile normalization
        gene_df = result[gene_cols].copy()
        ranked = gene_df.rank(method="average")
        mean_per_rank = gene_df.apply(lambda col: col.sort_values().values).mean(axis=1)

        for gene in gene_cols:
            ranks = ranked[gene]
            result[f"{gene}_norm"] = ranks.map(
                lambda r: mean_per_rank.iloc[int(r) - 1] if pd.notna(r) and int(r) > 0 else np.nan
            )

    return result


def build_provenance(db) -> dict:
    """Build provenance tracking for the expression matrix.

    Returns {"{protocol_id}_{gene}": {source_info}} for each cell.
    """
    rows = db.get_expression_matrix_data()
    provenance: dict[str, dict] = {}

    for r in rows:
        proto_id = r.get("protocol_id") or f"paper_{r['paper_id']}"
        gene = resolve_alias(r["gene_symbol"])
        if gene not in ALL_TARGET_GENES:
            continue

        key = f"{proto_id}_{gene}"

        # Keep highest-confidence source
        existing = provenance.get(key)
        confidence = r.get("confidence") or 0
        if existing and (existing.get("confidence") or 0) >= confidence:
            continue

        provenance[key] = {
            "value": r.get("value"),
            "original_unit": r.get("unit"),
            "source_type": r.get("source_type"),
            "source_detail": r.get("source_detail"),
            "confidence": confidence,
            "paper_pmc_id": r.get("pmc_id"),
            "condition_label": r.get("condition_label"),
        }

    return provenance
