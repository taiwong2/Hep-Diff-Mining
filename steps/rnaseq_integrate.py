"""Phase 4: Expression integration and matrix building.

Builds a standardized protocol x gene expression matrix linking expression
data to protocol records, with cross-study normalization.

Usage:
    python3 rnaseq_integrate.py                         # default rank normalization
    python3 rnaseq_integrate.py --normalize quantile    # quantile normalization
    python3 rnaseq_integrate.py --normalize relative    # within-study relative
    python3 rnaseq_integrate.py --min-genes 5           # require >= 5 genes per protocol
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data_layer.database import PipelineDB
from data_layer.expression_integrator import (
    build_protocol_expression_matrix,
    build_stage_expression_matrix,
    normalize_across_studies,
    build_provenance,
    TARGET_GENES_FLAT,
)
from data_layer.gene_panel import TARGET_GENES, ALL_TARGET_GENES

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/integrated")
MATRIX_PATH = OUTPUT_DIR / "protocol_expression_matrix.tsv"
STAGE_MATRIX_PATH = OUTPUT_DIR / "stage_expression_matrix.tsv"
PROVENANCE_PATH = OUTPUT_DIR / "expression_sources.json"


def run(
    db: PipelineDB,
    normalize: str = "rank",
    min_genes: int = 3,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build protocol-level matrix
    print("Building protocol expression matrix...")
    df = build_protocol_expression_matrix(db)

    if df.empty:
        print("No expression data available. Run Phase 3 first.")
        return

    gene_cols = [c for c in df.columns if c in ALL_TARGET_GENES]

    # Filter protocols with minimum gene coverage
    gene_coverage = df[gene_cols].notna().sum(axis=1)
    df["gene_count"] = gene_coverage
    before = len(df)
    df = df[df["gene_count"] >= min_genes].copy()
    print(f"  Protocols before filter: {before}")
    print(f"  Protocols after min_genes={min_genes} filter: {len(df)}")

    if df.empty:
        print(f"No protocols have >= {min_genes} target genes. Try --min-genes 1")
        return

    # Normalize
    norm_method = {
        "rank": "rank",
        "quantile": "quantile",
        "relative": "within_study_relative",
    }.get(normalize, "rank")

    print(f"  Applying {norm_method} normalization...")
    df = normalize_across_studies(df, method=norm_method)

    # Write matrix
    df.to_csv(MATRIX_PATH, sep="\t", index=False)
    print(f"  Protocol matrix written to {MATRIX_PATH}")

    # Build stage-level matrix
    print("\nBuilding stage expression matrix...")
    stage_df = build_stage_expression_matrix(db)
    if not stage_df.empty:
        stage_df = normalize_across_studies(stage_df, method=norm_method)
        stage_df.to_csv(STAGE_MATRIX_PATH, sep="\t", index=False)
        print(f"  Stage matrix written to {STAGE_MATRIX_PATH}")
    else:
        print("  No stage-level expression data available (requires GEO sample mappings)")

    # Build provenance
    print("\nBuilding provenance tracking...")
    provenance = build_provenance(db)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    print(f"  Provenance written to {PROVENANCE_PATH} ({len(provenance)} entries)")

    # Print summary
    print("\n" + "=" * 60)
    print("Expression Matrix Summary")
    print("=" * 60)
    print(f"  Protocols:       {len(df)}")
    print(f"  Papers:          {df['pmc_id'].nunique()}")
    print(f"  Target genes:    {len(gene_cols)}")

    print(f"\n  Gene coverage (protocols with value):")
    coverage = df[gene_cols].notna().sum().sort_values(ascending=False)
    for gene in coverage.head(15).index:
        n = int(coverage[gene])
        pct = 100 * n / len(df)
        print(f"    {gene:12s}: {n:4d} ({pct:5.1f}%)")
    if len(coverage) > 15:
        print(f"    ... and {len(coverage) - 15} more genes")

    print(f"\n  By gene category:")
    for cat, genes in TARGET_GENES.items():
        cat_genes = [g for g in genes if g in gene_cols]
        if not cat_genes:
            continue
        mean_cov = df[cat_genes].notna().any(axis=1).sum()
        print(f"    {cat:25s}: {mean_cov:4d} protocols with any gene")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Build integrated protocol x gene expression matrix",
    )
    parser.add_argument("--normalize", choices=["rank", "quantile", "relative"],
                        default="rank", help="Normalization method (default: rank)")
    parser.add_argument("--min-genes", type=int, default=3,
                        help="Minimum target genes required per protocol (default: 3)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db = PipelineDB()
    try:
        run(db, normalize=args.normalize, min_genes=args.min_genes)
    finally:
        db.close()


if __name__ == "__main__":
    main()
