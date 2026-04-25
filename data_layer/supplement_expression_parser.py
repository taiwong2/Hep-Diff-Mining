"""Parse supplement files (Excel/CSV) for gene expression data.

Classifies supplement files by content type (DEG list, count matrix, etc.)
and extracts target gene expression values.

Usage:
    from data_layer.supplement_expression_parser import classify_supplement_file, parse_deg_table
    ftype = classify_supplement_file(Path("table_s1.xlsx"))
    if ftype == "deg_list":
        degs = parse_deg_table(Path("table_s1.xlsx"), target_genes)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def classify_supplement_file(file_path: Path) -> str:
    """Classify a supplement Excel/CSV by its content.

    Returns one of: count_matrix, deg_list, pathway_table, sample_metadata, other.
    """
    import pandas as pd

    try:
        if file_path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path, nrows=5, engine="openpyxl")
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path, nrows=5)
        elif file_path.suffix == ".tsv" or file_path.name.endswith(".txt"):
            df = pd.read_csv(file_path, nrows=5, sep="\t")
        else:
            return "other"
    except Exception as e:
        logger.debug("Cannot read %s: %s", file_path.name, e)
        return "other"

    if df.empty or len(df.columns) < 2:
        return "other"

    cols_lower = [str(c).lower().strip() for c in df.columns]

    # DEG list: has fold-change and p-value columns
    has_fc = any("log2" in c or "fold" in c or "logfc" in c for c in cols_lower)
    has_pval = any(
        "padj" in c or "fdr" in c or "pvalue" in c or "p.value" in c or "p_value" in c
        for c in cols_lower
    )
    if has_fc and has_pval:
        return "deg_list"

    # Count/expression matrix: has expression unit columns
    expression_keywords = {"tpm", "fpkm", "rpkm", "cpm", "counts", "expression",
                           "raw_count", "normalized", "basemean"}
    if any(c in expression_keywords for c in cols_lower):
        return "count_matrix"

    # Check if columns look like sample IDs (many numeric columns)
    numeric_cols = sum(1 for c in cols_lower if not any(
        kw in c for kw in ("gene", "symbol", "name", "id", "description", "chr")
    ))
    if numeric_cols >= 3 and len(cols_lower) >= 5:
        # Check first data column for numeric values
        try:
            first_data = df.iloc[:, 1].astype(float)
            return "count_matrix"
        except (ValueError, TypeError):
            pass

    # Pathway table
    if any("pathway" in c or "go_term" in c or "kegg" in c or "enrichment" in c
           for c in cols_lower):
        return "pathway_table"

    # Sample metadata
    if any("sample" in c or "condition" in c or "treatment" in c for c in cols_lower):
        return "sample_metadata"

    return "other"


def parse_deg_table(
    file_path: Path, target_genes: set[str]
) -> list[dict]:
    """Extract DEG data for target genes from a DEG table.

    Returns list of {gene_symbol, log2fc, padj, basemean, comparison} dicts.
    """
    import pandas as pd

    target_upper = {g.upper() for g in target_genes}

    try:
        if file_path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        logger.warning("Cannot read DEG table %s: %s", file_path.name, e)
        return []

    if df.empty:
        return []

    # Identify gene column
    gene_col = _find_gene_column(df)
    if gene_col is None:
        return []

    # Identify log2FC column
    fc_col = _find_column(df, ["log2foldchange", "log2fc", "logfc", "log2_fold_change",
                                "foldchange", "fold_change"])

    # Identify padj column
    padj_col = _find_column(df, ["padj", "p_adj", "fdr", "q_value", "qvalue",
                                  "adjusted_pvalue", "adj.p.val"])

    # Identify basemean/expression column
    basemean_col = _find_column(df, ["basemean", "base_mean", "aveexpr", "meanexpr",
                                      "mean_expression"])

    from data_layer.gene_panel import resolve_alias

    results = []
    for _, row in df.iterrows():
        raw_gene = str(row[gene_col]).strip()
        canonical = resolve_alias(raw_gene)
        if canonical not in target_upper:
            continue

        entry: dict = {"gene_symbol": canonical}

        if fc_col is not None:
            try:
                entry["log2fc"] = float(row[fc_col])
            except (ValueError, TypeError):
                entry["log2fc"] = None

        if padj_col is not None:
            try:
                entry["padj"] = float(row[padj_col])
            except (ValueError, TypeError):
                entry["padj"] = None

        if basemean_col is not None:
            try:
                entry["basemean"] = float(row[basemean_col])
            except (ValueError, TypeError):
                entry["basemean"] = None

        results.append(entry)

    logger.info("Parsed DEG table %s: found %d/%d target genes",
                file_path.name, len(results), len(target_upper))
    return results


def parse_expression_table(
    file_path: Path, target_genes: set[str]
) -> dict[str, dict[str, float | None]]:
    """Extract gene x condition values from a count/expression matrix.

    Returns {gene_symbol: {condition_label: value, ...}, ...} for target genes.
    """
    import pandas as pd

    target_upper = {g.upper() for g in target_genes}

    try:
        if file_path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        logger.warning("Cannot read expression table %s: %s", file_path.name, e)
        return {}

    if df.empty:
        return {}

    # Identify gene column
    gene_col = _find_gene_column(df)
    if gene_col is None:
        # Try using index
        if df.index.dtype == object:
            df = df.reset_index()
            gene_col = _find_gene_column(df)
        if gene_col is None:
            return {}

    from data_layer.gene_panel import resolve_alias

    # Sample/condition columns are everything except the gene column
    # and other metadata columns
    meta_patterns = re.compile(
        r'gene|symbol|name|id|description|chr|start|end|strand|length|biotype',
        re.IGNORECASE,
    )
    sample_cols = [c for c in df.columns if c != gene_col and not meta_patterns.search(str(c))]

    gene_data: dict[str, dict[str, float | None]] = {}
    for _, row in df.iterrows():
        raw_gene = str(row[gene_col]).strip()
        canonical = resolve_alias(raw_gene)
        if canonical not in target_upper:
            continue

        values: dict[str, float | None] = {}
        for col in sample_cols:
            try:
                values[str(col)] = float(row[col])
            except (ValueError, TypeError):
                values[str(col)] = None
        gene_data[canonical] = values

    logger.info("Parsed expression table %s: found %d/%d target genes, %d conditions",
                file_path.name, len(gene_data), len(target_upper), len(sample_cols))
    return gene_data


def find_expression_files(supp_dir: Path) -> list[tuple[Path, str]]:
    """Find supplement files that might contain expression data.

    Returns list of (file_path, classification) tuples.
    """
    results: list[tuple[Path, str]] = []

    if not supp_dir.exists():
        return results

    for file_path in sorted(supp_dir.iterdir()):
        if file_path.suffix not in (".xlsx", ".xls", ".csv", ".tsv", ".txt"):
            continue
        if file_path.name.startswith("."):
            continue
        # Skip very large files (> 50MB) that are likely full datasets
        if file_path.stat().st_size > 50 * 1024 * 1024:
            continue

        classification = classify_supplement_file(file_path)
        if classification in ("deg_list", "count_matrix"):
            results.append((file_path, classification))

    return results


def _find_gene_column(df) -> str | None:
    """Find the column most likely to contain gene symbols or Ensembl IDs."""
    gene_patterns = ["gene_symbol", "gene_name", "symbol", "gene",
                     "genename", "gene.symbol", "external_gene_name",
                     "gene_id", "ensembl_gene_id", "ensembl_id"]

    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    for pat in gene_patterns:
        if pat in cols_lower:
            return cols_lower[pat]

    # Check first column
    first_col = df.columns[0]
    first_vals = df[first_col].dropna().astype(str).head(10)
    # Gene symbols are typically uppercase 2-10 char strings
    gene_like = sum(1 for v in first_vals if re.match(r'^[A-Z][A-Z0-9]{1,15}$', v.strip()))
    if gene_like >= 5:
        return first_col

    # Check for Ensembl ID patterns (ENSG...) in first column
    ensg_like = sum(1 for v in first_vals if v.strip().startswith("ENSG"))
    if ensg_like >= 5:
        return first_col

    return None


def _find_column(df, patterns: list[str]) -> str | None:
    """Find a column matching any of the given patterns (case-insensitive)."""
    cols_lower = {str(c).lower().strip().replace(" ", "_").replace(".", "_"): c
                  for c in df.columns}
    for pat in patterns:
        if pat in cols_lower:
            return cols_lower[pat]
    # Partial match
    for pat in patterns:
        for col_lower, col_orig in cols_lower.items():
            if pat in col_lower:
                return col_orig
    return None
