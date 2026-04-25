"""GEO series matrix download and parsing.

Downloads series matrix files from NCBI GEO FTP, parses them for target
gene expression values, and handles supplementary count matrix files.

Usage:
    from data_layer.geo_matrix_fetcher import download_series_matrix, parse_series_matrix
    path = download_series_matrix("GSE123456", Path("data/expression_data/geo_matrices"))
    gene_data = parse_series_matrix(path, target_genes)
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
from ftplib import FTP
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

GEO_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
GEO_MATRIX_BASE = "/geo/series"

DEFAULT_MATRIX_DIR = Path("data/expression_data/geo_matrices")


def download_series_matrix(gse_id: str, dest_dir: Path | None = None) -> Path | None:
    """Download GEO series matrix file via FTP.

    Returns the path to the downloaded (and decompressed) matrix, or None on failure.
    The matrix file is stored as {dest_dir}/{gse_id}_series_matrix.txt.
    """
    dest_dir = dest_dir or DEFAULT_MATRIX_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_path = dest_dir / f"{gse_id}_series_matrix.txt"
    if out_path.exists():
        logger.info("%s: matrix already downloaded", gse_id)
        return out_path

    # Construct FTP path: /geo/series/GSEnnn/GSExxxxxx/matrix/
    gse_num = gse_id.replace("GSE", "")
    prefix = f"GSE{gse_num[:-3]}nnn" if len(gse_num) > 3 else "GSE1nnn"
    ftp_dir = f"{GEO_MATRIX_BASE}/{prefix}/{gse_id}/matrix/"

    try:
        ftp = FTP(GEO_FTP_HOST)
        ftp.login()
        ftp.cwd(ftp_dir)

        # List files in matrix directory
        files = ftp.nlst()
        matrix_files = [f for f in files if "series_matrix" in f.lower()]

        if not matrix_files:
            logger.warning("%s: no series matrix files found at %s", gse_id, ftp_dir)
            ftp.quit()
            return None

        # Download the first (or only) matrix file
        remote_file = matrix_files[0]
        buf = BytesIO()
        ftp.retrbinary(f"RETR {remote_file}", buf.write)
        ftp.quit()

        buf.seek(0)

        # Decompress if gzipped
        if remote_file.endswith(".gz"):
            with gzip.open(buf, "rt", encoding="utf-8", errors="replace") as gz:
                out_path.write_text(gz.read())
        else:
            out_path.write_bytes(buf.read())

        logger.info("%s: downloaded series matrix (%d bytes)", gse_id, out_path.stat().st_size)
        return out_path

    except Exception as e:
        logger.warning("%s: FTP download failed: %s", gse_id, e)
        return None


def download_supplementary_file(url: str, dest_dir: Path, filename: str | None = None) -> Path | None:
    """Download a supplementary file from a URL.

    Returns the path to the downloaded file, or None on failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = url.rsplit("/", 1)[-1] if "/" in url else "unknown_file"

    out_path = dest_dir / filename
    if out_path.exists():
        return out_path

    # Convert FTP URLs to HTTP for simpler download
    dl_url = url
    if dl_url.startswith("ftp://"):
        dl_url = dl_url.replace("ftp://", "https://")

    req = Request(dl_url, headers={"User-Agent": "CellDiffMining/1.0"})
    try:
        with urlopen(req, timeout=120) as resp:
            out_path.write_bytes(resp.read())
        logger.info("Downloaded %s (%d bytes)", filename, out_path.stat().st_size)
        return out_path
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("Download failed for %s: %s", url, e)
        return None


def parse_series_matrix_metadata(matrix_path: Path) -> dict[str, str]:
    """Parse series matrix metadata to extract GSM ID → sample title mapping.

    Reads !Sample_geo_accession and !Sample_title rows from the metadata
    section before the data table. Returns {gsm_id: sample_title}.
    """
    gsm_ids: list[str] = []
    sample_titles: list[str] = []

    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    try:
        with opener(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("!Sample_geo_accession"):
                    gsm_ids = [col.strip().strip('"') for col in line.strip().split("\t")[1:]]
                elif line.startswith("!Sample_title"):
                    sample_titles = [col.strip().strip('"') for col in line.strip().split("\t")[1:]]
                elif line.startswith('"ID_REF"'):
                    break  # reached data table, stop
    except Exception as e:
        logger.warning("Failed to parse matrix metadata from %s: %s", matrix_path, e)
        return {}

    mapping: dict[str, str] = {}
    for i, gsm in enumerate(gsm_ids):
        if i < len(sample_titles) and gsm and sample_titles[i]:
            mapping[gsm] = sample_titles[i]

    return mapping


def parse_series_matrix(matrix_path: Path, target_genes: set[str]) -> dict[str, dict[str, float | None]]:
    """Parse GEO series matrix file, extracting only target genes.

    Returns {gene_symbol: {gsm_id: value, ...}, ...} for target genes.
    Only loads target gene rows into memory, skipping the rest.
    Resolves gene aliases and Ensembl IDs to canonical symbols.
    """
    from data_layer.gene_panel import resolve_alias

    target_upper = {g.upper() for g in target_genes}

    gsm_ids: list[str] = []
    gene_data: dict[str, dict[str, float | None]] = {}

    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    with opener(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
        in_table = False
        for line in f:
            # Skip metadata lines
            if line.startswith("!Series_") or line.startswith("!series_"):
                continue

            if line.startswith('"ID_REF"'):
                # Header row with sample IDs
                gsm_ids = [col.strip().strip('"') for col in line.strip().split("\t")[1:]]
                in_table = True
                continue

            if not in_table:
                continue

            if line.startswith("!series_matrix_table_end") or not line.strip():
                break

            parts = line.strip().split("\t")
            if not parts:
                continue

            raw_gene = parts[0].strip('"')
            canonical = resolve_alias(raw_gene)
            if canonical in target_upper:
                values: dict[str, float | None] = {}
                for j, v in enumerate(parts[1:]):
                    if j >= len(gsm_ids):
                        break
                    v = v.strip().strip('"')
                    try:
                        values[gsm_ids[j]] = float(v)
                    except (ValueError, TypeError):
                        values[gsm_ids[j]] = None
                gene_data[canonical] = values

    logger.info(
        "Parsed %s: %d target genes found out of %d (samples: %d)",
        matrix_path.name, len(gene_data), len(target_upper), len(gsm_ids),
    )
    return gene_data


def _find_gene_column(rows: list[list[str]], target_genes: set[str]) -> int:
    """Scan columns across initial rows to find which contains gene symbols.

    Returns the column index with the most target gene matches, or 0 as fallback.
    """
    from data_layer.gene_panel import resolve_alias

    target_upper = {g.upper() for g in target_genes}
    col_hits: dict[int, int] = {}
    for row in rows:
        for ci, val in enumerate(row):
            cleaned = val.strip().strip('"')
            canonical = resolve_alias(cleaned)
            if canonical in target_upper:
                col_hits[ci] = col_hits.get(ci, 0) + 1
    if col_hits:
        return max(col_hits, key=col_hits.get)
    return 0


def _parse_xlsx_table(file_path: Path, target_genes: set[str]) -> dict[str, dict[str, float | None]]:
    """Parse an Excel (.xlsx/.xls) count matrix for target genes.

    Scans all columns to find which contains gene symbols, then extracts
    expression values for target genes from the remaining columns.
    """
    import pandas as pd
    from data_layer.gene_panel import resolve_alias

    target_upper = {g.upper() for g in target_genes}
    gene_data: dict[str, dict[str, float | None]] = {}

    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        logger.warning("Failed to read Excel file %s: %s", file_path, e)
        return gene_data

    if df.empty:
        return gene_data

    # Find the gene column by sampling rows across the dataframe
    # Use up to 1000 evenly-spaced rows to handle large files where
    # target genes may not appear in the first 100 rows
    n_rows = len(df)
    if n_rows <= 1000:
        sample_df = df
    else:
        step = n_rows // 1000
        sample_df = df.iloc[::step].head(1000)

    best_col = None
    best_hits = 0
    for col in df.columns:
        hits = 0
        for val in sample_df[col].astype(str):
            canonical = resolve_alias(val.strip().strip('"'))
            if canonical in target_upper:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_col = col

    if best_col is None or best_hits == 0:
        logger.info("No gene column found in %s", file_path.name)
        return gene_data

    sample_cols = [c for c in df.columns if c != best_col]

    for _, row in df.iterrows():
        raw_gene = str(row[best_col]).strip().strip('"')
        canonical = resolve_alias(raw_gene)
        if canonical in target_upper:
            values: dict[str, float | None] = {}
            for sc in sample_cols:
                try:
                    values[str(sc)] = float(row[sc])
                except (ValueError, TypeError):
                    values[str(sc)] = None
            gene_data[canonical] = values

    logger.info(
        "Parsed XLSX %s: %d target genes found (gene col: %s, samples: %d)",
        file_path.name, len(gene_data), best_col, len(sample_cols),
    )
    return gene_data


def parse_count_table(file_path: Path, target_genes: set[str]) -> dict[str, dict[str, float | None]]:
    """Generic parser for count matrices from GEO supplementary files.

    Returns {gene_symbol: {sample_label: value, ...}, ...} for target genes.
    Handles .txt, .csv, .tsv, .xlsx, .xls files (optionally gzipped).
    Resolves gene aliases, Ensembl IDs, and Entrez IDs to canonical symbols.
    Automatically detects which column contains gene identifiers.
    """
    # Handle Excel files
    fp_str = str(file_path).lower()
    if fp_str.endswith(('.xlsx', '.xls')):
        return _parse_xlsx_table(file_path, target_genes)

    import csv
    from data_layer.gene_panel import resolve_alias

    target_upper = {g.upper() for g in target_genes}
    gene_data: dict[str, dict[str, float | None]] = {}

    opener = gzip.open if str(file_path).endswith(".gz") else open
    try:
        with opener(file_path, "rt", encoding="utf-8", errors="replace") as f:
            # Detect delimiter
            sample = f.read(4096)
            f.seek(0)
            if "\t" in sample:
                delimiter = "\t"
            elif ";" in sample and sample.count(";") > sample.count(","):
                delimiter = ";"
            elif "," in sample:
                delimiter = ","
            else:
                delimiter = "\t"

            reader = csv.reader(f, delimiter=delimiter)

            # Read header
            header = next(reader, None)
            if not header:
                return gene_data

            # Buffer all rows, then sample for gene column detection
            # (matches _parse_xlsx_table approach — first 100 rows may miss
            # target genes in large files like GSE70741 with 18K+ rows)
            all_rows: list[list[str]] = []
            for row in reader:
                if row:
                    all_rows.append(row)

            # Sample up to 1000 evenly-spaced rows for gene column detection
            n_rows = len(all_rows)
            if n_rows <= 1000:
                sample_rows = all_rows
            else:
                step = n_rows // 1000
                sample_rows = all_rows[::step][:1000]

            gene_col = _find_gene_column(sample_rows, target_genes)

            # Build sample IDs from header, excluding the gene column
            sample_ids = [h.strip().strip('"') for i, h in enumerate(header) if i != gene_col]

            def _process_row(row: list[str]) -> None:
                if not row or gene_col >= len(row):
                    return
                raw_gene = row[gene_col].strip().strip('"')
                canonical = resolve_alias(raw_gene)
                if canonical in target_upper:
                    values: dict[str, float | None] = {}
                    si = 0
                    for j, v in enumerate(row):
                        if j == gene_col:
                            continue
                        if si >= len(sample_ids):
                            break
                        v = v.strip().strip('"')
                        try:
                            values[sample_ids[si]] = float(v)
                        except (ValueError, TypeError):
                            values[sample_ids[si]] = None
                        si += 1
                    gene_data[canonical] = values

            for row in all_rows:
                _process_row(row)

    except Exception as e:
        logger.warning("Failed to parse count table %s: %s", file_path, e)

    if gene_data:
        logger.info(
            "Parsed %s: %d target genes found (gene col: %d, samples: %d)",
            file_path.name, len(gene_data), gene_col, len(sample_ids),
        )
    return gene_data
