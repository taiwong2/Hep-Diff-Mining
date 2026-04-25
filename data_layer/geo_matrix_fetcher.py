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
GPL_CACHE_DIR = Path("data/expression_data/gpl_annotations")


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


RNASEQ_GPLS = {
    "GPL16791", "GPL24676", "GPL11154", "GPL18573", "GPL20301", "GPL17303",
    "GPL24247", "GPL15520", "GPL21290", "GPL20795", "GPL18460", "GPL21697",
    "GPL21273", "GPL29480", "GPL28352", "GPL23479", "GPL25947", "GPL24106",
    "GPL23227", "GPL23159", "GPL21103", "GPL19117", "GPL19057", "GPL34281",
    "GPL30173", "GPL13112",
}

COUNT_FILE_PATTERNS = re.compile(
    r"(count|tpm|fpkm|rpkm|expression|quant|processed|"
    r"feature[_\s]?count|read[_\s]?count|norm|deseq|htseq|rsem|salmon|kallisto|star)",
    re.IGNORECASE,
)

SKIP_FILE_PATTERNS = re.compile(
    r"(_RAW\.tar|\.fastq|\.bam|\.bed|peak|\.wig|bigwig|narrowpeak|broadpeak|"
    r"\.cel|\.idat|\.sdf|filelist|readme|md5)",
    re.IGNORECASE,
)


def fetch_geo_supplementary_counts(
    gse_id: str, target_genes: set[str], dest_dir: Path | None = None
) -> dict[str, dict[str, float | None]]:
    """Fetch processed count matrices from GEO supplementary files.

    For RNA-seq datasets where series_matrix lacks gene-level data,
    downloads supplementary files that look like processed count matrices
    and parses them for target genes.

    Returns merged {gene_symbol: {sample_label: value}} across all
    successfully parsed supplementary files.
    """
    dest_dir = dest_dir or DEFAULT_MATRIX_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    gse_num = gse_id.replace("GSE", "")
    prefix = f"GSE{gse_num[:-3]}nnn" if len(gse_num) > 3 else "GSE1nnn"
    ftp_dir = f"{GEO_MATRIX_BASE}/{prefix}/{gse_id}/suppl/"

    try:
        ftp = FTP(GEO_FTP_HOST)
        ftp.login()
        ftp.cwd(ftp_dir)
        all_files = ftp.nlst()
        ftp.quit()
    except Exception as e:
        logger.debug("%s: no supplementary FTP dir or error: %s", gse_id, e)
        return {}

    candidates = []
    for fn in all_files:
        if SKIP_FILE_PATTERNS.search(fn):
            continue
        fn_lower = fn.lower()
        is_tabular = any(fn_lower.endswith(ext) for ext in (
            ".txt.gz", ".csv.gz", ".tsv.gz", ".txt", ".csv", ".tsv",
            ".xlsx", ".xls",
        ))
        if not is_tabular:
            continue
        if COUNT_FILE_PATTERNS.search(fn):
            candidates.append(fn)

    if not candidates:
        # If no obvious count files, try any tabular file that isn't skipped
        for fn in all_files:
            if SKIP_FILE_PATTERNS.search(fn):
                continue
            fn_lower = fn.lower()
            is_tabular = any(fn_lower.endswith(ext) for ext in (
                ".txt.gz", ".csv.gz", ".tsv.gz",
            ))
            if is_tabular:
                candidates.append(fn)

    if not candidates:
        logger.debug("%s: no candidate supplementary count files found", gse_id)
        return {}

    # Limit to 5 files and skip very large ones (>100MB, likely scRNA-seq atlases)
    candidates = candidates[:5]
    logger.info("%s: trying %d supplementary count file(s): %s",
                gse_id, len(candidates), ", ".join(candidates))

    merged_gene_data: dict[str, dict[str, float | None]] = {}

    for fn in candidates:
        out_path = dest_dir / f"{gse_id}_{fn}"
        if not out_path.exists():
            try:
                ftp = FTP(GEO_FTP_HOST)
                ftp.login()
                ftp.cwd(ftp_dir)
                file_size = ftp.size(fn)
                if file_size and file_size > 100_000_000:
                    logger.info("%s: skipping %s (%.0f MB, likely scRNA-seq atlas)",
                                gse_id, fn, file_size / 1e6)
                    ftp.quit()
                    continue
                buf = BytesIO()
                ftp.retrbinary(f"RETR {fn}", buf.write)
                ftp.quit()
                buf.seek(0)
                out_path.write_bytes(buf.read())
                logger.info("%s: downloaded supplementary %s (%d bytes)",
                            gse_id, fn, out_path.stat().st_size)
            except Exception as e:
                logger.debug("%s: failed to download %s: %s", gse_id, fn, e)
                continue
        elif out_path.stat().st_size > 100_000_000:
            logger.info("%s: skipping %s (%.0f MB on disk)", gse_id, fn, out_path.stat().st_size / 1e6)
            continue

        gene_data = parse_count_table(out_path, target_genes)
        if gene_data:
            for gene, sample_values in gene_data.items():
                if gene not in merged_gene_data:
                    merged_gene_data[gene] = {}
                merged_gene_data[gene].update(sample_values)

    if merged_gene_data:
        # Cap at 500 samples per gene — more than that signals scRNA-seq
        total_samples = max(len(v) for v in merged_gene_data.values()) if merged_gene_data else 0
        if total_samples > 500:
            logger.info("%s: skipping — %d samples suggests scRNA-seq, not bulk",
                        gse_id, total_samples)
            return {}
        logger.info("%s: recovered %d target genes from supplementary files",
                     gse_id, len(merged_gene_data))

    return merged_gene_data


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


def _download_gpl_annotation(gpl_id: str) -> Path | None:
    """Download GPL platform annotation table from GEO.

    Returns path to the cached annotation file, or None on failure.
    """
    GPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GPL_CACHE_DIR / f"{gpl_id}_annotation.txt"
    if out_path.exists():
        return out_path

    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl_id}&targ=self&form=text&view=data"
    req = Request(url, headers={"User-Agent": "CellDiffMining/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            out_path.write_bytes(resp.read())
        logger.info("Downloaded %s annotation (%d bytes)", gpl_id, out_path.stat().st_size)
        return out_path
    except (URLError, HTTPError, TimeoutError) as e:
        logger.warning("Failed to download %s annotation: %s", gpl_id, e)
        return None


def _build_probe_to_gene(gpl_path: Path, target_genes: set[str]) -> dict[str, str]:
    """Parse GPL annotation to build probe_id → canonical gene symbol mapping.

    Scans the annotation table for columns named Gene Symbol, Symbol,
    GENE_SYMBOL, gene_assignment, ORF, etc. Only maps probes that resolve
    to a target gene.
    """
    from data_layer.gene_panel import resolve_alias

    target_upper = {g.upper() for g in target_genes}
    probe_map: dict[str, str] = {}

    gene_col_names = {
        "gene symbol", "gene_symbol", "symbol", "gene.symbol",
        "genesymbol", "orf", "gene_name", "gene name", "ilmn_gene",
    }
    assignment_col_names = {"gene_assignment", "gene assignment", "mrna_assignment"}

    try:
        with open(gpl_path, "r", encoding="utf-8", errors="replace") as f:
            header_indices: list[int] = []
            assignment_indices: list[int] = []
            id_col = 0
            in_table = False

            for line in f:
                if line.startswith("#") or line.startswith("!") or line.startswith("^"):
                    continue

                parts = line.strip().split("\t")

                if not in_table:
                    lower_parts = [p.strip('"').lower() for p in parts]
                    if "id" in lower_parts:
                        id_col = lower_parts.index("id")
                        for i, lp in enumerate(lower_parts):
                            if lp in gene_col_names:
                                header_indices.append(i)
                            elif lp in assignment_col_names:
                                assignment_indices.append(i)
                        in_table = True
                    continue

                if len(parts) <= id_col:
                    continue

                probe_id = parts[id_col].strip('"')

                for ci in header_indices:
                    if ci >= len(parts):
                        continue
                    raw = parts[ci].strip('"').strip()
                    if not raw or raw == "---":
                        continue
                    for sym in re.split(r'[;/|,\s]+', raw):
                        sym = sym.strip()
                        canonical = resolve_alias(sym)
                        if canonical in target_upper:
                            probe_map[probe_id] = canonical
                            break
                    if probe_id in probe_map:
                        break

                if probe_id not in probe_map:
                    for ci in assignment_indices:
                        if ci >= len(parts):
                            continue
                        raw = parts[ci].strip('"').strip()
                        if not raw or raw == "---":
                            continue
                        for sym in re.split(r'[;/|,\s]+', raw):
                            sym = sym.strip()
                            canonical = resolve_alias(sym)
                            if canonical in target_upper:
                                probe_map[probe_id] = canonical
                                break
                        if probe_id in probe_map:
                            break

    except Exception as e:
        logger.warning("Failed to parse GPL annotation %s: %s", gpl_path, e)

    if probe_map:
        logger.info("Built probe→gene map from %s: %d probes → %d unique genes",
                     gpl_path.name, len(probe_map),
                     len(set(probe_map.values())))
    return probe_map


def _extract_platform_id(matrix_path: Path) -> str | None:
    """Extract GPL platform ID from a series matrix metadata header."""
    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    try:
        with opener(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("!Series_platform_id"):
                    m = re.search(r"GPL\d+", line)
                    if m:
                        return m.group()
                if line.startswith('"ID_REF"'):
                    break
    except Exception:
        pass
    return None


def parse_series_matrix(matrix_path: Path, target_genes: set[str]) -> dict[str, dict[str, float | None]]:
    """Parse GEO series matrix file, extracting only target genes.

    Returns {gene_symbol: {gsm_id: value, ...}, ...} for target genes.
    First tries direct gene symbol matching (resolve_alias). If no hits,
    downloads the GPL platform annotation and maps probe IDs to genes.
    When multiple probes map to the same gene, keeps the one with highest
    mean expression.
    """
    from data_layer.gene_panel import resolve_alias

    target_upper = {g.upper() for g in target_genes}

    gsm_ids: list[str] = []
    gene_data: dict[str, dict[str, float | None]] = {}

    opener = gzip.open if str(matrix_path).endswith(".gz") else open

    # First pass: try direct gene symbol matching
    with opener(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
        in_table = False
        for line in f:
            if line.startswith("!Series_") or line.startswith("!series_"):
                continue

            if line.startswith('"ID_REF"'):
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

    if gene_data:
        logger.info(
            "Parsed %s: %d target genes found out of %d (samples: %d)",
            matrix_path.name, len(gene_data), len(target_upper), len(gsm_ids),
        )
        return gene_data

    # No direct hits — try probe-to-gene mapping via GPL annotation
    if not gsm_ids:
        return gene_data

    gpl_id = _extract_platform_id(matrix_path)
    if not gpl_id:
        logger.info("Parsed %s: 0 target genes, no GPL platform ID found", matrix_path.name)
        return gene_data

    gpl_path = _download_gpl_annotation(gpl_id)
    if not gpl_path:
        logger.info("Parsed %s: 0 target genes, GPL annotation download failed", matrix_path.name)
        return gene_data

    probe_map = _build_probe_to_gene(gpl_path, target_genes)
    if not probe_map:
        logger.info("Parsed %s: 0 target genes after GPL %s lookup (%d probes checked)",
                     matrix_path.name, gpl_id, 0)
        return gene_data

    # Second pass: use probe mapping
    # Collect all matching probe rows; pick best per gene later
    candidates: dict[str, list[dict[str, float | None]]] = {}

    with opener(matrix_path, "rt", encoding="utf-8", errors="replace") as f:
        in_table = False
        for line in f:
            if line.startswith('"ID_REF"'):
                in_table = True
                continue
            if not in_table:
                continue
            if line.startswith("!series_matrix_table_end") or not line.strip():
                break

            parts = line.strip().split("\t")
            if not parts:
                continue

            probe_id = parts[0].strip('"')
            canonical = probe_map.get(probe_id)
            if not canonical:
                continue

            values: dict[str, float | None] = {}
            for j, v in enumerate(parts[1:]):
                if j >= len(gsm_ids):
                    break
                v = v.strip().strip('"')
                try:
                    values[gsm_ids[j]] = float(v)
                except (ValueError, TypeError):
                    values[gsm_ids[j]] = None

            candidates.setdefault(canonical, []).append(values)

    # Pick the probe with highest mean expression per gene
    for gene, probe_rows in candidates.items():
        best_row = probe_rows[0]
        best_mean = -float("inf")
        for row in probe_rows:
            vals = [v for v in row.values() if v is not None]
            mean = sum(vals) / len(vals) if vals else -float("inf")
            if mean > best_mean:
                best_mean = mean
                best_row = row
        gene_data[gene] = best_row

    logger.info(
        "Parsed %s: %d target genes found via GPL %s probe mapping (samples: %d)",
        matrix_path.name, len(gene_data), gpl_id, len(gsm_ids),
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
