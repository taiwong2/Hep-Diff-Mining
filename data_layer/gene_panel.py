"""Target gene panel for hepatocyte differentiation expression analysis.

Defines the genes of interest across differentiation stages, plus common
aliases for matching gene names from different sources (papers, GEO, etc.).
"""

from __future__ import annotations

import re

TARGET_GENES: dict[str, list[str]] = {
    "pluripotency": ["POU5F1", "NANOG", "SOX2", "LIN28A"],
    "definitive_endoderm": [
        "SOX17", "FOXA2", "CXCR4", "GATA4", "EOMES", "GSC", "MIXL1",
    ],
    "hepatic_progenitor": [
        "HNF4A", "AFP", "TBX3", "PROX1", "HNF1B", "ONECUT1", "ONECUT2", "HHEX",
    ],
    "mature_hepatocyte": [
        "ALB", "CYP3A4", "CYP1A2", "CYP2C9", "CYP2D6", "CYP2E1",
        "SERPINA1", "TTR", "TDO2", "TAT", "ASGR1", "SLC10A1",
        "SLCO1B1", "UGT1A1", "ADH1A", "ALDOB",
    ],
    "fetal_hepatocyte": ["AFP", "CYP3A7", "H19", "DLK1", "GPC3"],
    "cholangiocyte": ["KRT19", "KRT7", "SOX9", "EPCAM"],
    "mesenchymal": ["VIM", "ACTA2", "COL1A1", "S100A4"],
}

# Common aliases → canonical gene symbol
GENE_ALIASES: dict[str, str] = {
    "OCT4": "POU5F1",
    "OCT-4": "POU5F1",
    "OCT3/4": "POU5F1",
    "AAT": "SERPINA1",
    "NTCP": "SLC10A1",
    "OATP1B1": "SLCO1B1",
    "ALBUMIN": "ALB",
    "ALPHA-FETOPROTEIN": "AFP",
    "CK19": "KRT19",
    "CK7": "KRT7",
    "EPCAM": "EPCAM",
    "SMA": "ACTA2",
    "ALPHA-SMA": "ACTA2",
}

# Ensembl gene ID → canonical gene symbol
ENSEMBL_MAP: dict[str, str] = {
    "ENSG00000204531": "POU5F1",
    "ENSG00000111704": "NANOG",
    "ENSG00000181449": "SOX2",
    "ENSG00000131914": "LIN28A",
    "ENSG00000164736": "SOX17",
    "ENSG00000125798": "FOXA2",
    "ENSG00000163508": "CXCR4",
    "ENSG00000136574": "GATA4",
    "ENSG00000135925": "EOMES",
    "ENSG00000173542": "GSC",
    "ENSG00000185155": "MIXL1",
    "ENSG00000101076": "HNF4A",
    "ENSG00000081051": "AFP",
    "ENSG00000135111": "TBX3",
    "ENSG00000117707": "PROX1",
    "ENSG00000108753": "HNF1B",
    "ENSG00000169856": "ONECUT1",
    "ENSG00000119547": "ONECUT2",
    "ENSG00000152049": "HHEX",
    "ENSG00000163898": "ALB",
    "ENSG00000160868": "CYP3A4",
    "ENSG00000140505": "CYP1A2",
    "ENSG00000138109": "CYP2C9",
    "ENSG00000186377": "CYP2D6",
    "ENSG00000130649": "CYP2E1",
    "ENSG00000197249": "SERPINA1",
    "ENSG00000118271": "TTR",
    "ENSG00000151790": "TDO2",
    "ENSG00000198650": "TAT",
    "ENSG00000141505": "ASGR1",
    "ENSG00000100652": "SLC10A1",
    "ENSG00000134538": "SLCO1B1",
    "ENSG00000241635": "UGT1A1",
    "ENSG00000187758": "ADH1A",
    "ENSG00000136872": "ALDOB",
    "ENSG00000067064": "CYP3A7",
    "ENSG00000130600": "H19",
    "ENSG00000185559": "DLK1",
    "ENSG00000082781": "GPC3",
    "ENSG00000171345": "KRT19",
    "ENSG00000135480": "KRT7",
    "ENSG00000125398": "SOX9",
    "ENSG00000119888": "EPCAM",
    "ENSG00000026025": "VIM",
    "ENSG00000107796": "ACTA2",
    "ENSG00000108821": "COL1A1",
    "ENSG00000196154": "S100A4",
}

# NCBI Entrez Gene ID → canonical gene symbol
ENTREZ_MAP: dict[str, str] = {
    # pluripotency
    "5460": "POU5F1", "79923": "NANOG", "6657": "SOX2", "79727": "LIN28A",
    # definitive_endoderm
    "64321": "SOX17", "3170": "FOXA2", "7852": "CXCR4", "2626": "GATA4",
    "8320": "EOMES", "145258": "GSC", "83881": "MIXL1",
    # hepatic_progenitor
    "3172": "HNF4A", "174": "AFP", "6926": "TBX3", "5629": "PROX1",
    "6928": "HNF1B", "3175": "ONECUT1", "9480": "ONECUT2", "3087": "HHEX",
    # mature_hepatocyte
    "213": "ALB", "1576": "CYP3A4", "1544": "CYP1A2", "1559": "CYP2C9",
    "1565": "CYP2D6", "1571": "CYP2E1", "5265": "SERPINA1", "7276": "TTR",
    "6999": "TDO2", "6898": "TAT", "432": "ASGR1", "6554": "SLC10A1",
    "10599": "SLCO1B1", "54658": "UGT1A1", "124": "ADH1A", "229": "ALDOB",
    # fetal_hepatocyte
    "1551": "CYP3A7", "283120": "H19", "8788": "DLK1", "2719": "GPC3",
    # cholangiocyte
    "3880": "KRT19", "3855": "KRT7", "6662": "SOX9", "4072": "EPCAM",
    # mesenchymal
    "7431": "VIM", "59": "ACTA2", "1277": "COL1A1", "6275": "S100A4",
}

# Flat set of all target gene symbols (canonical names)
ALL_TARGET_GENES: set[str] = set()
for _genes in TARGET_GENES.values():
    ALL_TARGET_GENES.update(_genes)

# Combined lookup: includes aliases as keys for matching
ALL_GENE_NAMES: set[str] = ALL_TARGET_GENES | set(GENE_ALIASES.keys())


def resolve_alias(name: str) -> str:
    """Return canonical gene symbol, resolving aliases, Ensembl IDs, and Entrez IDs."""
    upper = name.upper().strip()
    # Strip Ensembl version suffix (ENSG00000163898.14 → ENSG00000163898)
    if upper.startswith("ENSG") and "." in upper:
        upper = upper.split(".")[0]
    if upper in ENSEMBL_MAP:
        return ENSEMBL_MAP[upper]
    # Check Entrez Gene IDs (numeric-only strings)
    if upper.isdigit() and upper in ENTREZ_MAP:
        return ENTREZ_MAP[upper]
    # Strip _chrN suffix (e.g. ALB_chr4 → ALB)
    if "_CHR" in upper:
        stripped = re.sub(r'_CHR\w*$', '', upper)
        if stripped in GENE_ALIASES:
            return GENE_ALIASES[stripped]
        if stripped != upper:
            upper = stripped
    return GENE_ALIASES.get(upper, upper)


def is_target_gene(name: str) -> bool:
    """Check if a gene name (or alias) is in the target panel."""
    return resolve_alias(name) in ALL_TARGET_GENES
