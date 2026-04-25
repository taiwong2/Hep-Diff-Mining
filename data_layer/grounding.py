"""Post-extraction grounding: verify growth factors, small molecules, and markers
against source text to eliminate hallucinated terms.

Two-layer approach:
1. MeSH-enriched alias tables map canonical names to synonym sets
2. is_term_grounded() checks each extracted term against paper + supplement text

Usage:
    # Build MeSH alias cache (run once)
    python3 -m data_layer.grounding --build-aliases

    # Test grounding on a single protocol
    python3 -m data_layer.grounding --test PMC10114490
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MESH_CACHE_PATH = Path(__file__).parent / "mesh_aliases.json"

# ------------------------------------------------------------------
# MeSH UID mappings (canonical_name -> MeSH descriptor UID)
# ------------------------------------------------------------------

GROWTH_FACTOR_MESH: dict[str, str] = {
    "HGF": "D017228",
    "BMP4": "D055419",
    "BMP2": "D055396",
    "Activin A": "D028341",
    "EGF": "D004815",
    "FGF2": "D016222",
    "FGF4": "D051526",
    "FGF10": "D051529",
    "OSM": "D053683",
    "TGFα": "D016212",
    "VEGF": "D042461",
    "KGF": "D051526",  # FGF7 descriptor
    "FGF7": "D051526",
}

SMALL_MOLECULE_MESH: dict[str, str] = {
    "dexamethasone": "D003907",
    "DMSO": "D004121",
    "valproic acid": "D014635",
    "nicotinamide": "D009536",
    "retinoic acid": "D014212",
    "hydrocortisone": "D006854",
    "insulin": "D007328",
    "ascorbic acid": "D001205",
    "sodium butyrate": "D020148",
}

MARKER_MESH: dict[str, str] = {
    "ALB": "D000418",
    "AFP": "D000509",
}

# ------------------------------------------------------------------
# Manual alias overrides (lab abbreviations, gene symbols, etc.)
# MeSH doesn't cover these well.
# ------------------------------------------------------------------

MANUAL_GF_ALIASES: dict[str, set[str]] = {
    "HGF": {"hgf", "hepatocyte growth factor", "scatter factor", "hepatopoietin"},
    "BMP4": {"bmp4", "bmp-4", "bone morphogenetic protein 4"},
    "BMP2": {"bmp2", "bmp-2", "bone morphogenetic protein 2"},
    "Activin A": {"activin a", "activin-a", "acta", "act-a", "inhba"},
    "EGF": {"egf", "epidermal growth factor"},
    "FGF2": {"fgf2", "fgf-2", "bfgf", "basic fgf", "basic fibroblast growth factor",
             "fibroblast growth factor 2"},
    "FGF4": {"fgf4", "fgf-4", "fibroblast growth factor 4"},
    "FGF10": {"fgf10", "fgf-10", "fibroblast growth factor 10"},
    "OSM": {"osm", "oncostatin m", "oncostatin-m"},
    "Wnt3a": {"wnt3a", "wnt-3a", "wnt 3a"},
    "KGF": {"kgf", "keratinocyte growth factor"},
    "FGF7": {"fgf7", "fgf-7"},
    "VEGF": {"vegf", "vascular endothelial growth factor", "vegf-a", "vegfa"},
    "TGFα": {"tgfα", "tgf-α", "tgf-alpha", "tgfalpha", "tgfa",
             "transforming growth factor alpha"},
    "TGFβ": {"tgfβ", "tgf-β", "tgf-beta", "tgfbeta", "tgfb", "tgfb1", "tgf-β1",
             "transforming growth factor beta"},
    "FGF1": {"fgf1", "fgf-1", "acidic fgf", "afgf"},
    "PDGF": {"pdgf", "platelet-derived growth factor"},
    "SCF": {"scf", "stem cell factor", "kit ligand", "kitl"},
    "IGF-1": {"igf-1", "igf1", "insulin-like growth factor 1"},
    "DKK1": {"dkk1", "dkk-1", "dickkopf-1", "dickkopf 1"},
    "Noggin": {"noggin"},
    "R-Spondin": {"r-spondin", "rspondin", "r-spondin 1", "rspo1"},
    "Jagged-1": {"jagged-1", "jagged1", "jag1"},
}

MANUAL_SM_ALIASES: dict[str, set[str]] = {
    "CHIR99021": {"chir99021", "chir-99021", "chir 99021", "chir"},
    "SB431542": {"sb431542", "sb-431542", "sb 431542"},
    "A83-01": {"a83-01", "a8301", "a 83-01"},
    "LDN193189": {"ldn193189", "ldn-193189", "ldn 193189", "ldn"},
    "dexamethasone": {"dexamethasone", "dex"},
    "DMSO": {"dmso", "dimethyl sulfoxide", "dimethylsulfoxide"},
    "compound E": {"compound e", "cpd e"},
    "Y-27632": {"y-27632", "y27632", "rock inhibitor y-27632"},
    "SB203580": {"sb203580", "sb-203580"},
    "dorsomorphin": {"dorsomorphin", "compound c", "dors"},
    "valproic acid": {"valproic acid", "vpa", "sodium valproate"},
    "nicotinamide": {"nicotinamide", "nam", "niacinamide"},
    "retinoic acid": {"retinoic acid", "all-trans retinoic acid", "atra"},
    "hydrocortisone": {"hydrocortisone", "hc"},
    "insulin": {"insulin"},
    "ascorbic acid": {"ascorbic acid", "vitamin c", "l-ascorbic acid"},
    "sodium butyrate": {"sodium butyrate", "nab", "butyrate"},
    "IWP2": {"iwp2", "iwp-2"},
    "IWR1": {"iwr1", "iwr-1", "iwr1-endo"},
    "PD0325901": {"pd0325901", "pd-0325901", "pd"},
    "SU5402": {"su5402", "su-5402"},
    "forskolin": {"forskolin", "fsk"},
    "dihexa": {"dihexa"},
    "5-aza-2′-deoxycytidine": {"5-aza-2'-deoxycytidine", "5-aza", "5aza", "decitabine",
                                "5-azacytidine", "azacytidine"},
    "trichostatin A": {"trichostatin a", "tsa"},
    "XAV939": {"xav939", "xav-939"},
    "ITS": {"its", "insulin-transferrin-selenium"},
    "rapamycin": {"rapamycin", "sirolimus"},
    "SB216763": {"sb216763", "sb-216763"},
}

MANUAL_MARKER_ALIASES: dict[str, set[str]] = {
    "ALB": {"alb", "albumin", "human albumin", "serum albumin"},
    "AFP": {"afp", "alpha-fetoprotein", "α-fetoprotein", "alpha fetoprotein", "alphafetoprotein"},
    "HNF4A": {"hnf4a", "hnf4α", "hnf4alpha", "hnf-4a", "hnf-4α",
              "hepatocyte nuclear factor 4 alpha", "hepatocyte nuclear factor 4a"},
    "SOX17": {"sox17", "sox-17"},
    "FOXA2": {"foxa2", "foxa-2", "hnf3β", "hnf3beta", "hnf-3β", "hnf3b"},
    "CXCR4": {"cxcr4", "cxcr-4", "cd184"},
    "CYP3A4": {"cyp3a4", "cyp-3a4", "cytochrome p450 3a4"},
    "CYP1A2": {"cyp1a2", "cyp-1a2", "cytochrome p450 1a2"},
    "CYP2C9": {"cyp2c9", "cyp-2c9"},
    "CYP2B6": {"cyp2b6", "cyp-2b6"},
    "CYP2D6": {"cyp2d6", "cyp-2d6"},
    "CYP2C19": {"cyp2c19"},
    "OCT4": {"oct4", "oct-4", "oct3/4", "pou5f1"},
    "NANOG": {"nanog"},
    "AAT": {"aat", "serpina1", "alpha-1-antitrypsin", "a1at", "α1-antitrypsin",
            "alpha-1 antitrypsin", "α1at", "a1-antitrypsin"},
    "TTR": {"ttr", "transthyretin", "prealbumin"},
    "ASGR1": {"asgr1", "asgpr1", "asialoglycoprotein receptor"},
    "TBX3": {"tbx3"},
    "PROX1": {"prox1", "prox-1"},
    "EpCAM": {"epcam", "ep-cam", "cd326", "epithelial cell adhesion molecule"},
    "CK18": {"ck18", "krt18", "cytokeratin 18", "keratin 18"},
    "CK19": {"ck19", "krt19", "cytokeratin 19", "keratin 19"},
    "CK7": {"ck7", "krt7", "cytokeratin 7", "keratin 7"},
    "GATA4": {"gata4", "gata-4"},
    "GATA6": {"gata6", "gata-6"},
    "SOX9": {"sox9"},
    "HHEX": {"hhex", "hex"},
    "HNF1B": {"hnf1b", "hnf1β", "hnf-1b", "hnf-1β", "hnf1beta"},
    "HNF1A": {"hnf1a", "hnf1α", "hnf-1a", "hnf-1α", "hnf1alpha"},
    "HNF6": {"hnf6", "onecut1", "oc-1"},
    "FOXA1": {"foxa1", "foxa-1", "hnf3α", "hnf3alpha"},
    "DLK1": {"dlk1", "dlk-1", "delta-like 1"},
    "GSC": {"gsc", "goosecoid"},
    "MIXL1": {"mixl1"},
    "SSEA4": {"ssea4", "ssea-4"},
    "TRA-1-60": {"tra-1-60", "tra160"},
    "TRA-1-81": {"tra-1-81", "tra181"},
    "A1AT": {"a1at", "α1at"},  # duplicate coverage for common variant
    "NTCP": {"ntcp", "slc10a1", "sodium taurocholate cotransporting polypeptide"},
    "MRP2": {"mrp2", "abcc2"},
    "BSEP": {"bsep", "abcb11"},
    "UGT1A1": {"ugt1a1"},
    "TAT": {"tat", "tyrosine aminotransferase"},
    "TDO2": {"tdo2", "tryptophan 2,3-dioxygenase"},
    "G6PC": {"g6pc", "glucose-6-phosphatase"},
    "PCK1": {"pck1", "pepck", "phosphoenolpyruvate carboxykinase"},
}


# ------------------------------------------------------------------
# Alias table construction
# ------------------------------------------------------------------

def _fetch_mesh_entry_terms(uid: str, api_key: str | None = None) -> list[str]:
    """Fetch entry terms (synonyms) from NCBI MeSH for a descriptor UID."""
    params = {
        "db": "mesh",
        "id": uid,
        "rettype": "full",
        "retmode": "text",
    }
    if api_key:
        params["api_key"] = api_key

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        logger.warning("MeSH fetch failed for %s: %s", uid, e)
        return []

    # Parse entry terms from the MeSH record text format
    entry_terms = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Entry Term"):
            # Format: "Entry Term = hepatocyte growth factor" or similar
            parts = line.split("=", 1)
            if len(parts) == 2:
                entry_terms.append(parts[1].strip())
        elif line.startswith("MH = "):
            # Main heading
            entry_terms.append(line[5:].strip())
        # Also catch "ENTRY = " format
        elif line.startswith("ENTRY = ") or line.startswith("PRINT ENTRY = "):
            parts = line.split("=", 1)
            if len(parts) == 2:
                # ENTRY terms can have pipe-delimited qualifiers
                term = parts[1].strip().split("|")[0].strip()
                if term:
                    entry_terms.append(term)

    return entry_terms


def build_mesh_aliases(api_key: str | None = None) -> dict[str, dict[str, list[str]]]:
    """Query NCBI MeSH for all UIDs and build combined alias tables.

    Returns dict with keys 'growth_factors', 'small_molecules', 'markers',
    each mapping canonical name to list of aliases.
    """
    all_mappings = {
        "growth_factors": GROWTH_FACTOR_MESH,
        "small_molecules": SMALL_MOLECULE_MESH,
        "markers": MARKER_MESH,
    }
    manual_tables = {
        "growth_factors": MANUAL_GF_ALIASES,
        "small_molecules": MANUAL_SM_ALIASES,
        "markers": MANUAL_MARKER_ALIASES,
    }

    result: dict[str, dict[str, list[str]]] = {}

    for category, mesh_map in all_mappings.items():
        cat_aliases: dict[str, list[str]] = {}

        # Collect all unique UIDs to fetch
        fetched_uids: dict[str, list[str]] = {}
        unique_uids = set(mesh_map.values())

        for uid in sorted(unique_uids):
            logger.info("Fetching MeSH %s...", uid)
            terms = _fetch_mesh_entry_terms(uid, api_key=api_key)
            fetched_uids[uid] = terms
            # Rate limit: NCBI allows 10/sec with API key, 3/sec without
            time.sleep(0.15 if api_key else 0.4)

        # Build alias sets per canonical name
        for canonical, uid in mesh_map.items():
            aliases = set()
            # Always include the canonical name lowercase
            aliases.add(canonical.lower())
            # Add MeSH entry terms
            for term in fetched_uids.get(uid, []):
                aliases.add(term.lower())
            # Add manual overrides
            manual = manual_tables[category].get(canonical, set())
            aliases.update(manual)
            cat_aliases[canonical] = sorted(aliases)

        # Add entries that only have manual aliases (no MeSH UID)
        for canonical, manual_set in manual_tables[category].items():
            if canonical not in cat_aliases:
                aliases = {canonical.lower()} | manual_set
                cat_aliases[canonical] = sorted(aliases)

        result[category] = cat_aliases

    return result


def save_mesh_aliases(aliases: dict, path: Path = MESH_CACHE_PATH) -> None:
    """Write alias tables to JSON cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(aliases, f, indent=2, ensure_ascii=False)
    logger.info("Saved MeSH aliases to %s", path)


def load_alias_tables() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Load alias tables, preferring cached MeSH data, falling back to manual only.

    Returns (gf_aliases, sm_aliases, marker_aliases) where each maps
    canonical name -> set of lowercase aliases.
    """
    gf: dict[str, set[str]] = {}
    sm: dict[str, set[str]] = {}
    mk: dict[str, set[str]] = {}

    if MESH_CACHE_PATH.exists():
        with open(MESH_CACHE_PATH) as f:
            cached = json.load(f)
        for canonical, aliases in cached.get("growth_factors", {}).items():
            gf[canonical] = set(aliases)
        for canonical, aliases in cached.get("small_molecules", {}).items():
            sm[canonical] = set(aliases)
        for canonical, aliases in cached.get("markers", {}).items():
            mk[canonical] = set(aliases)
    else:
        logger.info("No MeSH cache found, using manual aliases only")
        for canonical, aliases in MANUAL_GF_ALIASES.items():
            gf[canonical] = {canonical.lower()} | aliases
        for canonical, aliases in MANUAL_SM_ALIASES.items():
            sm[canonical] = {canonical.lower()} | aliases
        for canonical, aliases in MANUAL_MARKER_ALIASES.items():
            mk[canonical] = {canonical.lower()} | aliases

    return gf, sm, mk


# ------------------------------------------------------------------
# Grounding logic
# ------------------------------------------------------------------

# Pre-compiled word boundary patterns for short aliases
_WB_CACHE: dict[str, re.Pattern] = {}


def _get_wb_pattern(alias: str) -> re.Pattern:
    """Get or create a word-boundary regex for a short alias."""
    if alias not in _WB_CACHE:
        _WB_CACHE[alias] = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
    return _WB_CACHE[alias]


def is_term_grounded(
    term_name: str,
    alias_table: dict[str, set[str]],
    text_lower: str,
) -> bool:
    """Check if a term appears in the source text via any known alias.

    For short aliases (<=3 chars), uses word-boundary matching to avoid
    false positives (e.g. 'ra' matching 'ratio').
    """
    if not term_name:
        return True  # skip items with no name rather than removing them

    aliases = alias_table.get(term_name)
    if aliases is None:
        # Unknown term — fall back to direct substring search
        needle = term_name.lower()
        if len(needle) <= 3:
            return bool(_get_wb_pattern(needle).search(text_lower))
        return needle in text_lower

    for alias in aliases:
        if len(alias) <= 3:
            if _get_wb_pattern(alias).search(text_lower):
                return True
        else:
            if alias in text_lower:
                return True

    return False


def ground_protocol(
    protocol: dict,
    paper_text: str,
    supplement_text: str | None = None,
) -> tuple[dict, list[dict]]:
    """Validate extracted terms against source text, removing ungrounded items.

    Returns (cleaned_protocol, removal_report) where removal_report is a list
    of dicts describing each removed item.
    """
    gf_aliases, sm_aliases, mk_aliases = load_alias_tables()

    # Build combined search text, lowercased once
    combined = paper_text
    if supplement_text:
        combined += "\n" + supplement_text
    text_lower = combined.lower()

    removals: list[dict] = []
    protocol = _deep_copy_protocol(protocol)

    # Walk stages
    for stage in (protocol.get("stages") or []):
        if not isinstance(stage, dict):
            continue
        stage_name = stage.get("stage_name", "?")

        # Check growth factors
        gf_list = stage.get("growth_factors") or []
        kept_gf = []
        for item in gf_list:
            if not isinstance(item, dict):
                kept_gf.append(item)
                continue
            name = item.get("name", "")
            if is_term_grounded(name, gf_aliases, text_lower):
                kept_gf.append(item)
            else:
                removals.append({
                    "category": "growth_factor",
                    "term": name,
                    "stage": stage_name,
                    "reason": "not_found_in_text",
                })
        stage["growth_factors"] = kept_gf

        # Check small molecules
        sm_list = stage.get("small_molecules") or []
        kept_sm = []
        for item in sm_list:
            if not isinstance(item, dict):
                kept_sm.append(item)
                continue
            name = item.get("name", "")
            if is_term_grounded(name, sm_aliases, text_lower):
                kept_sm.append(item)
            else:
                removals.append({
                    "category": "small_molecule",
                    "term": name,
                    "stage": stage_name,
                    "reason": "not_found_in_text",
                })
        stage["small_molecules"] = kept_sm

        # Check stage markers
        sm_markers = stage.get("stage_markers") or []
        kept_markers = []
        for item in sm_markers:
            if not isinstance(item, dict):
                kept_markers.append(item)
                continue
            name = item.get("marker_name", "")
            if is_term_grounded(name, mk_aliases, text_lower):
                kept_markers.append(item)
            else:
                removals.append({
                    "category": "marker",
                    "term": name,
                    "stage": stage_name,
                    "reason": "not_found_in_text",
                })
        stage["stage_markers"] = kept_markers

    # Check endpoint markers
    ea = protocol.get("endpoint_assessment")
    if isinstance(ea, dict):
        ep_markers = ea.get("markers") or []
        kept_ep = []
        for item in ep_markers:
            if not isinstance(item, dict):
                kept_ep.append(item)
                continue
            name = item.get("marker_name", "")
            if is_term_grounded(name, mk_aliases, text_lower):
                kept_ep.append(item)
            else:
                removals.append({
                    "category": "marker",
                    "term": name,
                    "stage": "endpoint",
                    "reason": "not_found_in_text",
                })
        ea["markers"] = kept_ep
        # Do NOT touch functional_assays — assay_name is a category label

    return protocol, removals


def _deep_copy_protocol(protocol: dict) -> dict:
    """Deep copy the mutable parts of a protocol dict."""
    import copy
    return copy.deepcopy(protocol)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _cli_build_aliases() -> None:
    """Build MeSH alias cache."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_keys = os.environ.get("NCBI_API_KEYS", "")
    api_key = api_keys.split(",")[0].strip() if api_keys else None

    if not api_key:
        logger.warning("No NCBI_API_KEY found — fetching will be slower (3 req/sec)")

    aliases = build_mesh_aliases(api_key=api_key)
    save_mesh_aliases(aliases)

    # Print summary
    for cat, entries in aliases.items():
        total_aliases = sum(len(v) for v in entries.values())
        print(f"  {cat}: {len(entries)} terms, {total_aliases} total aliases")


def _cli_test(pmc_id: str) -> None:
    """Test grounding on a single paper's protocols."""
    from data_layer.database import PipelineDB

    db = PipelineDB()
    paper = db.get_paper(pmc_id=pmc_id)
    if not paper:
        print(f"Paper {pmc_id} not found")
        db.close()
        return

    text_path = paper.get("parsed_text_path")
    if not text_path or not Path(text_path).exists():
        print(f"No parsed text for {pmc_id}")
        db.close()
        return

    paper_text = Path(text_path).read_text()
    supp_path = paper.get("supplement_text_path")
    supp_text = Path(supp_path).read_text() if supp_path and Path(supp_path).exists() else None

    protocols = db.get_protocols_for_paper(paper["id"])
    db.close()

    for proto in protocols:
        arm = proto.get("protocol_arm", "?")
        cleaned, removals = ground_protocol(proto, paper_text, supp_text)
        if removals:
            print(f"\n{pmc_id} / {arm}: {len(removals)} items would be removed:")
            for r in removals:
                print(f"  [{r['category']}] {r['term']} (stage: {r['stage']})")
        else:
            print(f"\n{pmc_id} / {arm}: all items grounded")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grounding: verify extracted terms against source text",
    )
    parser.add_argument("--build-aliases", action="store_true",
                        help="Build MeSH alias cache (run once)")
    parser.add_argument("--test", type=str, default=None, metavar="PMC_ID",
                        help="Test grounding on a single paper")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.build_aliases:
        _cli_build_aliases()
    elif args.test:
        _cli_test(args.test)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
