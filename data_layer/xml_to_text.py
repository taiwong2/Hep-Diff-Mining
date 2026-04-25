"""Convert PMC XML articles into structured markdown for LLM extraction.

Parses <article> elements, extracts sections with hierarchy, converts
tables to markdown format, and applies a text budget to stay within
LLM context limits.

Usage:
    from data_layer.xml_to_text import parse_pmc_xml_to_text
    parsed = parse_pmc_xml_to_text("data/PMC76/PMC7612819.xml")
    print(parsed.full_text[:500])
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Text budget: 120K chars ≈ 30K tokens for main text
MAIN_TEXT_BUDGET = 120_000

# Section priority for trimming (higher = trim first)
SECTION_PRIORITY = {
    "discussion": 5,
    "introduction": 4,
    "supplementary": 3,
    "abstract": 0,  # never trim
    "methods": 0,    # never trim
    "materials and methods": 0,
    "results": 1,    # trim only if over budget after discussion+intro
}


@dataclass
class ParsedPaper:
    pmc_id: str
    title: str
    abstract: str
    sections: dict[str, str] = field(default_factory=dict)
    full_text: str = ""
    has_methods: bool = False
    has_supplement_inline: bool = False
    char_count: int = 0
    tables_found: int = 0


def _get_all_text(elem: ET.Element) -> str:
    """Get all text content from an element, joining inline elements."""
    return "".join(elem.itertext()).strip()


def _element_to_markdown(elem: ET.Element, depth: int = 0) -> str:
    """Recursively convert an XML element to markdown text."""
    parts: list[str] = []

    if elem.tag == "table-wrap":
        parts.append(_table_wrap_to_markdown(elem))
        return "\n".join(parts)

    if elem.tag == "fig":
        caption = elem.find(".//caption")
        if caption is not None:
            cap_text = _get_all_text(caption)
            if cap_text:
                label = elem.find("label")
                label_text = _get_all_text(label) if label is not None else ""
                parts.append(f"\n**{label_text}** {cap_text}\n")
        return "\n".join(parts)

    # Handle inline elements that carry text
    if elem.text:
        parts.append(elem.text)

    for child in elem:
        if child.tag == "title":
            # Section titles become headers
            title_text = _get_all_text(child)
            if title_text:
                hashes = "#" * min(depth + 2, 6)
                parts.append(f"\n{hashes} {title_text}\n")
        elif child.tag == "sec":
            parts.append(_element_to_markdown(child, depth + 1))
        elif child.tag == "p":
            p_text = _paragraph_to_text(child)
            if p_text:
                parts.append(f"\n{p_text}\n")
        elif child.tag == "table-wrap":
            parts.append(_table_wrap_to_markdown(child))
        elif child.tag == "fig":
            parts.append(_element_to_markdown(child, depth))
        elif child.tag == "list":
            parts.append(_list_to_markdown(child))
        elif child.tag in ("supplementary-material",):
            # Inline supplementary content
            sup_text = _get_all_text(child)
            if sup_text and len(sup_text) > 50:
                parts.append(f"\n**Supplementary Material:**\n{sup_text}\n")
        else:
            # Recurse into other elements
            sub = _element_to_markdown(child, depth)
            if sub:
                parts.append(sub)

        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def _paragraph_to_text(p_elem: ET.Element) -> str:
    """Convert a <p> element to text, handling inline markup."""
    parts: list[str] = []
    if p_elem.text:
        parts.append(p_elem.text)

    for child in p_elem:
        if child.tag == "bold" or child.tag == "b":
            inner = _get_all_text(child)
            parts.append(f"**{inner}**")
        elif child.tag == "italic" or child.tag == "i":
            inner = _get_all_text(child)
            parts.append(f"*{inner}*")
        elif child.tag == "sup":
            inner = _get_all_text(child)
            parts.append(f"^{inner}")
        elif child.tag == "sub":
            inner = _get_all_text(child)
            parts.append(f"_{inner}")
        elif child.tag == "xref":
            inner = _get_all_text(child)
            parts.append(inner)
        elif child.tag == "ext-link":
            inner = _get_all_text(child)
            parts.append(inner)
        else:
            inner = _get_all_text(child)
            parts.append(inner)

        if child.tail:
            parts.append(child.tail)

    return "".join(parts).strip()


def _list_to_markdown(list_elem: ET.Element) -> str:
    """Convert a <list> element to markdown bullet list."""
    items = []
    for item in list_elem.findall("list-item"):
        text = _get_all_text(item)
        if text:
            items.append(f"- {text}")
    return "\n".join(items)


def _table_wrap_to_markdown(tw: ET.Element) -> str:
    """Convert a <table-wrap> element to a markdown table."""
    parts: list[str] = []

    # Label and caption
    label = tw.find("label")
    caption = tw.find("caption")
    label_text = _get_all_text(label) if label is not None else ""
    caption_text = ""
    if caption is not None:
        cap_title = caption.find("title")
        if cap_title is not None:
            caption_text = _get_all_text(cap_title)
        if not caption_text:
            caption_text = _get_all_text(caption)

    if label_text or caption_text:
        parts.append(f"\n**{label_text}** {caption_text}\n")

    # Parse the actual table
    table = tw.find(".//table")
    if table is not None:
        md_table = _table_to_markdown(table)
        if md_table:
            parts.append(md_table)

    return "\n".join(parts)


def _table_to_markdown(table: ET.Element) -> str:
    """Convert a <table> element to markdown table format."""
    rows: list[list[str]] = []

    # Collect all rows from thead and tbody
    for tr in table.iter("tr"):
        cells: list[str] = []
        for cell in tr:
            if cell.tag in ("th", "td"):
                text = _get_all_text(cell).replace("|", "\\|")
                text = " ".join(text.split())  # normalize whitespace
                cells.append(text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    # Normalize column count
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    # Build markdown table
    lines: list[str] = []
    # Header row
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    # Data rows
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _classify_section(title: str) -> str:
    """Classify a section title into a priority category."""
    t = title.lower().strip()
    if any(kw in t for kw in ("method", "material", "experimental", "procedure",
                               "cell culture", "differentiation protocol",
                               "cell maintenance")):
        return "methods"
    if any(kw in t for kw in ("result",)):
        return "results"
    if any(kw in t for kw in ("discuss",)):
        return "discussion"
    if any(kw in t for kw in ("introduc", "background")):
        return "introduction"
    if any(kw in t for kw in ("supplement", "supporting", "additional file")):
        return "supplementary"
    if any(kw in t for kw in ("abstract",)):
        return "abstract"
    # Default: treat as results-priority (important content)
    return "results"


def parse_pmc_xml_to_text(xml_path: str | Path) -> ParsedPaper | None:
    """Parse a PMC XML file into structured markdown text.

    Returns ParsedPaper with sections and budget-trimmed full_text,
    or None if the XML cannot be parsed.
    """
    xml_path = Path(xml_path)
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        logger.warning("Malformed XML: %s", xml_path)
        return None

    root = tree.getroot()
    article_el = root if root.tag == "article" else root.find(".//article")
    if article_el is None:
        logger.warning("No <article> element: %s", xml_path)
        return None

    meta = article_el.find(".//article-meta")
    if meta is None:
        logger.warning("No article-meta: %s", xml_path)
        return None

    # Extract IDs
    pmc_id = None
    for aid in meta.findall("article-id"):
        if aid.get("pub-id-type") == "pmcid":
            pmc_id = (aid.text or "").strip()
            break
    if not pmc_id:
        pmc_id = xml_path.stem

    # Title
    title_el = meta.find(".//article-title")
    title = _get_all_text(title_el) if title_el is not None else ""

    # Abstract
    abstract = ""
    for ab in meta.findall("abstract"):
        abstract = _extract_abstract_md(ab)
        if abstract:
            break

    # Body sections
    body = article_el.find("body")
    sections: dict[str, str] = {}
    tables_found = 0
    has_methods = False
    has_supplement_inline = False

    if body is not None:
        for sec in body.findall("sec"):
            title_elem = sec.find("title")
            sec_title = _get_all_text(title_elem) if title_elem is not None else "Untitled"
            sec_text = _element_to_markdown(sec, depth=0)

            category = _classify_section(sec_title)
            if category == "methods":
                has_methods = True

            # Count tables
            tables_found += len(list(sec.iter("table-wrap")))

            # Merge into sections dict (may have multiple under same category)
            key = sec_title
            if key in sections:
                sections[key] += "\n" + sec_text
            else:
                sections[key] = sec_text

        # Also check for floating content not in <sec> elements
        for p in body.findall("p"):
            p_text = _paragraph_to_text(p)
            if p_text:
                sections.setdefault("Body", "")
                sections["Body"] += "\n" + p_text

    # Check for inline supplementary materials
    for supp in article_el.iter("supplementary-material"):
        supp_text = _get_all_text(supp)
        if supp_text and len(supp_text) > 100:
            has_supplement_inline = True
            sections.setdefault("Supplementary Materials", "")
            sections["Supplementary Materials"] += f"\n{supp_text}\n"

    # Also check back matter
    back = article_el.find("back")
    if back is not None:
        for sec in back.findall(".//sec"):
            title_elem = sec.find("title")
            sec_title = _get_all_text(title_elem) if title_elem is not None else "Back Matter"
            sec_text = _element_to_markdown(sec, depth=0)
            if sec_text.strip():
                sections[sec_title] = sec_text

    # Build full text with budget management
    full_text = _build_budgeted_text(title, abstract, sections, MAIN_TEXT_BUDGET)

    return ParsedPaper(
        pmc_id=pmc_id,
        title=title,
        abstract=abstract,
        sections=sections,
        full_text=full_text,
        has_methods=has_methods,
        has_supplement_inline=has_supplement_inline,
        char_count=len(full_text),
        tables_found=tables_found,
    )


def _extract_abstract_md(abstract_elem: ET.Element) -> str:
    """Extract abstract as markdown."""
    parts: list[str] = []
    sections = abstract_elem.findall("sec")
    if sections:
        for sec in sections:
            title_el = sec.find("title")
            if title_el is not None:
                title_text = _get_all_text(title_el)
                if title_text:
                    parts.append(f"**{title_text}**")
            for p in sec.findall("p"):
                p_text = _paragraph_to_text(p)
                if p_text:
                    parts.append(p_text)
    else:
        for p in abstract_elem.findall("p"):
            p_text = _paragraph_to_text(p)
            if p_text:
                parts.append(p_text)

    if not parts:
        fallback = _get_all_text(abstract_elem)
        if fallback:
            parts.append(fallback)

    return "\n".join(parts)


def _build_budgeted_text(title: str, abstract: str,
                         sections: dict[str, str],
                         budget: int) -> str:
    """Assemble full text from sections, trimming lowest-priority first."""
    # Always include title + abstract
    header = f"# {title}\n\n## Abstract\n{abstract}\n\n"
    remaining = budget - len(header)

    # Classify and sort sections by priority
    classified: list[tuple[int, str, str]] = []
    for sec_title, sec_text in sections.items():
        category = _classify_section(sec_title)
        priority = SECTION_PRIORITY.get(category, 2)
        classified.append((priority, sec_title, sec_text))

    # Sort: lowest priority number first (most important first)
    classified.sort(key=lambda x: x[0])

    # Add sections respecting budget
    included: list[str] = [header]
    total = len(header)

    for priority, sec_title, sec_text in classified:
        sec_block = f"## {sec_title}\n{sec_text}\n\n"
        if total + len(sec_block) <= budget:
            included.append(sec_block)
            total += len(sec_block)
        elif priority <= 1:
            # Must-include sections: truncate to fit
            available = budget - total
            if available > 200:
                truncated = sec_block[:available - 50] + "\n\n[... truncated ...]\n"
                included.append(truncated)
                total += len(truncated)
        # else: skip this section (over budget)

    return "".join(included)


def extract_ref_list(xml_path: str | Path) -> list[dict]:
    """Extract references from <ref-list> in a PMC XML.

    Returns list of dicts with 'doi', 'pmid', 'pmc_id', 'title',
    'first_author_surname', 'year', and 'label' fields.
    Used by the reference graph builder and DOI resolution.
    """
    xml_path = Path(xml_path)
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    refs: list[dict] = []

    for ref in root.iter("ref"):
        ref_dict: dict[str, str | None] = {
            "doi": None, "pmid": None, "pmc_id": None, "title": None,
            "first_author_surname": None, "year": None, "label": None,
        }
        for pub_id in ref.findall(".//pub-id"):
            id_type = pub_id.get("pub-id-type", "")
            text = (pub_id.text or "").strip()
            if id_type == "doi":
                ref_dict["doi"] = text
            elif id_type == "pmid":
                ref_dict["pmid"] = text
            elif id_type in ("pmcid", "pmc"):
                ref_dict["pmc_id"] = text

        # Try to get title from article-title or source
        for title_tag in ("article-title", "source"):
            title_el = ref.find(f".//{title_tag}")
            if title_el is not None:
                ref_dict["title"] = _get_all_text(title_el)
                break

        # Extract label (e.g. "17", "1.", "[3]")
        label_el = ref.find("label")
        if label_el is not None:
            ref_dict["label"] = (label_el.text or "").strip().strip(".[]()")

        # Extract first author surname
        # Look in citation elements (element-citation, mixed-citation, nlm-citation)
        cite_el = None
        for tag in ("element-citation", "mixed-citation", "nlm-citation", "citation"):
            cite_el = ref.find(f".//{tag}")
            if cite_el is not None:
                break
        search_root = cite_el if cite_el is not None else ref

        # Try <person-group> first, then bare <name>
        surname_el = search_root.find(".//person-group/name/surname")
        if surname_el is None:
            surname_el = search_root.find(".//name/surname")
        if surname_el is None:
            surname_el = search_root.find(".//surname")
        if surname_el is not None:
            ref_dict["first_author_surname"] = (surname_el.text or "").strip()

        # Extract year
        year_el = search_root.find(".//year")
        if year_el is not None:
            ref_dict["year"] = (year_el.text or "").strip()

        # Include if we have any identifier OR surname+year (for matching)
        if (ref_dict["doi"] or ref_dict["pmid"] or ref_dict["pmc_id"]
                or (ref_dict["first_author_surname"] and ref_dict["year"])):
            refs.append(ref_dict)

    return refs
