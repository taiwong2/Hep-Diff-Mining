"""Demonstration script for the Cell Differentiation Mining pipeline.

Runs the XML-to-markdown conversion on example PMC papers to demonstrate
the data processing stages without requiring API keys.

Usage:
    python demo.py

Expected run time: <30 seconds on a standard desktop.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def demo_xml_parsing() -> None:
    """Demonstrate Stage 2: PMC XML to structured markdown conversion."""
    from data_layer.xml_to_text import parse_pmc_xml_to_text

    xml_files = sorted(EXAMPLES_DIR.glob("*.xml"))
    if not xml_files:
        print("No example XML files found in examples/")
        sys.exit(1)

    print(f"Found {len(xml_files)} example paper(s)\n")

    for xml_path in xml_files:
        pmc_id = xml_path.stem
        print(f"{'=' * 60}")
        print(f"Parsing: {pmc_id}")
        print(f"{'=' * 60}")

        t0 = time.time()
        parsed = parse_pmc_xml_to_text(str(xml_path))
        elapsed = time.time() - t0

        if not parsed or not parsed.full_text:
            print(f"  Failed to parse {xml_path.name}\n")
            continue

        print(f"  Title:    {parsed.title}")
        print(f"  PMC ID:   {parsed.pmc_id}")
        print(f"  Sections: {len(parsed.sections)}")
        print(f"  Tables:   {parsed.tables_found}")
        print(f"  Text length: {len(parsed.full_text):,} characters")
        print(f"  Parse time:  {elapsed:.2f}s")

        if parsed.sections:
            section_names = list(parsed.sections.keys())
            print(f"\n  Section structure:")
            for name in section_names[:10]:
                print(f"    - {name}")
            if len(section_names) > 10:
                print(f"    ... and {len(section_names) - 10} more")

        # Show first 500 chars of output
        print(f"\n  Output preview (first 500 chars):")
        preview = parsed.full_text[:500].replace("\n", "\n    ")
        print(f"    {preview}")
        print()


def demo_database() -> None:
    """Demonstrate Stage 1: database schema initialization."""
    import sqlite3
    import tempfile

    from data_layer.database import PipelineDB

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "demo.db"
        db = PipelineDB(db_path=str(db_path))

        print(f"{'=' * 60}")
        print("Database schema")
        print(f"{'=' * 60}")

        tables = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        for (table_name,) in tables:
            cols = db._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            col_names = [c[1] for c in cols]
            print(f"\n  {table_name}:")
            for col in col_names:
                print(f"    - {col}")

        db.close()
        print()


def demo_reference_graph() -> None:
    """Demonstrate Stage 5: citation graph construction."""
    from data_layer.xml_to_text import parse_pmc_xml_to_text

    xml_files = sorted(EXAMPLES_DIR.glob("*.xml"))

    print(f"{'=' * 60}")
    print("Reference extraction")
    print(f"{'=' * 60}")

    for xml_path in xml_files:
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        refs = tree.findall(".//ref-list/ref")
        print(f"\n  {xml_path.stem}: {len(refs)} references found")
        for ref in refs[:5]:
            pub_id = ref.find(".//pub-id[@pub-id-type='doi']")
            title_el = ref.find(".//article-title")
            title = title_el.text[:60] if title_el is not None and title_el.text else "(no title)"
            doi = pub_id.text if pub_id is not None and pub_id.text else "(no DOI)"
            print(f"    - {title}... [DOI: {doi}]")
        if len(refs) > 5:
            print(f"    ... and {len(refs) - 5} more")

    print()


def main() -> None:
    print("Cell Differentiation Mining — Demo")
    print("=" * 60)
    print("This demo shows the data processing pipeline stages")
    print("using example PMC papers (no API keys required).\n")

    t0 = time.time()

    demo_xml_parsing()
    demo_database()
    demo_reference_graph()

    total = time.time() - t0
    print(f"{'=' * 60}")
    print(f"Demo complete in {total:.1f}s")
    print(f"{'=' * 60}")
    print("\nTo run the full pipeline (requires API keys):")
    print("  1. Copy .env.example to .env and fill in your API keys")
    print("  2. Install NCBI EDirect: https://www.ncbi.nlm.nih.gov/books/NBK179288/")
    print("  3. Run: python run_pipeline.py")


if __name__ == "__main__":
    main()
