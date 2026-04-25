"""Shared fixtures for CellDifferentiationMining test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_layer.database import PipelineDB

# Paths to example PMC XMLs shipped in the repository
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_XML_PMC4894932 = EXAMPLES_DIR / "PMC4894932.xml"
EXAMPLE_XML_PMC10033665 = EXAMPLES_DIR / "PMC10033665.xml"


@pytest.fixture()
def example_xml_4894932() -> Path:
    """Path to example PMC4894932.xml (Rashidi et al., fluid shear stress)."""
    assert EXAMPLE_XML_PMC4894932.exists(), f"Missing example XML: {EXAMPLE_XML_PMC4894932}"
    return EXAMPLE_XML_PMC4894932


@pytest.fixture()
def example_xml_10033665() -> Path:
    """Path to example PMC10033665.xml (hemophilia A iPSC gene editing)."""
    assert EXAMPLE_XML_PMC10033665.exists(), f"Missing example XML: {EXAMPLE_XML_PMC10033665}"
    return EXAMPLE_XML_PMC10033665


@pytest.fixture()
def temp_db(tmp_path: Path) -> PipelineDB:
    """Create a temporary PipelineDB backed by a temp-dir SQLite file."""
    db_path = tmp_path / "test.db"
    db = PipelineDB(db_path=db_path)
    yield db
    db.close()


@pytest.fixture()
def populated_db(temp_db: PipelineDB) -> PipelineDB:
    """A PipelineDB pre-loaded with a few test papers and triage results."""
    cur = temp_db._conn.cursor()

    # Insert three papers
    papers = [
        ("PMC4894932", "10.1007/s00204-016-1689-8", "26979076",
         "Fluid shear stress modulation of hepatocyte-like cell function",
         "primary_protocol"),
        ("PMC10033665", "10.3389/fgene.2023.1115831", "36968612",
         "Correction of F8 intron 1 inversion in hemophilia A patient-specific iPSCs",
         "disease_model"),
        ("PMC9999999", "10.1000/test.review", None,
         "A review of hepatocyte differentiation protocols",
         "review"),
    ]
    for pmc_id, doi, pmid, title, category in papers:
        cur.execute(
            """INSERT INTO papers (pmc_id, doi, pmid, title, triage_category)
               VALUES (?, ?, ?, ?, ?)""",
            (pmc_id, doi, pmid, title, category),
        )
        paper_id = cur.lastrowid
        cur.execute(
            """INSERT INTO triage_results
               (paper_id, category, confidence, reasoning,
                base_protocols, key_cell_types, disease_context, supplement_signal)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (paper_id, category, 0.9, "test reasoning",
             json.dumps(["Szkolnicka 2014"]), json.dumps(["iPSC", "HLC"]),
             None, 0),
        )

    temp_db._conn.commit()
    return temp_db


@pytest.fixture()
def triage_jsonl(tmp_path: Path) -> Path:
    """Create a small triage JSONL file for import testing."""
    records = [
        {
            "pmc_id": "PMC1111111",
            "doi": "10.1000/test1",
            "pmid": "11111111",
            "title": "Test paper one",
            "category": "primary_protocol",
            "confidence": 0.95,
            "reasoning": "Clear hepatocyte differentiation protocol",
            "base_protocols": ["Si-Tayeb 2010"],
            "key_cell_types": ["iPSC", "hepatocyte"],
            "disease_context": None,
            "supplement_signal": True,
        },
        {
            "pmc_id": "PMC2222222",
            "doi": "10.1000/test2",
            "pmid": "22222222",
            "title": "Test paper two",
            "category": "not_relevant",
            "confidence": 0.8,
            "reasoning": "Not about hepatocyte differentiation",
            "base_protocols": [],
            "key_cell_types": [],
            "disease_context": None,
            "supplement_signal": False,
        },
    ]
    jsonl_path = tmp_path / "triage_results.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return jsonl_path
