"""Tests for data_layer.reference_graph — citation DAG and topological ordering."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from data_layer.database import PipelineDB
from data_layer.reference_graph import build_reference_graph


@pytest.fixture()
def graph_db(tmp_path: Path) -> PipelineDB:
    """DB with papers that have XML files containing cross-references."""
    db = PipelineDB(db_path=tmp_path / "graph.db")

    # Create three papers: paper_c cites paper_b, paper_b cites paper_a.
    # Expected order: A -> B -> C
    papers = [
        ("PMC_A", "10.1000/a", "Foundational protocol A", "primary_protocol"),
        ("PMC_B", "10.1000/b", "Protocol B based on A", "primary_protocol"),
        ("PMC_C", "10.1000/c", "Protocol C based on B", "disease_model"),
    ]

    for pmc_id, doi, title, category in papers:
        db._conn.execute(
            """INSERT INTO papers (pmc_id, doi, title, triage_category, xml_path)
               VALUES (?, ?, ?, ?, ?)""",
            (pmc_id, doi, title, category, None),
        )
    db._conn.commit()

    # Create minimal XML files with reference lists
    xml_dir = tmp_path / "xmls"
    xml_dir.mkdir()

    # Paper A: no in-corpus references
    _write_minimal_xml(xml_dir / "a.xml", "PMC_A", "10.1000/a", [])

    # Paper B: references Paper A
    _write_minimal_xml(xml_dir / "b.xml", "PMC_B", "10.1000/b", [
        {"doi": "10.1000/a", "surname": "Smith", "year": "2020"},
    ])

    # Paper C: references Paper B
    _write_minimal_xml(xml_dir / "c.xml", "PMC_C", "10.1000/c", [
        {"doi": "10.1000/b", "surname": "Jones", "year": "2021"},
    ])

    # Update XML paths in DB
    for pmc_id, xml_name in [("PMC_A", "a.xml"), ("PMC_B", "b.xml"), ("PMC_C", "c.xml")]:
        db._conn.execute(
            "UPDATE papers SET xml_path = ? WHERE pmc_id = ?",
            (str(xml_dir / xml_name), pmc_id),
        )
    db._conn.commit()

    yield db
    db.close()


def _write_minimal_xml(path: Path, pmc_id: str, doi: str,
                        refs: list[dict]) -> None:
    """Write a minimal PMC XML with a reference list."""
    ref_entries = []
    for i, ref in enumerate(refs, 1):
        entry = f"""<ref id="R{i}">
            <element-citation>
                <person-group person-group-type="author">
                    <name><surname>{ref['surname']}</surname></name>
                </person-group>
                <year>{ref['year']}</year>
                <pub-id pub-id-type="doi">{ref['doi']}</pub-id>
            </element-citation>
        </ref>"""
        ref_entries.append(entry)

    ref_list_xml = "<ref-list>" + "".join(ref_entries) + "</ref-list>" if ref_entries else ""

    xml = f"""<article>
        <front><article-meta>
            <article-id pub-id-type="pmcid">{pmc_id}</article-id>
            <article-id pub-id-type="doi">{doi}</article-id>
            <title-group><article-title>Test</article-title></title-group>
        </article-meta></front>
        <body><sec><title>Methods</title><p>Test.</p></sec></body>
        <back>{ref_list_xml}</back>
    </article>"""
    path.write_text(xml)


class TestBuildReferenceGraph:

    def test_returns_topological_order(self, graph_db: PipelineDB):
        order = build_reference_graph(graph_db)
        assert isinstance(order, list)
        assert len(order) >= 3

        # Get paper IDs
        paper_a = graph_db.get_paper(pmc_id="PMC_A")
        paper_b = graph_db.get_paper(pmc_id="PMC_B")
        paper_c = graph_db.get_paper(pmc_id="PMC_C")

        # A should come before B, B before C
        idx_a = order.index(paper_a["id"])
        idx_b = order.index(paper_b["id"])
        idx_c = order.index(paper_c["id"])

        assert idx_a < idx_b, "Foundational paper A should be before B"
        assert idx_b < idx_c, "Paper B should be before C"

    def test_sets_processing_priority(self, graph_db: PipelineDB):
        build_reference_graph(graph_db)

        paper_a = graph_db.get_paper(pmc_id="PMC_A")
        paper_b = graph_db.get_paper(pmc_id="PMC_B")
        paper_c = graph_db.get_paper(pmc_id="PMC_C")

        assert paper_a["processing_priority"] < paper_b["processing_priority"]
        assert paper_b["processing_priority"] < paper_c["processing_priority"]

    def test_adds_references_to_db(self, graph_db: PipelineDB):
        build_reference_graph(graph_db)

        paper_b = graph_db.get_paper(pmc_id="PMC_B")
        refs = graph_db.get_references(paper_b["id"])
        assert len(refs) >= 1
        assert any(r["referenced_doi"] == "10.1000/a" for r in refs)

    def test_handles_no_xml_paths(self, tmp_path: Path):
        """Papers without XML paths are excluded from extraction order."""
        db = PipelineDB(db_path=tmp_path / "no_xml.db")
        db._conn.execute(
            """INSERT INTO papers (pmc_id, title, triage_category)
               VALUES ('PMC_X', 'Test', 'primary_protocol')"""
        )
        db._conn.commit()

        order = build_reference_graph(db)
        assert len(order) == 0
        db.close()


class TestCyclicReferences:

    def test_breaks_cycles_gracefully(self, tmp_path: Path):
        """Cyclic citations should be handled by removing edges."""
        db = PipelineDB(db_path=tmp_path / "cycle.db")
        xml_dir = tmp_path / "xmls"
        xml_dir.mkdir()

        # A -> B -> A (cycle)
        for pmc_id, doi in [("PMC_X", "10.1/x"), ("PMC_Y", "10.1/y")]:
            db._conn.execute(
                "INSERT INTO papers (pmc_id, doi, title, triage_category, xml_path) "
                "VALUES (?, ?, 'Test', 'primary_protocol', ?)",
                (pmc_id, doi, str(xml_dir / f"{pmc_id}.xml")),
            )
        db._conn.commit()

        _write_minimal_xml(xml_dir / "PMC_X.xml", "PMC_X", "10.1/x",
                           [{"doi": "10.1/y", "surname": "A", "year": "2020"}])
        _write_minimal_xml(xml_dir / "PMC_Y.xml", "PMC_Y", "10.1/y",
                           [{"doi": "10.1/x", "surname": "B", "year": "2021"}])

        # Should not raise — cycles are broken
        order = build_reference_graph(db)
        assert len(order) == 2
        db.close()


class TestIsolatedPapers:

    def test_isolated_papers_appended(self, tmp_path: Path):
        """Papers with no citation links should appear at the end of the order."""
        db = PipelineDB(db_path=tmp_path / "isolated.db")
        xml_dir = tmp_path / "xmls"
        xml_dir.mkdir()

        for pmc_id in ["PMC_P", "PMC_Q"]:
            xml_path = xml_dir / f"{pmc_id}.xml"
            _write_minimal_xml(xml_path, pmc_id, f"10.1/{pmc_id}", [])
            db._conn.execute(
                "INSERT INTO papers (pmc_id, doi, title, triage_category, xml_path) "
                "VALUES (?, ?, 'Test', 'primary_protocol', ?)",
                (pmc_id, f"10.1/{pmc_id}", str(xml_path)),
            )
        db._conn.commit()

        order = build_reference_graph(db)
        assert len(order) == 2
        db.close()
