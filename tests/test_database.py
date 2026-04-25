"""Tests for data_layer.database — PipelineDB SQLite operations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_layer.database import (
    EXTRACTABLE_CATEGORIES,
    PipelineDB,
    REVIEW_CATEGORY,
)


# ------------------------------------------------------------------ #
# Database creation and schema
# ------------------------------------------------------------------ #

class TestDatabaseCreation:

    def test_creates_db_file(self, tmp_path: Path):
        db_path = tmp_path / "subdir" / "test.db"
        db = PipelineDB(db_path=db_path)
        assert db_path.exists()
        db.close()

    def test_tables_exist(self, temp_db: PipelineDB):
        cur = temp_db._conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cur.fetchall()}
        expected = {"papers", "triage_results", "paper_references",
                    "protocols", "corpus_cache", "processing_log"}
        assert expected.issubset(tables)

    def test_wal_mode_enabled(self, temp_db: PipelineDB):
        mode = temp_db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, temp_db: PipelineDB):
        fk = temp_db._conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_context_manager(self, tmp_path: Path):
        db_path = tmp_path / "ctx.db"
        with PipelineDB(db_path=db_path) as db:
            assert db_path.exists()
            db._conn.execute("SELECT 1")


# ------------------------------------------------------------------ #
# Import from triage JSONL
# ------------------------------------------------------------------ #

class TestImportFromTriageJsonl:

    def test_imports_records(self, temp_db: PipelineDB, triage_jsonl: Path):
        count = temp_db.import_from_triage_jsonl(triage_jsonl)
        assert count == 2

    def test_imported_papers_have_correct_fields(self, temp_db: PipelineDB, triage_jsonl: Path):
        temp_db.import_from_triage_jsonl(triage_jsonl)
        paper = temp_db.get_paper(pmc_id="PMC1111111")
        assert paper is not None
        assert paper["doi"] == "10.1000/test1"
        assert paper["title"] == "Test paper one"
        assert paper["triage_category"] == "primary_protocol"

    def test_triage_results_stored(self, temp_db: PipelineDB, triage_jsonl: Path):
        temp_db.import_from_triage_jsonl(triage_jsonl)
        paper = temp_db.get_paper(pmc_id="PMC1111111")
        triage = temp_db.get_triage_result(paper["id"])
        assert triage is not None
        assert triage["category"] == "primary_protocol"
        assert triage["confidence"] == 0.95
        assert triage["supplement_signal"] == 1
        assert isinstance(triage["base_protocols"], list)
        assert "Si-Tayeb 2010" in triage["base_protocols"]

    def test_idempotent_import(self, temp_db: PipelineDB, triage_jsonl: Path):
        first_count = temp_db.import_from_triage_jsonl(triage_jsonl)
        second_count = temp_db.import_from_triage_jsonl(triage_jsonl)
        assert first_count == 2
        assert second_count == 0  # no duplicates

    def test_skips_malformed_lines(self, temp_db: PipelineDB, tmp_path: Path):
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text('{"pmc_id": "PMC1234567", "category": "review"}\nnot json\n\n')
        count = temp_db.import_from_triage_jsonl(jsonl)
        assert count == 1


# ------------------------------------------------------------------ #
# Paper lookups
# ------------------------------------------------------------------ #

class TestPaperLookups:

    def test_get_paper_by_pmc_id(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        assert paper is not None
        assert paper["pmc_id"] == "PMC4894932"

    def test_get_paper_by_doi(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(doi="10.1007/s00204-016-1689-8")
        assert paper is not None
        assert paper["pmc_id"] == "PMC4894932"

    def test_get_paper_by_id(self, populated_db: PipelineDB):
        paper1 = populated_db.get_paper(pmc_id="PMC4894932")
        paper2 = populated_db.get_paper(paper_id=paper1["id"])
        assert paper2 is not None
        assert paper2["pmc_id"] == "PMC4894932"

    def test_get_paper_returns_none_for_missing(self, populated_db: PipelineDB):
        assert populated_db.get_paper(pmc_id="PMC0000000") is None
        assert populated_db.get_paper(doi="10.0000/nonexistent") is None
        assert populated_db.get_paper(paper_id=99999) is None

    def test_get_paper_no_args_returns_none(self, populated_db: PipelineDB):
        assert populated_db.get_paper() is None


# ------------------------------------------------------------------ #
# Update helpers
# ------------------------------------------------------------------ #

class TestUpdatePaper:

    def test_update_single_field(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.update_paper(paper["id"], parsed_text_path="/tmp/test.md")
        updated = populated_db.get_paper(paper_id=paper["id"])
        assert updated["parsed_text_path"] == "/tmp/test.md"

    def test_update_multiple_fields(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.update_paper(
            paper["id"],
            extraction_status="completed",
            supplement_dir="/data/supps",
        )
        updated = populated_db.get_paper(paper_id=paper["id"])
        assert updated["extraction_status"] == "completed"
        assert updated["supplement_dir"] == "/data/supps"

    def test_set_extraction_status(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.set_extraction_status(paper["id"], "in_progress")
        updated = populated_db.get_paper(paper_id=paper["id"])
        assert updated["extraction_status"] == "in_progress"

    def test_update_with_no_fields_is_noop(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.update_paper(paper["id"])  # no fields
        updated = populated_db.get_paper(paper_id=paper["id"])
        assert updated["title"] == paper["title"]


# ------------------------------------------------------------------ #
# Protocol storage and retrieval
# ------------------------------------------------------------------ #

class TestProtocols:

    @pytest.fixture()
    def sample_protocol(self) -> dict:
        return {
            "protocol_arm": "Standard differentiation",
            "is_optimized": False,
            "cell_source": {"type": "iPSC", "line_name": "WTC-11"},
            "culture_system": {"type": "2D", "surface": "Matrigel"},
            "stages": [
                {
                    "stage_name": "Definitive endoderm",
                    "duration_days": 3,
                    "growth_factors": [
                        {"name": "Activin A", "concentration": "100 ng/ml"}
                    ],
                    "small_molecules": [],
                },
                {
                    "stage_name": "Hepatic specification",
                    "duration_days": 5,
                    "growth_factors": [
                        {"name": "HGF", "concentration": "10 ng/ml"}
                    ],
                    "small_molecules": [
                        {"name": "DMSO", "concentration": "1%"}
                    ],
                },
            ],
            "endpoint_assessment": {
                "markers": ["ALB", "AFP", "CYP3A4"],
                "functional_assays": ["albumin ELISA"],
            },
            "modifications": None,
            "step_sources": None,
            "base_protocol_doi": "10.1002/hep.23354",
            "extraction_confidence": 0.85,
            "extraction_notes": "Well-documented protocol",
            "incomplete_flags": [],
            "pass_number": 2,
        }

    def test_store_protocol(self, populated_db: PipelineDB, sample_protocol: dict):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        proto_id = populated_db.store_protocol(paper["id"], sample_protocol)
        assert proto_id > 0

    def test_get_protocols_for_paper(self, populated_db: PipelineDB, sample_protocol: dict):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.store_protocol(paper["id"], sample_protocol)

        protos = populated_db.get_protocols_for_paper(paper["id"])
        assert len(protos) == 1
        proto = protos[0]
        assert proto["protocol_arm"] == "Standard differentiation"
        assert proto["extraction_confidence"] == 0.85
        assert proto["base_protocol_doi"] == "10.1002/hep.23354"

    def test_protocols_json_fields_deserialized(self, populated_db: PipelineDB, sample_protocol: dict):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.store_protocol(paper["id"], sample_protocol)

        protos = populated_db.get_protocols_for_paper(paper["id"])
        proto = protos[0]

        # JSON fields should be deserialized to Python objects
        assert isinstance(proto["cell_source"], dict)
        assert proto["cell_source"]["type"] == "iPSC"

        assert isinstance(proto["stages"], list)
        assert len(proto["stages"]) == 2
        assert proto["stages"][0]["stage_name"] == "Definitive endoderm"

        assert isinstance(proto["endpoint_assessment"], dict)
        assert "ALB" in proto["endpoint_assessment"]["markers"]

    def test_update_protocol(self, populated_db: PipelineDB, sample_protocol: dict):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        proto_id = populated_db.store_protocol(paper["id"], sample_protocol)

        populated_db.update_protocol(proto_id, {
            "extraction_confidence": 0.92,
            "pass_number": 3,
            "stages": sample_protocol["stages"] + [
                {"stage_name": "Maturation", "duration_days": 10}
            ],
        })

        protos = populated_db.get_protocols_for_paper(paper["id"])
        assert protos[0]["extraction_confidence"] == 0.92
        assert protos[0]["pass_number"] == 3
        assert len(protos[0]["stages"]) == 3

    def test_delete_protocols_for_paper(self, populated_db: PipelineDB, sample_protocol: dict):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.store_protocol(paper["id"], sample_protocol)
        populated_db.store_protocol(paper["id"], {**sample_protocol, "protocol_arm": "Arm 2"})

        deleted = populated_db.delete_protocols_for_paper(paper["id"])
        assert deleted == 2
        assert populated_db.get_protocols_for_paper(paper["id"]) == []

    def test_multiple_protocols_per_paper(self, populated_db: PipelineDB, sample_protocol: dict):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.store_protocol(paper["id"], sample_protocol)
        populated_db.store_protocol(paper["id"], {**sample_protocol, "protocol_arm": "3D spheroid"})

        protos = populated_db.get_protocols_for_paper(paper["id"])
        assert len(protos) == 2
        arms = {p["protocol_arm"] for p in protos}
        assert arms == {"Standard differentiation", "3D spheroid"}


# ------------------------------------------------------------------ #
# References
# ------------------------------------------------------------------ #

class TestReferences:

    def test_add_and_get_reference(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.add_reference(paper["id"], doi="10.1038/nmat1001")

        refs = populated_db.get_references(paper["id"])
        assert len(refs) >= 1
        assert any(r["referenced_doi"] == "10.1038/nmat1001" for r in refs)

    def test_in_corpus_flag_when_doi_matches(self, populated_db: PipelineDB):
        """Reference to an in-corpus paper (by DOI) should set in_corpus=1."""
        paper1 = populated_db.get_paper(pmc_id="PMC4894932")
        paper2 = populated_db.get_paper(pmc_id="PMC10033665")
        populated_db.add_reference(paper1["id"], doi=paper2["doi"])

        refs = populated_db.get_references(paper1["id"])
        matching = [r for r in refs if r["referenced_doi"] == paper2["doi"]]
        assert len(matching) == 1
        assert matching[0]["in_corpus"] == 1
        assert matching[0]["referenced_id"] == paper2["id"]


# ------------------------------------------------------------------ #
# Corpus cache
# ------------------------------------------------------------------ #

class TestCorpusCache:

    def test_cache_and_retrieve_by_doi(self, temp_db: PipelineDB):
        temp_db.cache_text(
            doi="10.1002/hep.23354",
            pmc_id="PMC5555555",
            title="Test cached paper",
            text="Full text of the paper...",
            source="fetch_reference",
        )
        cached = temp_db.get_cached_text(doi="10.1002/hep.23354")
        assert cached is not None
        assert cached["title"] == "Test cached paper"
        assert "Full text" in cached["full_text"]

    def test_cache_and_retrieve_by_pmc_id(self, temp_db: PipelineDB):
        temp_db.cache_text(
            doi=None,
            pmc_id="PMC5555555",
            title="PMC cached",
            text="Some text",
            source="test",
        )
        cached = temp_db.get_cached_text(pmc_id="PMC5555555")
        assert cached is not None

    def test_cache_update_prefers_longer_text(self, temp_db: PipelineDB):
        temp_db.cache_text("10.1/test", None, "Title", "Short", "test")
        temp_db.cache_text("10.1/test", None, "Title", "Much longer text content here", "test2")

        cached = temp_db.get_cached_text(doi="10.1/test")
        assert cached["full_text"] == "Much longer text content here"

    def test_cache_does_not_replace_longer_with_shorter(self, temp_db: PipelineDB):
        temp_db.cache_text("10.1/test", None, "Title", "This is the longer original text", "test")
        temp_db.cache_text("10.1/test", None, "Title", "Short", "test2")

        cached = temp_db.get_cached_text(doi="10.1/test")
        assert cached["full_text"] == "This is the longer original text"

    def test_get_cached_text_returns_none_for_missing(self, temp_db: PipelineDB):
        assert temp_db.get_cached_text(doi="10.0000/nonexistent") is None
        assert temp_db.get_cached_text(pmc_id="PMC0000000") is None


# ------------------------------------------------------------------ #
# Search corpus
# ------------------------------------------------------------------ #

class TestSearchCorpus:

    def test_search_by_doi(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.store_protocol(paper["id"], {
            "protocol_arm": "test arm",
            "stages": [],
            "extraction_confidence": 0.8,
        })

        results = populated_db.search_corpus("10.1007/s00204-016-1689-8")
        assert len(results) >= 1
        assert results[0]["doi"] == "10.1007/s00204-016-1689-8"

    def test_search_by_title_keyword(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.store_protocol(paper["id"], {
            "protocol_arm": "test",
            "stages": [],
        })

        results = populated_db.search_corpus("shear stress")
        assert len(results) >= 1

    def test_search_returns_empty_for_no_match(self, populated_db: PipelineDB):
        results = populated_db.search_corpus("zzzzz_nonexistent_zzzzz")
        assert results == []


# ------------------------------------------------------------------ #
# Papers for extraction
# ------------------------------------------------------------------ #

class TestPapersForExtraction:

    def test_get_papers_for_extraction_excludes_completed(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.update_paper(paper["id"],
                                  parsed_text_path="/tmp/test.md",
                                  extraction_status="completed")

        papers = populated_db.get_papers_for_extraction()
        pmc_ids = [p["pmc_id"] for p in papers]
        assert "PMC4894932" not in pmc_ids

    def test_get_papers_for_extraction_requires_text(self, populated_db: PipelineDB):
        """Papers without parsed_text_path should not be returned."""
        papers = populated_db.get_papers_for_extraction()
        for p in papers:
            assert p["parsed_text_path"] is not None

    def test_get_papers_for_extraction_filters_by_category(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.update_paper(paper["id"], parsed_text_path="/tmp/t.md")

        papers = populated_db.get_papers_for_extraction(category="primary_protocol")
        pmc_ids = [p["pmc_id"] for p in papers]
        assert "PMC4894932" in pmc_ids

        papers = populated_db.get_papers_for_extraction(category="disease_model")
        pmc_ids = [p["pmc_id"] for p in papers]
        assert "PMC4894932" not in pmc_ids


# ------------------------------------------------------------------ #
# Processing log
# ------------------------------------------------------------------ #

class TestProcessingLog:

    def test_log_processing(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.log_processing(
            paper["id"],
            stage="extraction_pass2",
            status="completed",
            tokens_used=5000,
            duration_secs=12.5,
        )
        rows = populated_db._conn.execute(
            "SELECT * FROM processing_log WHERE paper_id = ?", (paper["id"],)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["stage"] == "extraction_pass2"
        assert rows[0]["tokens_used"] == 5000


# ------------------------------------------------------------------ #
# Statistics
# ------------------------------------------------------------------ #

class TestStats:

    def test_get_stats(self, populated_db: PipelineDB):
        stats = populated_db.get_stats()
        assert stats["total_papers"] == 3
        assert "by_category" in stats
        assert stats["by_category"]["primary_protocol"] == 1
        assert stats["by_category"]["disease_model"] == 1
        assert stats["by_category"]["review"] == 1
        assert stats["protocols_extracted"] == 0
        assert stats["corpus_cache_entries"] == 0

    def test_stats_count_protocols(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.store_protocol(paper["id"], {
            "protocol_arm": "arm1",
            "stages": [],
        })
        populated_db.store_protocol(paper["id"], {
            "protocol_arm": "arm2",
            "stages": [],
        })
        stats = populated_db.get_stats()
        assert stats["protocols_extracted"] == 2


# ------------------------------------------------------------------ #
# Triage results
# ------------------------------------------------------------------ #

class TestTriageResults:

    def test_get_triage_result(self, populated_db: PipelineDB):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        triage = populated_db.get_triage_result(paper["id"])
        assert triage is not None
        assert triage["category"] == "primary_protocol"
        assert triage["confidence"] == 0.9
        assert isinstance(triage["base_protocols"], list)
        assert isinstance(triage["key_cell_types"], list)

    def test_get_triage_result_returns_none_for_missing(self, populated_db: PipelineDB):
        assert populated_db.get_triage_result(99999) is None
