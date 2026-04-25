"""Tests for data_layer.xml_to_text — PMC XML to structured markdown conversion."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from data_layer.xml_to_text import (
    MAIN_TEXT_BUDGET,
    ParsedPaper,
    _build_budgeted_text,
    _classify_section,
    _element_to_markdown,
    _get_all_text,
    _list_to_markdown,
    _paragraph_to_text,
    _table_to_markdown,
    extract_ref_list,
    parse_pmc_xml_to_text,
)


# ------------------------------------------------------------------ #
# parse_pmc_xml_to_text with real example XMLs
# ------------------------------------------------------------------ #

class TestParsePmcXmlToText:
    """Test full XML parsing against shipped example files."""

    def test_parses_pmc4894932(self, example_xml_4894932: Path):
        result = parse_pmc_xml_to_text(example_xml_4894932)

        assert result is not None
        assert isinstance(result, ParsedPaper)
        assert result.pmc_id == "PMC4894932"
        assert "Fluid shear stress" in result.title
        assert result.has_methods is True
        assert result.char_count > 0
        assert len(result.full_text) > 0
        assert len(result.sections) > 0

    def test_parses_pmc10033665(self, example_xml_10033665: Path):
        result = parse_pmc_xml_to_text(example_xml_10033665)

        assert result is not None
        assert result.pmc_id == "PMC10033665"
        assert "hemophilia" in result.title.lower() or "F8" in result.title
        assert result.has_methods is True
        assert result.tables_found >= 1

    def test_abstract_extracted(self, example_xml_4894932: Path):
        result = parse_pmc_xml_to_text(example_xml_4894932)

        assert result is not None
        assert len(result.abstract) > 100
        assert "hepatocyte" in result.abstract.lower()

    def test_sections_contain_methods(self, example_xml_4894932: Path):
        result = parse_pmc_xml_to_text(example_xml_4894932)
        assert result is not None

        # At least one section title should reference methods/materials
        section_titles_lower = [t.lower() for t in result.sections.keys()]
        assert any("method" in t or "material" in t for t in section_titles_lower)

    def test_full_text_includes_title(self, example_xml_4894932: Path):
        result = parse_pmc_xml_to_text(example_xml_4894932)
        assert result is not None
        assert result.title in result.full_text

    def test_full_text_within_budget(self, example_xml_4894932: Path):
        result = parse_pmc_xml_to_text(example_xml_4894932)
        assert result is not None
        assert result.char_count <= MAIN_TEXT_BUDGET

    def test_returns_none_for_nonexistent_file(self, tmp_path: Path):
        fake_path = tmp_path / "nonexistent.xml"
        try:
            result = parse_pmc_xml_to_text(fake_path)
            assert result is None
        except FileNotFoundError:
            pass

    def test_returns_none_for_malformed_xml(self, tmp_path: Path):
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("<not-closed>")
        result = parse_pmc_xml_to_text(bad_xml)
        assert result is None

    def test_returns_none_for_xml_without_article(self, tmp_path: Path):
        no_article = tmp_path / "no_article.xml"
        no_article.write_text("<root><data>test</data></root>")
        result = parse_pmc_xml_to_text(no_article)
        assert result is None

    def test_pmc_id_falls_back_to_filename(self, tmp_path: Path):
        """When there is no pmcid article-id, PMC ID should come from filename."""
        xml_content = """<article>
            <front><article-meta>
                <article-id pub-id-type="doi">10.1000/test</article-id>
                <title-group><article-title>Test Article</article-title></title-group>
            </article-meta></front>
            <body><sec><title>Methods</title><p>Test methods paragraph.</p></sec></body>
        </article>"""
        xml_path = tmp_path / "PMC7777777.xml"
        xml_path.write_text(xml_content)
        result = parse_pmc_xml_to_text(xml_path)
        assert result is not None
        assert result.pmc_id == "PMC7777777"


# ------------------------------------------------------------------ #
# Section classification
# ------------------------------------------------------------------ #

class TestClassifySection:

    @pytest.mark.parametrize("title,expected", [
        ("Materials and methods", "methods"),
        ("Experimental Procedures", "methods"),
        ("Cell Culture", "methods"),
        ("Results", "results"),
        ("Discussion", "discussion"),
        ("Introduction", "introduction"),
        ("Background", "introduction"),
        ("Supplementary Data", "supplementary"),
        ("Supporting Information", "supplementary"),
        ("Abstract", "abstract"),
        ("Acknowledgements", "results"),  # default fallback
    ])
    def test_section_classification(self, title: str, expected: str):
        assert _classify_section(title) == expected


# ------------------------------------------------------------------ #
# Text budget
# ------------------------------------------------------------------ #

class TestBuildBudgetedText:

    def test_fits_within_budget(self):
        sections = {
            "Methods": "M" * 5000,
            "Results": "R" * 5000,
            "Discussion": "D" * 5000,
            "Introduction": "I" * 5000,
        }
        text = _build_budgeted_text("Title", "Abstract text", sections, budget=10000)
        assert len(text) <= 10000

    def test_methods_preserved_over_discussion(self):
        """Methods (priority 0) should be kept; discussion (priority 5) may be trimmed."""
        methods_text = "METHODS_MARKER " * 500
        discussion_text = "DISCUSSION_MARKER " * 500
        sections = {
            "Methods": methods_text,
            "Discussion": discussion_text,
        }
        text = _build_budgeted_text("Title", "Abstract", sections, budget=6000)
        assert "METHODS_MARKER" in text
        # Discussion may be partially or fully cut
        # but methods should definitely be present

    def test_includes_title_and_abstract(self):
        sections = {"Methods": "Test content"}
        text = _build_budgeted_text("My Title", "My Abstract", sections, budget=50000)
        assert "# My Title" in text
        assert "## Abstract" in text
        assert "My Abstract" in text


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #

class TestGetAllText:

    def test_simple_element(self):
        elem = ET.fromstring("<p>Hello world</p>")
        assert _get_all_text(elem) == "Hello world"

    def test_nested_elements(self):
        elem = ET.fromstring("<p>Hello <b>bold</b> text</p>")
        text = _get_all_text(elem)
        assert "Hello" in text
        assert "bold" in text
        assert "text" in text


class TestParagraphToText:

    def test_plain_paragraph(self):
        p = ET.fromstring("<p>Simple paragraph text.</p>")
        assert _paragraph_to_text(p) == "Simple paragraph text."

    def test_bold_markup(self):
        p = ET.fromstring("<p>Some <bold>important</bold> text.</p>")
        result = _paragraph_to_text(p)
        assert "**important**" in result

    def test_italic_markup(self):
        p = ET.fromstring("<p>Some <italic>emphasized</italic> text.</p>")
        result = _paragraph_to_text(p)
        assert "*emphasized*" in result

    def test_superscript(self):
        p = ET.fromstring("<p>10<sup>5</sup> cells</p>")
        result = _paragraph_to_text(p)
        assert "^5" in result

    def test_xref_preserved(self):
        p = ET.fromstring('<p>See Fig. <xref ref-type="fig" rid="Fig1">1</xref>.</p>')
        result = _paragraph_to_text(p)
        assert "1" in result


class TestTableToMarkdown:

    def test_simple_table(self):
        xml = """<table>
            <tr><th>Gene</th><th>Expression</th></tr>
            <tr><td>ALB</td><td>High</td></tr>
            <tr><td>AFP</td><td>Low</td></tr>
        </table>"""
        table = ET.fromstring(xml)
        md = _table_to_markdown(table)
        assert "| Gene | Expression |" in md
        assert "| --- | --- |" in md
        assert "| ALB | High |" in md
        assert "| AFP | Low |" in md

    def test_empty_table(self):
        table = ET.fromstring("<table></table>")
        md = _table_to_markdown(table)
        assert md == ""

    def test_uneven_columns_normalized(self):
        xml = """<table>
            <tr><th>A</th><th>B</th><th>C</th></tr>
            <tr><td>1</td><td>2</td></tr>
        </table>"""
        table = ET.fromstring(xml)
        md = _table_to_markdown(table)
        lines = md.strip().split("\n")
        # All rows should have same number of pipes
        pipe_counts = [line.count("|") for line in lines]
        assert len(set(pipe_counts)) == 1

    def test_pipe_escaped(self):
        xml = """<table>
            <tr><th>Value</th></tr>
            <tr><td>A|B</td></tr>
        </table>"""
        table = ET.fromstring(xml)
        md = _table_to_markdown(table)
        assert "A\\|B" in md


class TestListToMarkdown:

    def test_bullet_list(self):
        xml = """<list>
            <list-item><p>First item</p></list-item>
            <list-item><p>Second item</p></list-item>
        </list>"""
        elem = ET.fromstring(xml)
        md = _list_to_markdown(elem)
        assert "- First item" in md
        assert "- Second item" in md


# ------------------------------------------------------------------ #
# Reference extraction
# ------------------------------------------------------------------ #

class TestExtractRefList:

    def test_extracts_references_from_pmc4894932(self, example_xml_4894932: Path):
        refs = extract_ref_list(example_xml_4894932)
        assert len(refs) > 0

        # All refs should have at least some identifiers
        for ref in refs:
            has_id = ref["doi"] or ref["pmid"] or ref["pmc_id"]
            has_author_year = ref["first_author_surname"] and ref["year"]
            assert has_id or has_author_year

    def test_reference_fields_populated(self, example_xml_4894932: Path):
        refs = extract_ref_list(example_xml_4894932)

        # Find a known reference (Bao & Suresh 2003, has DOI and PMID)
        bao_ref = [r for r in refs if r.get("first_author_surname") == "Bao"]
        assert len(bao_ref) >= 1
        bao = bao_ref[0]
        assert bao["doi"] == "10.1038/nmat1001"
        assert bao["pmid"] == "14593396"
        assert bao["year"] == "2003"

    def test_returns_empty_for_bad_xml(self, tmp_path: Path):
        bad = tmp_path / "bad.xml"
        bad.write_text("<broken>")
        refs = extract_ref_list(bad)
        assert refs == []

    def test_returns_empty_for_xml_without_refs(self, tmp_path: Path):
        no_refs = tmp_path / "no_refs.xml"
        no_refs.write_text("<article><front/><body/></article>")
        refs = extract_ref_list(no_refs)
        assert refs == []

    def test_pmc_id_extracted(self, example_xml_4894932: Path):
        refs = extract_ref_list(example_xml_4894932)
        # Some references in this paper have PMC IDs
        pmc_refs = [r for r in refs if r.get("pmc_id")]
        # At least a few should have PMC IDs (Cameron 2015 has PMC4682209)
        assert len(pmc_refs) >= 1
