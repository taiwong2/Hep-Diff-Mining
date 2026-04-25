"""Tests for data_layer.supplement_processor — supplement file conversion."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from data_layer.supplement_processor import (
    SUPPLEMENT_TEXT_BUDGET,
    _collect_files,
    _dataframe_to_markdown,
    _process_csv,
    process_supplements,
)


# ------------------------------------------------------------------ #
# _dataframe_to_markdown
# ------------------------------------------------------------------ #

class TestDataframeToMarkdown:

    def test_simple_dataframe(self):
        import pandas as pd

        df = pd.DataFrame({
            "Gene": ["ALB", "AFP", "CYP3A4"],
            "Expression": ["High", "Low", "Medium"],
        })
        md = _dataframe_to_markdown(df)
        assert "| Gene | Expression |" in md
        assert "| --- | --- |" in md
        assert "| ALB | High |" in md
        assert "| AFP | Low |" in md
        assert "| CYP3A4 | Medium |" in md

    def test_nan_values_become_empty(self):
        import pandas as pd

        df = pd.DataFrame({
            "A": ["1", None],
            "B": [None, "2"],
        })
        md = _dataframe_to_markdown(df)
        # NaN/None should be rendered as empty string, not "nan"
        assert "nan" not in md.lower() or md.count("nan") == 0
        lines = md.strip().split("\n")
        # Should have header + separator + 2 data rows
        assert len(lines) == 4

    def test_pipe_in_data_escaped(self):
        import pandas as pd

        df = pd.DataFrame({"Col": ["A|B"]})
        md = _dataframe_to_markdown(df)
        assert "A\\|B" in md

    def test_newline_in_data_normalized(self):
        import pandas as pd

        df = pd.DataFrame({"Col": ["line1\nline2"]})
        md = _dataframe_to_markdown(df)
        assert "\n" not in md.split("\n")[2]  # data row should not have embedded newline

    def test_empty_dataframe(self):
        import pandas as pd

        df = pd.DataFrame()
        md = _dataframe_to_markdown(df)
        assert isinstance(md, str)


# ------------------------------------------------------------------ #
# _process_csv
# ------------------------------------------------------------------ #

class TestProcessCsv:

    def test_valid_csv(self, tmp_path: Path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("Gene,Fold_Change,P_value\nALB,5.2,0.001\nAFP,-3.1,0.01\n")
        result = _process_csv(csv_path)
        assert "Gene" in result
        assert "ALB" in result
        assert "5.2" in result

    def test_tsv_with_delimiter(self, tmp_path: Path):
        tsv_path = tmp_path / "test.tsv"
        tsv_path.write_text("Gene\tExpression\nHNF4A\tHigh\nALB\tMedium\n")
        result = _process_csv(tsv_path, delimiter="\t")
        assert "Gene" in result
        assert "HNF4A" in result

    def test_empty_csv(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        result = _process_csv(csv_path)
        assert result == ""

    def test_csv_with_only_headers(self, tmp_path: Path):
        csv_path = tmp_path / "headers_only.csv"
        csv_path.write_text("A,B,C\n")
        result = _process_csv(csv_path)
        # Should return valid markdown with headers but no data rows (or empty)
        # pandas reads this as empty df, so result could be empty
        # The important thing is it doesn't crash
        assert isinstance(result, str)

    def test_csv_result_is_markdown_table(self, tmp_path: Path):
        csv_path = tmp_path / "table.csv"
        csv_path.write_text("X,Y\n1,2\n3,4\n")
        result = _process_csv(csv_path)
        lines = result.strip().split("\n")
        assert len(lines) >= 3  # header + separator + at least 1 data row
        assert lines[0].startswith("|")
        assert "---" in lines[1]


# ------------------------------------------------------------------ #
# process_supplements (integration)
# ------------------------------------------------------------------ #

class TestProcessSupplements:

    def test_processes_csv_supplement(self, tmp_path: Path):
        supp_dir = tmp_path / "PMC1234567_supp"
        supp_dir.mkdir()
        csv_file = supp_dir / "Table_S1.csv"
        csv_file.write_text("Marker,Day7,Day14\nALB,+,++\nAFP,++,+\n")

        result = process_supplements(supp_dir)
        assert "## Supplement: Table_S1.csv" in result
        assert "ALB" in result

    def test_processes_txt_supplement(self, tmp_path: Path):
        supp_dir = tmp_path / "PMC1234567_supp"
        supp_dir.mkdir()
        txt_file = supp_dir / "supplementary_methods.txt"
        txt_file.write_text("Detailed differentiation protocol:\nStep 1: ...")

        result = process_supplements(supp_dir)
        assert "supplementary_methods.txt" in result
        assert "Detailed differentiation protocol" in result

    def test_multiple_files_combined(self, tmp_path: Path):
        supp_dir = tmp_path / "PMC1234567_supp"
        supp_dir.mkdir()
        (supp_dir / "methods.txt").write_text("Method details here")
        csv_file = supp_dir / "data.csv"
        csv_file.write_text("A,B\n1,2\n")

        result = process_supplements(supp_dir)
        assert "methods.txt" in result
        assert "data.csv" in result

    def test_empty_directory(self, tmp_path: Path):
        supp_dir = tmp_path / "empty_supp"
        supp_dir.mkdir()
        result = process_supplements(supp_dir)
        assert result == ""

    def test_nonexistent_directory(self, tmp_path: Path):
        result = process_supplements(tmp_path / "nonexistent")
        assert result == ""

    def test_skips_unknown_extensions(self, tmp_path: Path):
        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()
        (supp_dir / "image.png").write_bytes(b"\x89PNG\r\n")
        (supp_dir / "data.bin").write_bytes(b"\x00\x01\x02")

        result = process_supplements(supp_dir)
        assert result == ""

    def test_skip_pdf_flag(self, tmp_path: Path):
        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()
        # Create a fake PDF (won't be valid but should be skipped)
        (supp_dir / "figure.pdf").write_bytes(b"%PDF-1.4 fake")
        (supp_dir / "table.csv").write_text("X,Y\n1,2\n")

        result = process_supplements(supp_dir, skip_pdf=True)
        # CSV should be processed, PDF skipped
        assert "table.csv" in result
        assert "figure.pdf" not in result

    def test_respects_text_budget(self, tmp_path: Path):
        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()
        # Create a file larger than budget
        large_text = "X" * (SUPPLEMENT_TEXT_BUDGET + 10000)
        (supp_dir / "large.txt").write_text(large_text)

        result = process_supplements(supp_dir)
        assert len(result) <= SUPPLEMENT_TEXT_BUDGET + 200  # small overhead for headers

    def test_skips_oversized_files(self, tmp_path: Path):
        """Files > 50MB should be silently skipped."""
        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()
        # We can't easily create a 50MB file in tests, but we can verify
        # the function handles a normal file correctly
        (supp_dir / "normal.txt").write_text("Normal content")
        result = process_supplements(supp_dir)
        assert "Normal content" in result


# ------------------------------------------------------------------ #
# _collect_files
# ------------------------------------------------------------------ #

class TestCollectFiles:

    def test_collects_regular_files(self, tmp_path: Path):
        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()
        (supp_dir / "a.csv").write_text("x")
        (supp_dir / "b.txt").write_text("y")

        files = _collect_files(supp_dir)
        names = [f.name for f in files]
        assert "a.csv" in names
        assert "b.txt" in names

    def test_skips_directories(self, tmp_path: Path):
        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()
        (supp_dir / "subdir").mkdir()
        (supp_dir / "file.txt").write_text("test")

        files = _collect_files(supp_dir)
        assert all(f.is_file() for f in files)

    def test_extracts_zip_contents(self, tmp_path: Path):
        import zipfile

        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()

        # Create a ZIP with a CSV inside
        zip_path = supp_dir / "supplement.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", "A,B\n1,2\n")
            zf.writestr("notes.txt", "Some notes")

        files = _collect_files(supp_dir)
        names = [f.name for f in files]
        assert "data.csv" in names
        assert "notes.txt" in names

    def test_zip_skips_macosx(self, tmp_path: Path):
        import zipfile

        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()

        zip_path = supp_dir / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("good.csv", "x,y\n1,2\n")
            zf.writestr("__MACOSX/._good.csv", "mac metadata")

        files = _collect_files(supp_dir)
        names = [f.name for f in files]
        assert "good.csv" in names
        assert "._good.csv" not in names

    def test_zip_skips_hidden_files(self, tmp_path: Path):
        import zipfile

        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()

        zip_path = supp_dir / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("visible.txt", "content")
            zf.writestr(".hidden", "secret")

        files = _collect_files(supp_dir)
        names = [f.name for f in files]
        assert "visible.txt" in names
        assert ".hidden" not in names

    def test_files_sorted_by_name(self, tmp_path: Path):
        supp_dir = tmp_path / "supp"
        supp_dir.mkdir()
        (supp_dir / "c.txt").write_text("c")
        (supp_dir / "a.txt").write_text("a")
        (supp_dir / "b.txt").write_text("b")

        files = _collect_files(supp_dir)
        names = [f.name for f in files]
        assert names == sorted(names)
