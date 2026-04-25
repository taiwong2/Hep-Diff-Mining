"""Tests for tools/ — tool schema registry and individual tool execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import ALL_TOOL_SCHEMAS, TOOL_DISPATCH
from tools.flag_incomplete import (
    TOOL_SCHEMA as FLAG_INCOMPLETE_SCHEMA,
    VALID_REASONS,
    execute as execute_flag_incomplete,
)
from tools.search_corpus import (
    TOOL_SCHEMA as SEARCH_CORPUS_SCHEMA,
    execute as execute_search_corpus,
)


# ------------------------------------------------------------------ #
# Tool registry
# ------------------------------------------------------------------ #

class TestToolRegistry:

    def test_all_tool_schemas_has_three_entries(self):
        assert len(ALL_TOOL_SCHEMAS) == 3

    def test_tool_dispatch_has_three_entries(self):
        assert len(TOOL_DISPATCH) == 3

    def test_dispatch_keys_match_schema_names(self):
        schema_names = {s["function"]["name"] for s in ALL_TOOL_SCHEMAS}
        dispatch_names = set(TOOL_DISPATCH.keys())
        assert schema_names == dispatch_names

    def test_expected_tool_names(self):
        expected = {"search_corpus", "fetch_reference", "flag_incomplete"}
        schema_names = {s["function"]["name"] for s in ALL_TOOL_SCHEMAS}
        assert schema_names == expected

    def test_dispatch_values_are_callable(self):
        for name, func in TOOL_DISPATCH.items():
            assert callable(func), f"{name} dispatch is not callable"


# ------------------------------------------------------------------ #
# Tool schemas follow OpenAI function-calling format
# ------------------------------------------------------------------ #

class TestToolSchemaFormat:

    @pytest.mark.parametrize("schema", ALL_TOOL_SCHEMAS)
    def test_schema_has_type_function(self, schema):
        assert schema["type"] == "function"

    @pytest.mark.parametrize("schema", ALL_TOOL_SCHEMAS)
    def test_schema_has_function_name(self, schema):
        assert "name" in schema["function"]
        assert isinstance(schema["function"]["name"], str)
        assert len(schema["function"]["name"]) > 0

    @pytest.mark.parametrize("schema", ALL_TOOL_SCHEMAS)
    def test_schema_has_description(self, schema):
        assert "description" in schema["function"]
        assert len(schema["function"]["description"]) > 10

    @pytest.mark.parametrize("schema", ALL_TOOL_SCHEMAS)
    def test_schema_has_parameters(self, schema):
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params

    @pytest.mark.parametrize("schema", ALL_TOOL_SCHEMAS)
    def test_schema_has_required_fields(self, schema):
        params = schema["function"]["parameters"]
        assert "required" in params
        assert isinstance(params["required"], list)
        assert len(params["required"]) >= 1


# ------------------------------------------------------------------ #
# search_corpus tool
# ------------------------------------------------------------------ #

class TestSearchCorpusSchema:

    def test_requires_query(self):
        params = SEARCH_CORPUS_SCHEMA["function"]["parameters"]
        assert "query" in params["required"]
        assert params["properties"]["query"]["type"] == "string"


# ------------------------------------------------------------------ #
# flag_incomplete tool
# ------------------------------------------------------------------ #

class TestFlagIncomplete:

    def test_valid_reasons_defined(self):
        assert "not_reported" in VALID_REASONS
        assert "behind_paywall_reference" in VALID_REASONS
        assert "ambiguous_text" in VALID_REASONS
        assert len(VALID_REASONS) >= 5

    def test_schema_enum_matches_valid_reasons(self):
        schema_enum = set(
            FLAG_INCOMPLETE_SCHEMA["function"]["parameters"]["properties"]["reason"]["enum"]
        )
        assert schema_enum == VALID_REASONS

    def test_execute_valid_flag(self):
        result_json = execute_flag_incomplete(None, {
            "field": "de_stage_growth_factors",
            "reason": "not_reported",
            "details": "Paper does not specify growth factor concentrations",
        })
        result = json.loads(result_json)
        assert result["status"] == "flagged"
        assert result["flag"]["field"] == "de_stage_growth_factors"
        assert result["flag"]["reason"] == "not_reported"

    def test_execute_with_paywall_reason(self):
        result_json = execute_flag_incomplete(None, {
            "field": "base_protocol_details",
            "reason": "behind_paywall_reference",
            "details": "Protocol described in DOI 10.1000/paywall which is not OA",
        })
        result = json.loads(result_json)
        assert result["status"] == "flagged"
        assert "paywall" in result["flag"]["details"].lower()

    def test_execute_invalid_reason(self):
        result_json = execute_flag_incomplete(None, {
            "field": "test_field",
            "reason": "invalid_reason_xyz",
        })
        result = json.loads(result_json)
        assert "error" in result

    def test_execute_missing_field(self):
        result_json = execute_flag_incomplete(None, {
            "field": "",
            "reason": "not_reported",
        })
        result = json.loads(result_json)
        assert "error" in result

    def test_execute_without_details(self):
        result_json = execute_flag_incomplete(None, {
            "field": "seeding_density",
            "reason": "not_reported",
        })
        result = json.loads(result_json)
        assert result["status"] == "flagged"
        # Details should be empty string, not an error
        assert result["flag"]["details"] == ""


# ------------------------------------------------------------------ #
# search_corpus tool execution
# ------------------------------------------------------------------ #

class TestSearchCorpusExecution:

    def test_execute_empty_query(self):
        result_json = execute_search_corpus(None, {"query": ""})
        result = json.loads(result_json)
        assert "error" in result

    def test_execute_with_db(self, populated_db):
        """Search should work against a populated DB even with no protocols."""
        result_json = execute_search_corpus(populated_db, {"query": "shear stress"})
        result = json.loads(result_json)
        # No protocols stored, so results should be empty
        assert "results" in result
        assert len(result["results"]) == 0

    def test_execute_finds_protocol_by_doi(self, populated_db):
        """After storing a protocol, search by DOI should find it."""
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.store_protocol(paper["id"], {
            "protocol_arm": "Standard",
            "stages": [{"stage_name": "DE", "duration_days": 3}],
            "cell_source": {"type": "ESC", "line_name": "H9"},
            "extraction_confidence": 0.85,
        })

        result_json = execute_search_corpus(
            populated_db, {"query": "10.1007/s00204-016-1689-8"}
        )
        result = json.loads(result_json)
        assert len(result["results"]) >= 1
        assert result["results"][0]["doi"] == "10.1007/s00204-016-1689-8"

    def test_execute_finds_protocol_by_title(self, populated_db):
        paper = populated_db.get_paper(pmc_id="PMC4894932")
        populated_db.store_protocol(paper["id"], {
            "protocol_arm": "test",
            "stages": [],
        })

        result_json = execute_search_corpus(
            populated_db, {"query": "Fluid shear stress"}
        )
        result = json.loads(result_json)
        assert len(result["results"]) >= 1
