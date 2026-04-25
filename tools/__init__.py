"""Extraction tools for the agentic extractor.

Provides OpenAI-compatible function schemas and execute() functions
for search_corpus, fetch_reference, and flag_incomplete.
"""

from __future__ import annotations

from tools.search_corpus import TOOL_SCHEMA as SEARCH_CORPUS_SCHEMA
from tools.search_corpus import execute as execute_search_corpus
from tools.fetch_reference import TOOL_SCHEMA as FETCH_REFERENCE_SCHEMA
from tools.fetch_reference import execute as execute_fetch_reference
from tools.flag_incomplete import TOOL_SCHEMA as FLAG_INCOMPLETE_SCHEMA
from tools.flag_incomplete import execute as execute_flag_incomplete

# All tool schemas for the LLM tools parameter
ALL_TOOL_SCHEMAS = [
    SEARCH_CORPUS_SCHEMA,
    FETCH_REFERENCE_SCHEMA,
    FLAG_INCOMPLETE_SCHEMA,
]

# Dispatch map: function_name -> execute function
TOOL_DISPATCH = {
    "search_corpus": execute_search_corpus,
    "fetch_reference": execute_fetch_reference,
    "flag_incomplete": execute_flag_incomplete,
}
