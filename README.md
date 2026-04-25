# Cell Differentiation Mining

A Python pipeline for mining hepatocyte differentiation protocols from PMC open-access literature. It discovers relevant papers via NCBI queries, classifies them with LLM-based triage, downloads and parses full-text XMLs and supplements, then extracts structured protocol data through a multi-pass LLM extraction system with tool-calling capabilities.

## Overview

This pipeline processes PubMed Central (PMC) open-access papers to build a structured database of hepatocyte differentiation protocols from iPSC, ESC, and direct reprogramming sources. The extraction system uses a three-pass architecture:

1. **Structure identification** — identifies protocol arms and stage structure
2. **Detailed extraction with tool calling** — extracts per-arm protocol records (cell source, growth factors, small molecules, duration, markers) with the ability to search previously extracted protocols and fetch referenced papers
3. **Supplement enrichment** — augments protocols with data from supplementary figures and tables

## System Requirements

- **Operating system:** Linux, macOS, or Windows (WSL)
- **Python:** 3.10 or later
- **Disk space:** ~2 GB for PMC XMLs and processed data
- **RAM:** 4 GB minimum; 8 GB recommended if processing PDF supplements
- **External tools:** [NCBI EDirect](https://www.ncbi.nlm.nih.gov/books/NBK179288/) command-line utilities (for paper discovery)
- **API keys:** NCBI API key(s) and an [OpenRouter](https://openrouter.ai/) API key (for LLM access)

## Installation

**Typical install time: 2–5 minutes** (excluding PDF processing dependencies).

```bash
# Clone the repository
git clone https://github.com/taiwong2/CellDifferentiationMining.git
cd CellDifferentiationMining

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install core dependencies
pip install -e .

# Optional: PDF supplement processing (requires ~1 GB for model weights)
pip install -e ".[pdf]"

# Optional: Legacy document format support (.doc, .ppt)
pip install -e ".[legacy-docs]"

# Optional: Development/testing dependencies
pip install -e ".[dev]"

# Or install everything
pip install -e ".[all]"
```

### NCBI EDirect Installation

The pipeline uses NCBI EDirect for paper discovery. Install it separately:

```bash
sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"
export PATH="${HOME}/edirect:${PATH}"
```

### Environment Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your NCBI and OpenRouter API keys
```

## Demo

**Expected run time: <30 seconds.**

The demo demonstrates the data processing stages (XML parsing, database schema, reference extraction) using included example papers, without requiring API keys:

```bash
python demo.py
```

This parses two example PMC XML files into structured markdown, displays the database schema, and extracts reference metadata — showing the core data processing pipeline.

## Usage

### Full Pipeline

```bash
# Run the complete 16-step pipeline
python run_pipeline.py

# Resume from a specific stage
python run_pipeline.py --from-step 2

# Run only one stage
python run_pipeline.py --only-step 6

# Limit extraction to N papers (for testing)
python run_pipeline.py --limit 5

# Skip PMC XML re-fetch (use cached data)
python run_pipeline.py --skip-fetch

# Skip PDF OCR in supplement processing
python run_pipeline.py --skip-pdf
```

### Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 0 | Paper Discovery | Queries PMC via NCBI EDirect for iPSC/ESC hepatocyte differentiation papers |
| 1 | Database Bootstrap | Imports triage classification results into SQLite |
| 2 | XML to Markdown | Converts PMC XML to structured markdown with section hierarchy |
| 3 | Supplement Fetch | Downloads supplement archives from PMC Open Access FTP |
| 4 | Supplement Processing | Converts PDF/Word/Excel/CSV supplements to text |
| 5 | Reference Graph | Builds citation DAG and sets topological processing order |
| 6 | GEO Discovery | Text mining + elink + SOFT validation for GEO accessions |
| 6.5 | Accession Grounding | Verify and validate GEO accessions against paper text |
| 7 | Protocol Extraction | Multi-pass LLM extraction with tool calling |
| 8 | Grounding Cleanup | Validate extracted terms against source text |
| 9 | GEO Sample Mapping | Map GEO samples to protocol stages |
| 10 | Statistics | Reports extraction coverage and quality metrics |
| 11 | RNA-seq Extraction | LLM-based RNA-seq metadata extraction |
| 12 | Repository Cross-Ref | Cross-reference accessions with ENA/SRA/GEO APIs |
| 13 | Expression Retrieval | Retrieve expression data from GEO matrices |
| 14 | Expression Integration | Build protocol x gene expression matrix |
| 15 | Export | Produce final multi-sheet Excel workbook |

### Utility Scripts

```bash
# Behind-paywall reference ingestion
python -m scripts.manual_fetch show
python -m scripts.manual_fetch ingest
python -m scripts.manual_fetch re-extract

# Standalone supplement enrichment (Pass 3)
python -m scripts.supplement_enrich

# Stratified quality audit
python -m scripts.audit
```

## Project Structure

```
run_pipeline.py                    — 16-step pipeline orchestrator
demo.py                            — Demo script for journal reviewers

steps/                             — Pipeline step implementations
  ground_accessions.py             — Step 6.5: accession grounding
  grounding_cleanup.py             — Step 8: term validation
  rnaseq_extract.py                — Step 11: RNA-seq metadata extraction
  rnaseq_crossref.py               — Step 12: repository cross-referencing
  rnaseq_retrieve.py               — Step 13: expression data retrieval
  rnaseq_integrate.py              — Step 14: expression integration
  export_results.py                — Step 15: final Excel export

data_layer/                        — Data access and processing
  database.py                      — SQLite schema + access layer
  xml_to_text.py                   — PMC XML → markdown conversion
  fetch_supplements.py             — PMC OA FTP downloader
  supplement_processor.py          — PDF/Word/Excel/CSV/PPTX → text
  reference_graph.py               — Citation DAG + topological sort
  grounding.py                     — Post-extraction term grounding
  geo_linker.py                    — GEO accession discovery
  geo_sample_mapper.py             — GEO sample-to-stage mapping
  geo_matrix_fetcher.py            — GEO matrix data retrieval
  expression_integrator.py         — Cross-study expression integration
  ena_client.py                    — ENA/SRA API client

llm/                               — LLM agents and prompts
  openrouter/client.py             — Async OpenRouter API client
  agents/
    triage_classifier.py           — 6-category abstract classifier
    agentic_extractor.py           — 3-pass protocol extraction
    review_extractor.py            — Single-pass review extraction
    prompts/                       — System prompts for each pass

tools/                             — Tool-calling for LLM agents
  search_corpus.py                 — Search extracted protocols
  fetch_reference.py               — Fetch open-access paper text
  flag_incomplete.py               — Flag unresolvable fields

scripts/                           — Standalone utility workflows
  audit.py                         — Stratified quality audit
  manual_fetch.py                  — Behind-paywall PDF ingestion
  supplement_enrich.py             — Standalone Pass 3 enrichment

examples/                          — Example PMC XML files for demo
tests/                             — Unit tests (147 tests)
```

## Database Schema

SQLite database with six tables:

- **`papers`** — Paper metadata, file paths, and extraction status
- **`triage_results`** — LLM classification with confidence scores
- **`protocols`** — Extracted protocol records (cell source, stages, growth factors, markers, endpoints)
- **`paper_references`** — Citation graph edges
- **`corpus_cache`** — Cached full text of referenced papers
- **`processing_log`** — Audit trail of pipeline operations

## Output Format

Extracted protocols are stored as JSON records in `data/results/extraction_results.jsonl`. Each protocol record contains:

- **Cell source:** iPSC/ESC line, species, disease context
- **Culture system:** 2D/3D, coating, media base
- **Stages:** Ordered array of differentiation stages, each with growth factors, small molecules, duration, and stage-specific markers
- **Endpoint assessment:** Functional assays, marker expression, efficiency metrics
- **Confidence score:** Model self-assessed extraction confidence (0–1)
- **Provenance:** Source paper, protocol arm, base protocol references

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| LLM model | `openai/gpt-4o-mini` via OpenRouter | `llm/openrouter/client.py` |
| Temperature | 0 (deterministic) | Extraction agents |
| Main text budget | 120,000 characters | `data_layer/xml_to_text.py` |
| Supplement text budget | 40,000 characters | `data_layer/supplement_processor.py` |
| Max tool calls per paper | 5 | `llm/agents/agentic_extractor.py` |
| Max conversation turns | 10 | `llm/agents/agentic_extractor.py` |
| Concurrency | 10 simultaneous requests | `llm/openrouter/client.py` |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{wong2026celldiff,
  author = {Wong, Tai and Mattis, Aras},
  title = {Cell Differentiation Mining: LLM-Based Protocol Extraction from Biomedical Literature},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

<!-- TODO: Replace 10.5281/zenodo.XXXXXXX with your actual Zenodo DOI after deposit -->

The extracted protocol dataset is available at:

```bibtex
@dataset{wong2026celldiff_data,
  author = {Wong, Tai and Mattis, Aras},
  title = {Hepatocyte Differentiation Protocol Database},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.YYYYYYY},
  url = {https://doi.org/10.5281/zenodo.YYYYYYY}
}
```

<!-- TODO: Replace 10.5281/zenodo.YYYYYYY with your actual data deposit DOI -->
