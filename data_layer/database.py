"""SQLite database for the hepatocyte differentiation mining pipeline.

Stores papers, triage results, extracted protocols, reference graph,
corpus cache, and processing logs. All JSON columns stored as TEXT
and queryable via sqlite3 json_extract().

Usage:
    db = PipelineDB()                         # opens data/db/protocols.db
    db.import_from_triage_jsonl(path)          # bootstrap from triage output
    papers = db.get_papers_for_extraction()    # get papers ready for extraction
    db.store_protocol(paper_id, protocol)      # save extracted protocol
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "db" / "protocols.db"
DATA_ROOT = Path(__file__).parent.parent / "data" / "db"

# Categories that proceed to extraction
EXTRACTABLE_CATEGORIES = {"primary_protocol", "disease_model", "methods_tool"}
REVIEW_CATEGORY = "review"


class PipelineDB:
    """Synchronous SQLite database for the pipeline."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any):
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                pmc_id          TEXT UNIQUE,
                doi             TEXT,
                pmid            TEXT,
                title           TEXT,
                abstract        TEXT,
                article_type    TEXT,
                triage_category TEXT,
                xml_path        TEXT,
                parsed_text_path TEXT,
                supplement_dir  TEXT,
                supplement_text_path TEXT,
                extraction_status TEXT DEFAULT 'pending',
                processing_priority INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS triage_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id        INTEGER NOT NULL REFERENCES papers(id),
                category        TEXT NOT NULL,
                confidence      REAL,
                reasoning       TEXT,
                base_protocols  TEXT,  -- JSON array
                key_cell_types  TEXT,  -- JSON array
                disease_context TEXT,
                supplement_signal INTEGER DEFAULT 0,
                UNIQUE(paper_id)
            );

            CREATE TABLE IF NOT EXISTS paper_references (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id        INTEGER NOT NULL REFERENCES papers(id),
                referenced_doi  TEXT,
                referenced_pmc_id TEXT,
                referenced_id   INTEGER REFERENCES papers(id),
                in_corpus       INTEGER DEFAULT 0,
                resolved        INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS protocols (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id              INTEGER NOT NULL REFERENCES papers(id),
                protocol_arm          TEXT,
                is_optimized          INTEGER DEFAULT 0,
                cell_source           TEXT,  -- JSON
                culture_system        TEXT,  -- JSON
                stages                TEXT,  -- JSON array of StageRecord
                endpoint_assessment   TEXT,  -- JSON
                modifications         TEXT,  -- JSON
                step_sources          TEXT,  -- JSON per-step provenance
                base_protocol_doi     TEXT,
                extraction_confidence REAL,
                extraction_notes      TEXT,
                incomplete_flags      TEXT,  -- JSON array
                pass_number           INTEGER DEFAULT 2,
                created_at            TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS corpus_cache (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doi       TEXT,
                pmc_id    TEXT,
                title     TEXT,
                full_text TEXT,
                source    TEXT,
                cached_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_corpus_doi ON corpus_cache(doi);
            CREATE INDEX IF NOT EXISTS idx_corpus_pmc ON corpus_cache(pmc_id);

            CREATE TABLE IF NOT EXISTS processing_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id      INTEGER REFERENCES papers(id),
                stage         TEXT NOT NULL,
                status        TEXT NOT NULL,
                error_message TEXT,
                tokens_used   INTEGER,
                duration_secs REAL,
                created_at    TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_papers_pmc ON papers(pmc_id);
            CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
            CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(extraction_status);
            CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(triage_category);

            -- GEO accession discovery: paper → GEO series link
            CREATE TABLE IF NOT EXISTS geo_accessions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id          INTEGER NOT NULL REFERENCES papers(id),
                gse_id            TEXT NOT NULL,
                context           TEXT,          -- own_data|referenced|ambiguous
                confidence        REAL,
                discovery_strategies TEXT,        -- JSON array of strategy names
                data_type         TEXT,           -- Bulk RNA-seq|scRNA-seq|ATAC-seq|microarray
                platform          TEXT,
                sample_count      INTEGER,
                series_title      TEXT,
                series_summary    TEXT,
                linked_pmids      TEXT,           -- JSON array
                submission_date   TEXT,
                soft_fetched      INTEGER DEFAULT 0,
                created_at        TEXT DEFAULT (datetime('now')),
                UNIQUE(paper_id, gse_id)
            );
            CREATE INDEX IF NOT EXISTS idx_geo_acc_paper ON geo_accessions(paper_id);
            CREATE INDEX IF NOT EXISTS idx_geo_acc_gse ON geo_accessions(gse_id);

            -- Per-sample metadata from GEO SOFT
            CREATE TABLE IF NOT EXISTS geo_samples (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                geo_accession_id    INTEGER NOT NULL REFERENCES geo_accessions(id),
                gsm_id              TEXT NOT NULL,
                sample_title        TEXT,
                source_name         TEXT,
                description         TEXT,
                characteristics     TEXT,          -- JSON dict
                sra_accession       TEXT,
                UNIQUE(geo_accession_id, gsm_id)
            );
            CREATE INDEX IF NOT EXISTS idx_geo_samples_acc ON geo_samples(geo_accession_id);

            -- Core mapping: GEO sample ↔ protocol stage/day
            CREATE TABLE IF NOT EXISTS geo_sample_stage_mappings (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                geo_sample_id       INTEGER NOT NULL REFERENCES geo_samples(id),
                protocol_id         INTEGER NOT NULL REFERENCES protocols(id),
                stage_name          TEXT,
                stage_number        INTEGER,
                time_point_day      INTEGER,
                condition_label     TEXT,
                mapping_confidence  REAL,
                mapping_method      TEXT,          -- tier1_regex|tier2_llm
                UNIQUE(geo_sample_id, protocol_id)
            );
            CREATE INDEX IF NOT EXISTS idx_geo_map_sample ON geo_sample_stage_mappings(geo_sample_id);
            CREATE INDEX IF NOT EXISTS idx_geo_map_proto ON geo_sample_stage_mappings(protocol_id);

            -- RNA-seq metadata extracted via LLM (Phase 1)
            CREATE TABLE IF NOT EXISTS rnaseq_metadata (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id            INTEGER NOT NULL REFERENCES papers(id),
                has_rnaseq          INTEGER DEFAULT 0,
                rnaseq_type         TEXT,
                technology          TEXT,
                library_prep        TEXT,
                read_type           TEXT,
                read_length_bp      INTEGER,
                sequencing_depth    TEXT,
                genome_build        TEXT,
                annotation          TEXT,
                alignment_tool      TEXT,
                quantification_tool TEXT,
                normalization       TEXT,
                de_method           TEXT,
                accessions          TEXT,       -- JSON array
                deg_summary         TEXT,       -- JSON
                pathway_analysis    TEXT,       -- JSON
                data_availability   TEXT,       -- JSON
                extraction_notes    TEXT,
                created_at          TEXT DEFAULT (datetime('now')),
                UNIQUE(paper_id)
            );

            -- Repository metadata from external APIs (Phase 2)
            CREATE TABLE IF NOT EXISTS repository_metadata (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id            INTEGER NOT NULL REFERENCES papers(id),
                accession           TEXT NOT NULL,
                repository          TEXT NOT NULL,
                project_title       TEXT,
                organism            TEXT,
                data_type           TEXT,
                platform            TEXT,
                sample_count        INTEGER,
                has_processed_matrix INTEGER DEFAULT 0,
                supplementary_files TEXT,       -- JSON array
                sample_metadata     TEXT,       -- JSON array
                fetch_status        TEXT DEFAULT 'pending',
                raw_response        TEXT,
                created_at          TEXT DEFAULT (datetime('now')),
                UNIQUE(paper_id, accession)
            );

            -- Per-gene expression values (Phase 3)
            CREATE TABLE IF NOT EXISTS expression_values (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id            INTEGER NOT NULL REFERENCES papers(id),
                protocol_id         INTEGER REFERENCES protocols(id),
                gene_symbol         TEXT NOT NULL,
                gene_alias          TEXT,
                value               REAL,
                unit                TEXT,
                condition_label     TEXT,
                time_point_day      INTEGER,
                comparison          TEXT,
                padj                REAL,
                source_type         TEXT NOT NULL,
                source_detail       TEXT,
                confidence          REAL DEFAULT 0.7,
                created_at          TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_expr_paper ON expression_values(paper_id);
            CREATE INDEX IF NOT EXISTS idx_expr_gene ON expression_values(gene_symbol);
            CREATE INDEX IF NOT EXISTS idx_expr_proto ON expression_values(protocol_id);
        """)
        self._conn.commit()
        self._migrate_geo_status()
        self._migrate_rnaseq_status()
        self._migrate_geo_organism()

    def _migrate_geo_status(self) -> None:
        """Add geo_status column to papers if it doesn't exist yet."""
        cur = self._conn.cursor()
        cols = {row[1] for row in cur.execute("PRAGMA table_info(papers)").fetchall()}
        if "geo_status" not in cols:
            cur.execute("ALTER TABLE papers ADD COLUMN geo_status TEXT DEFAULT NULL")
            self._conn.commit()

    def _migrate_rnaseq_status(self) -> None:
        """Add rnaseq_status column to papers if it doesn't exist yet."""
        cur = self._conn.cursor()
        cols = {row[1] for row in cur.execute("PRAGMA table_info(papers)").fetchall()}
        if "rnaseq_status" not in cols:
            cur.execute("ALTER TABLE papers ADD COLUMN rnaseq_status TEXT DEFAULT NULL")
            self._conn.commit()

    def _migrate_geo_organism(self) -> None:
        """Add organism column to geo_accessions if it doesn't exist yet."""
        cur = self._conn.cursor()
        cols = {row[1] for row in cur.execute("PRAGMA table_info(geo_accessions)").fetchall()}
        if "organism" not in cols:
            cur.execute("ALTER TABLE geo_accessions ADD COLUMN organism TEXT DEFAULT NULL")
            self._conn.commit()

    # ------------------------------------------------------------------
    # Import from triage JSONL
    # ------------------------------------------------------------------

    def import_from_triage_jsonl(self, jsonl_path: str | Path) -> int:
        """Bootstrap papers and triage_results from triage_results.jsonl.

        Returns the number of papers imported.
        """
        jsonl_path = Path(jsonl_path)
        imported = 0
        cur = self._conn.cursor()

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                pmc_id = rec.get("pmc_id")
                if not pmc_id:
                    continue

                # Check if already imported
                existing = cur.execute(
                    "SELECT id FROM papers WHERE pmc_id = ?", (pmc_id,)
                ).fetchone()
                if existing:
                    continue

                # Find XML path
                xml_path = self._find_xml_path(pmc_id)

                # Insert paper
                cur.execute(
                    """INSERT INTO papers
                       (pmc_id, doi, pmid, title, triage_category, xml_path)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        pmc_id,
                        rec.get("doi"),
                        rec.get("pmid"),
                        rec.get("title"),
                        rec.get("category"),
                        xml_path,
                    ),
                )
                paper_id = cur.lastrowid

                # Insert triage result
                category = rec.get("category")
                if category:
                    cur.execute(
                        """INSERT INTO triage_results
                           (paper_id, category, confidence, reasoning,
                            base_protocols, key_cell_types, disease_context,
                            supplement_signal)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            paper_id,
                            category,
                            rec.get("confidence"),
                            rec.get("reasoning"),
                            json.dumps(rec.get("base_protocols", [])),
                            json.dumps(rec.get("key_cell_types", [])),
                            rec.get("disease_context"),
                            1 if rec.get("supplement_signal") else 0,
                        ),
                    )

                imported += 1

        self._conn.commit()
        logger.info("Imported %d papers from %s", imported, jsonl_path)
        return imported

    @staticmethod
    def _find_xml_path(pmc_id: str) -> str | None:
        """Resolve the sharded XML path for a PMC ID."""
        tag = pmc_id if pmc_id.startswith("PMC") else f"PMC{pmc_id}"
        prefix = tag[:5]
        path = DATA_ROOT / prefix / f"{tag}.xml"
        if path.exists():
            return str(path)
        return None

    # ------------------------------------------------------------------
    # Paper lookups
    # ------------------------------------------------------------------

    def get_paper(
        self,
        pmc_id: str | None = None,
        doi: str | None = None,
        paper_id: int | None = None,
    ) -> dict | None:
        """Look up a paper by any identifier. Returns dict or None."""
        cur = self._conn.cursor()
        if paper_id is not None:
            row = cur.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        elif pmc_id is not None:
            row = cur.execute("SELECT * FROM papers WHERE pmc_id = ?", (pmc_id,)).fetchone()
        elif doi is not None:
            row = cur.execute("SELECT * FROM papers WHERE doi = ?", (doi,)).fetchone()
        else:
            return None
        return dict(row) if row else None

    def get_papers_needing_text(self) -> list[dict]:
        """Papers that passed triage but don't have parsed text yet."""
        rows = self._conn.execute(
            """SELECT p.* FROM papers p
               WHERE p.triage_category IN (?, ?, ?, ?)
               AND p.parsed_text_path IS NULL
               AND p.xml_path IS NOT NULL
               ORDER BY p.id""",
            (*EXTRACTABLE_CATEGORIES, REVIEW_CATEGORY),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_papers_for_extraction(self, category: str | None = None) -> list[dict]:
        """Papers ready for extraction, ordered by processing_priority."""
        if category:
            rows = self._conn.execute(
                """SELECT p.* FROM papers p
                   WHERE p.triage_category = ?
                   AND p.extraction_status = 'pending'
                   AND p.parsed_text_path IS NOT NULL
                   ORDER BY p.processing_priority ASC, p.id ASC""",
                (category,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT p.* FROM papers p
                   WHERE p.triage_category IN (?, ?, ?)
                   AND p.extraction_status = 'pending'
                   AND p.parsed_text_path IS NOT NULL
                   ORDER BY p.processing_priority ASC, p.id ASC""",
                tuple(EXTRACTABLE_CATEGORIES),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_review_papers_for_extraction(self) -> list[dict]:
        """Review papers ready for lighter extraction."""
        return self.get_papers_for_extraction(category=REVIEW_CATEGORY)

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def update_paper(self, paper_id: int, **fields: Any) -> None:
        """Update arbitrary fields on a paper record."""
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [paper_id]
        self._conn.execute(
            f"UPDATE papers SET {set_clause} WHERE id = ?", values
        )
        self._conn.commit()

    def set_extraction_status(self, paper_id: int, status: str) -> None:
        self.update_paper(paper_id, extraction_status=status)

    # ------------------------------------------------------------------
    # Protocols
    # ------------------------------------------------------------------

    def store_protocol(self, paper_id: int, protocol: dict) -> int:
        """Insert an extracted protocol. Returns the new protocol ID."""
        cur = self._conn.execute(
            """INSERT INTO protocols
               (paper_id, protocol_arm, is_optimized, cell_source,
                culture_system, stages, endpoint_assessment,
                modifications, step_sources, base_protocol_doi,
                extraction_confidence, extraction_notes, incomplete_flags,
                pass_number)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper_id,
                protocol.get("protocol_arm"),
                1 if protocol.get("is_optimized") else 0,
                json.dumps(protocol.get("cell_source")),
                json.dumps(protocol.get("culture_system")),
                json.dumps(protocol.get("stages", [])),
                json.dumps(protocol.get("endpoint_assessment")),
                json.dumps(protocol.get("modifications")),
                json.dumps(protocol.get("step_sources")),
                protocol.get("base_protocol_doi"),
                protocol.get("extraction_confidence"),
                protocol.get("extraction_notes"),
                json.dumps(protocol.get("incomplete_flags", [])),
                protocol.get("pass_number", 2),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_protocol(self, protocol_id: int, updates: dict) -> None:
        """Update fields on an existing protocol (e.g. after Pass 3)."""
        json_fields = {
            "cell_source", "culture_system", "stages", "endpoint_assessment",
            "modifications", "step_sources", "incomplete_flags",
        }
        set_parts = []
        values = []
        for k, v in updates.items():
            set_parts.append(f"{k} = ?")
            values.append(json.dumps(v) if k in json_fields else v)
        values.append(protocol_id)
        self._conn.execute(
            f"UPDATE protocols SET {', '.join(set_parts)} WHERE id = ?", values
        )
        self._conn.commit()

    def delete_protocols_for_paper(self, paper_id: int) -> int:
        """Delete all protocols for a paper (for re-extraction). Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM protocols WHERE paper_id = ?", (paper_id,)
        )
        self._conn.commit()
        return cur.rowcount

    def get_protocols_for_paper(self, paper_id: int) -> list[dict]:
        """Get all extracted protocols for a paper."""
        rows = self._conn.execute(
            "SELECT * FROM protocols WHERE paper_id = ?", (paper_id,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for k in ("cell_source", "culture_system", "stages",
                       "endpoint_assessment", "modifications", "step_sources",
                       "incomplete_flags"):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def add_reference(self, paper_id: int, doi: str | None = None,
                      pmc_id: str | None = None) -> None:
        """Add a citation edge from paper_id to a referenced paper."""
        # Check if the referenced paper is in corpus
        ref_id = None
        in_corpus = 0
        if doi:
            row = self._conn.execute(
                "SELECT id FROM papers WHERE doi = ?", (doi,)
            ).fetchone()
            if row:
                ref_id = row["id"]
                in_corpus = 1
        if not ref_id and pmc_id:
            row = self._conn.execute(
                "SELECT id FROM papers WHERE pmc_id = ?", (pmc_id,)
            ).fetchone()
            if row:
                ref_id = row["id"]
                in_corpus = 1

        self._conn.execute(
            """INSERT OR IGNORE INTO paper_references
               (paper_id, referenced_doi, referenced_pmc_id,
                referenced_id, in_corpus, resolved)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (paper_id, doi, pmc_id, ref_id, in_corpus, 1 if in_corpus else 0),
        )
        self._conn.commit()

    def get_references(self, paper_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM paper_references WHERE paper_id = ?", (paper_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Corpus cache (for fetch_reference tool)
    # ------------------------------------------------------------------

    def cache_text(self, doi: str | None, pmc_id: str | None,
                   title: str | None, text: str, source: str) -> None:
        """Cache fetched reference text for the fetch_reference tool.

        If a cache entry with the same DOI already exists, only replace it
        when the new text is longer (better quality).
        """
        if doi:
            existing = self._conn.execute(
                "SELECT id, length(full_text) as tlen FROM corpus_cache WHERE doi = ?",
                (doi,),
            ).fetchone()
            if existing:
                if len(text) > (existing["tlen"] or 0):
                    self._conn.execute(
                        """UPDATE corpus_cache
                           SET full_text = ?, source = ?, title = COALESCE(?, title),
                               pmc_id = COALESCE(?, pmc_id)
                           WHERE id = ?""",
                        (text, source, title, pmc_id, existing["id"]),
                    )
                    self._conn.commit()
                return

        self._conn.execute(
            """INSERT INTO corpus_cache (doi, pmc_id, title, full_text, source)
               VALUES (?, ?, ?, ?, ?)""",
            (doi, pmc_id, title, text, source),
        )
        self._conn.commit()

    def get_cached_text(self, doi: str | None = None,
                        pmc_id: str | None = None) -> dict | None:
        """Look up cached text by DOI or PMC ID.  Prefers longest text."""
        if doi:
            row = self._conn.execute(
                "SELECT * FROM corpus_cache WHERE doi = ? ORDER BY length(full_text) DESC",
                (doi,),
            ).fetchone()
            if row:
                return dict(row)
        if pmc_id:
            row = self._conn.execute(
                "SELECT * FROM corpus_cache WHERE pmc_id = ? ORDER BY length(full_text) DESC",
                (pmc_id,),
            ).fetchone()
            if row:
                return dict(row)
        return None

    # ------------------------------------------------------------------
    # Search corpus (for search_corpus tool)
    # ------------------------------------------------------------------

    def search_corpus(self, query: str, limit: int = 5) -> list[dict]:
        """Search extracted protocols by DOI exact match or title LIKE.

        Returns list of dicts with paper + protocol summary info.
        """
        results = []

        # Try DOI exact match first
        if query.startswith("10.") or "doi.org" in query:
            doi = query.replace("https://doi.org/", "").replace("http://doi.org/", "")
            rows = self._conn.execute(
                """SELECT p.id AS paper_id, p.pmc_id, p.doi, p.title,
                          pr.protocol_arm, pr.stages, pr.cell_source,
                          pr.extraction_confidence
                   FROM papers p
                   JOIN protocols pr ON pr.paper_id = p.id
                   WHERE p.doi = ?
                   LIMIT ?""",
                (doi, limit),
            ).fetchall()
            results.extend(dict(r) for r in rows)

        # Title search
        if len(results) < limit:
            remaining = limit - len(results)
            seen_ids = {r["paper_id"] for r in results}
            like_query = f"%{query}%"
            rows = self._conn.execute(
                """SELECT p.id AS paper_id, p.pmc_id, p.doi, p.title,
                          pr.protocol_arm, pr.stages, pr.cell_source,
                          pr.extraction_confidence
                   FROM papers p
                   JOIN protocols pr ON pr.paper_id = p.id
                   WHERE p.title LIKE ?
                   LIMIT ?""",
                (like_query, remaining + len(seen_ids)),
            ).fetchall()
            for r in rows:
                d = dict(r)
                if d["paper_id"] not in seen_ids:
                    results.append(d)
                    if len(results) >= limit:
                        break

        # Deserialize JSON fields in results
        for r in results:
            for k in ("stages", "cell_source"):
                if r.get(k) and isinstance(r[k], str):
                    try:
                        r[k] = json.loads(r[k])
                    except (json.JSONDecodeError, TypeError):
                        pass
        return results

    # ------------------------------------------------------------------
    # Processing log
    # ------------------------------------------------------------------

    def log_processing(self, paper_id: int, stage: str, status: str,
                       error_message: str | None = None,
                       tokens_used: int | None = None,
                       duration_secs: float | None = None) -> None:
        self._conn.execute(
            """INSERT INTO processing_log
               (paper_id, stage, status, error_message, tokens_used, duration_secs)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (paper_id, stage, status, error_message, tokens_used, duration_secs),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # GEO accessions
    # ------------------------------------------------------------------

    def store_geo_accession(self, paper_id: int, accession: dict) -> int:
        """Insert or update a GEO accession for a paper. Returns the row ID."""
        cur = self._conn.execute(
            """INSERT INTO geo_accessions
               (paper_id, gse_id, context, confidence, discovery_strategies,
                data_type, platform, sample_count, series_title, series_summary,
                linked_pmids, submission_date, soft_fetched, organism)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(paper_id, gse_id) DO UPDATE SET
                   context = COALESCE(excluded.context, context),
                   confidence = MAX(COALESCE(excluded.confidence, 0), COALESCE(confidence, 0)),
                   discovery_strategies = excluded.discovery_strategies,
                   data_type = COALESCE(excluded.data_type, data_type),
                   platform = COALESCE(excluded.platform, platform),
                   sample_count = COALESCE(excluded.sample_count, sample_count),
                   series_title = COALESCE(excluded.series_title, series_title),
                   series_summary = COALESCE(excluded.series_summary, series_summary),
                   linked_pmids = COALESCE(excluded.linked_pmids, linked_pmids),
                   submission_date = COALESCE(excluded.submission_date, submission_date),
                   soft_fetched = MAX(excluded.soft_fetched, soft_fetched),
                   organism = COALESCE(excluded.organism, organism)
            """,
            (
                paper_id,
                accession["gse_id"],
                accession.get("context"),
                accession.get("confidence"),
                json.dumps(accession.get("discovery_strategies", [])),
                accession.get("data_type"),
                accession.get("platform"),
                accession.get("sample_count"),
                accession.get("series_title"),
                accession.get("series_summary"),
                json.dumps(accession.get("linked_pmids", [])),
                accession.get("submission_date"),
                1 if accession.get("soft_fetched") else 0,
                accession.get("organism"),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_geo_accessions(self, paper_id: int) -> list[dict]:
        """Get all GEO accessions for a paper."""
        rows = self._conn.execute(
            "SELECT * FROM geo_accessions WHERE paper_id = ?", (paper_id,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for k in ("discovery_strategies", "linked_pmids"):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result

    def store_geo_sample(self, geo_accession_id: int, sample: dict) -> int:
        """Insert or update a GEO sample. Returns the row ID."""
        cur = self._conn.execute(
            """INSERT INTO geo_samples
               (geo_accession_id, gsm_id, sample_title, source_name,
                description, characteristics, sra_accession)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(geo_accession_id, gsm_id) DO UPDATE SET
                   sample_title = COALESCE(excluded.sample_title, sample_title),
                   source_name = COALESCE(excluded.source_name, source_name),
                   description = COALESCE(excluded.description, description),
                   characteristics = COALESCE(excluded.characteristics, characteristics),
                   sra_accession = COALESCE(excluded.sra_accession, sra_accession)
            """,
            (
                geo_accession_id,
                sample["gsm_id"],
                sample.get("sample_title"),
                sample.get("source_name"),
                sample.get("description"),
                json.dumps(sample.get("characteristics")) if sample.get("characteristics") else None,
                sample.get("sra_accession"),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_geo_samples(self, geo_accession_id: int) -> list[dict]:
        """Get all GEO samples for an accession."""
        rows = self._conn.execute(
            "SELECT * FROM geo_samples WHERE geo_accession_id = ?",
            (geo_accession_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("characteristics"):
                try:
                    d["characteristics"] = json.loads(d["characteristics"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(d)
        return result

    def store_sample_stage_mapping(self, mapping: dict) -> int:
        """Insert or update a sample-to-stage mapping. Returns the row ID."""
        cur = self._conn.execute(
            """INSERT INTO geo_sample_stage_mappings
               (geo_sample_id, protocol_id, stage_name, stage_number,
                time_point_day, condition_label, mapping_confidence, mapping_method)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(geo_sample_id, protocol_id) DO UPDATE SET
                   stage_name = excluded.stage_name,
                   stage_number = excluded.stage_number,
                   time_point_day = excluded.time_point_day,
                   condition_label = excluded.condition_label,
                   mapping_confidence = excluded.mapping_confidence,
                   mapping_method = excluded.mapping_method
            """,
            (
                mapping["geo_sample_id"],
                mapping["protocol_id"],
                mapping.get("stage_name"),
                mapping.get("stage_number"),
                mapping.get("time_point_day"),
                mapping.get("condition_label"),
                mapping.get("mapping_confidence"),
                mapping.get("mapping_method"),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_sample_stage_mappings(self, protocol_id: int) -> list[dict]:
        """Get all sample-to-stage mappings for a protocol."""
        rows = self._conn.execute(
            """SELECT m.*, s.gsm_id, s.sample_title, s.sra_accession
               FROM geo_sample_stage_mappings m
               JOIN geo_samples s ON s.id = m.geo_sample_id
               WHERE m.protocol_id = ?""",
            (protocol_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_papers_needing_geo(self) -> list[dict]:
        """Papers eligible for GEO discovery (extractable + not yet checked)."""
        rows = self._conn.execute(
            """SELECT p.* FROM papers p
               WHERE p.triage_category IN (?, ?, ?, ?)
               AND p.extraction_status = 'completed'
               AND p.geo_status IS NULL
               ORDER BY p.id""",
            (*EXTRACTABLE_CATEGORIES, REVIEW_CATEGORY),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_geo_accessions_with_paper(self) -> list[dict]:
        """Get all GEO accessions joined with paper info (xml_path, pmid, supplement_text_path)."""
        rows = self._conn.execute(
            """SELECT ga.*, p.pmc_id, p.xml_path, p.pmid, p.supplement_text_path
               FROM geo_accessions ga
               JOIN papers p ON p.id = ga.paper_id
               ORDER BY ga.id"""
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for k in ("discovery_strategies", "linked_pmids"):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result

    def remove_geo_accession(self, acc_id: int) -> None:
        """Delete a geo_accession row and cascade to samples and mappings."""
        # Delete sample-stage mappings for samples under this accession
        self._conn.execute(
            """DELETE FROM geo_sample_stage_mappings
               WHERE geo_sample_id IN (
                   SELECT id FROM geo_samples WHERE geo_accession_id = ?
               )""",
            (acc_id,),
        )
        # Delete samples
        self._conn.execute(
            "DELETE FROM geo_samples WHERE geo_accession_id = ?", (acc_id,)
        )
        # Delete the accession itself
        self._conn.execute(
            "DELETE FROM geo_accessions WHERE id = ?", (acc_id,)
        )
        self._conn.commit()

    def mark_accession_grounded(self, acc_id: int, grounding_status: str) -> None:
        """Update grounding metadata on a geo_accession row.

        Adds a 'grounding_status' column if it doesn't exist yet, then sets it.
        """
        # Ensure column exists
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(geo_accessions)").fetchall()}
        if "grounding_status" not in cols:
            self._conn.execute(
                "ALTER TABLE geo_accessions ADD COLUMN grounding_status TEXT DEFAULT NULL"
            )
        self._conn.execute(
            "UPDATE geo_accessions SET grounding_status = ? WHERE id = ?",
            (grounding_status, acc_id),
        )
        self._conn.commit()

    def get_papers_needing_geo_mapping(self) -> list[dict]:
        """Papers with GEO accessions that have samples but no mappings yet."""
        rows = self._conn.execute(
            """SELECT DISTINCT p.* FROM papers p
               JOIN geo_accessions ga ON ga.paper_id = p.id
               JOIN geo_samples gs ON gs.geo_accession_id = ga.id
               WHERE ga.context = 'own_data'
               AND ga.soft_fetched = 1
               AND gs.id NOT IN (
                   SELECT geo_sample_id FROM geo_sample_stage_mappings
               )
               ORDER BY p.id"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # RNA-seq metadata
    # ------------------------------------------------------------------

    def get_papers_needing_rnaseq(self) -> list[dict]:
        """Papers eligible for RNA-seq extraction (completed extraction, not yet checked)."""
        rows = self._conn.execute(
            """SELECT p.* FROM papers p
               WHERE p.extraction_status = 'completed'
               AND p.rnaseq_status IS NULL
               ORDER BY p.id"""
        ).fetchall()
        return [dict(r) for r in rows]

    def store_rnaseq_metadata(self, paper_id: int, data: dict) -> int:
        """Insert or update RNA-seq metadata for a paper. Returns the row ID."""
        cur = self._conn.execute(
            """INSERT INTO rnaseq_metadata
               (paper_id, has_rnaseq, rnaseq_type, technology, library_prep,
                read_type, read_length_bp, sequencing_depth, genome_build,
                annotation, alignment_tool, quantification_tool, normalization,
                de_method, accessions, deg_summary, pathway_analysis,
                data_availability, extraction_notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(paper_id) DO UPDATE SET
                   has_rnaseq = excluded.has_rnaseq,
                   rnaseq_type = excluded.rnaseq_type,
                   technology = excluded.technology,
                   library_prep = excluded.library_prep,
                   read_type = excluded.read_type,
                   read_length_bp = excluded.read_length_bp,
                   sequencing_depth = excluded.sequencing_depth,
                   genome_build = excluded.genome_build,
                   annotation = excluded.annotation,
                   alignment_tool = excluded.alignment_tool,
                   quantification_tool = excluded.quantification_tool,
                   normalization = excluded.normalization,
                   de_method = excluded.de_method,
                   accessions = excluded.accessions,
                   deg_summary = excluded.deg_summary,
                   pathway_analysis = excluded.pathway_analysis,
                   data_availability = excluded.data_availability,
                   extraction_notes = excluded.extraction_notes
            """,
            (
                paper_id,
                1 if data.get("has_rnaseq") else 0,
                data.get("rnaseq_type"),
                data.get("technology"),
                data.get("library_prep"),
                data.get("read_type"),
                data.get("read_length_bp"),
                data.get("sequencing_depth"),
                data.get("genome_build"),
                data.get("annotation"),
                data.get("alignment_tool"),
                data.get("quantification_tool"),
                data.get("normalization"),
                data.get("de_method"),
                json.dumps(data.get("accessions")) if data.get("accessions") else None,
                json.dumps(data.get("deg_summary")) if data.get("deg_summary") else None,
                json.dumps(data.get("pathway_analysis")) if data.get("pathway_analysis") else None,
                json.dumps(data.get("data_availability")) if data.get("data_availability") else None,
                data.get("extraction_notes"),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_rnaseq_metadata(self, paper_id: int) -> dict | None:
        """Get RNA-seq metadata for a paper."""
        row = self._conn.execute(
            "SELECT * FROM rnaseq_metadata WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for k in ("accessions", "deg_summary", "pathway_analysis", "data_availability"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    # ------------------------------------------------------------------
    # Repository metadata
    # ------------------------------------------------------------------

    def store_repository_metadata(self, paper_id: int, data: dict) -> int:
        """Insert or update repository metadata for a paper+accession. Returns row ID."""
        cur = self._conn.execute(
            """INSERT INTO repository_metadata
               (paper_id, accession, repository, project_title, organism,
                data_type, platform, sample_count, has_processed_matrix,
                supplementary_files, sample_metadata, fetch_status, raw_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(paper_id, accession) DO UPDATE SET
                   repository = excluded.repository,
                   project_title = COALESCE(excluded.project_title, project_title),
                   organism = COALESCE(excluded.organism, organism),
                   data_type = COALESCE(excluded.data_type, data_type),
                   platform = COALESCE(excluded.platform, platform),
                   sample_count = COALESCE(excluded.sample_count, sample_count),
                   has_processed_matrix = MAX(excluded.has_processed_matrix, has_processed_matrix),
                   supplementary_files = COALESCE(excluded.supplementary_files, supplementary_files),
                   sample_metadata = COALESCE(excluded.sample_metadata, sample_metadata),
                   fetch_status = excluded.fetch_status,
                   raw_response = COALESCE(excluded.raw_response, raw_response)
            """,
            (
                paper_id,
                data["accession"],
                data["repository"],
                data.get("project_title"),
                data.get("organism"),
                data.get("data_type"),
                data.get("platform"),
                data.get("sample_count"),
                1 if data.get("has_processed_matrix") else 0,
                json.dumps(data.get("supplementary_files")) if data.get("supplementary_files") else None,
                json.dumps(data.get("sample_metadata")) if data.get("sample_metadata") else None,
                data.get("fetch_status", "fetched"),
                data.get("raw_response"),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_repository_metadata(self, paper_id: int) -> list[dict]:
        """Get all repository metadata for a paper."""
        rows = self._conn.execute(
            "SELECT * FROM repository_metadata WHERE paper_id = ?", (paper_id,)
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for k in ("supplementary_files", "sample_metadata"):
                if d.get(k):
                    try:
                        d[k] = json.loads(d[k])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result

    def get_papers_needing_crossref(self) -> list[dict]:
        """Papers with RNA-seq accessions that haven't been cross-referenced yet."""
        rows = self._conn.execute(
            """SELECT p.* FROM papers p
               JOIN rnaseq_metadata rm ON rm.paper_id = p.id
               WHERE rm.has_rnaseq = 1
               AND rm.accessions IS NOT NULL
               AND p.id NOT IN (
                   SELECT DISTINCT paper_id FROM repository_metadata
               )
               ORDER BY p.id"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Expression values
    # ------------------------------------------------------------------

    def store_expression_value(self, paper_id: int, gene: str, value: float | None,
                               **fields) -> int:
        """Insert a single expression value. Returns row ID."""
        cur = self._conn.execute(
            """INSERT INTO expression_values
               (paper_id, protocol_id, gene_symbol, gene_alias, value, unit,
                condition_label, time_point_day, comparison, padj,
                source_type, source_detail, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper_id,
                fields.get("protocol_id"),
                gene,
                fields.get("gene_alias"),
                value,
                fields.get("unit"),
                fields.get("condition_label"),
                fields.get("time_point_day"),
                fields.get("comparison"),
                fields.get("padj"),
                fields.get("source_type", "unknown"),
                fields.get("source_detail"),
                fields.get("confidence", 0.7),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def store_expression_values_batch(self, values: list[dict]) -> int:
        """Insert multiple expression values. Returns count inserted."""
        cur = self._conn.cursor()
        count = 0
        for v in values:
            cur.execute(
                """INSERT INTO expression_values
                   (paper_id, protocol_id, gene_symbol, gene_alias, value, unit,
                    condition_label, time_point_day, comparison, padj,
                    source_type, source_detail, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    v["paper_id"],
                    v.get("protocol_id"),
                    v["gene_symbol"],
                    v.get("gene_alias"),
                    v.get("value"),
                    v.get("unit"),
                    v.get("condition_label"),
                    v.get("time_point_day"),
                    v.get("comparison"),
                    v.get("padj"),
                    v.get("source_type", "unknown"),
                    v.get("source_detail"),
                    v.get("confidence", 0.7),
                ),
            )
            count += 1
        self._conn.commit()
        return count

    def get_expression_values(self, paper_id: int) -> list[dict]:
        """Get all expression values for a paper."""
        rows = self._conn.execute(
            "SELECT * FROM expression_values WHERE paper_id = ?", (paper_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_expression_by_gene(self, gene_symbol: str) -> list[dict]:
        """Get all expression values for a gene across papers."""
        rows = self._conn.execute(
            "SELECT * FROM expression_values WHERE gene_symbol = ?", (gene_symbol,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_expression_matrix_data(self) -> list[dict]:
        """Aggregated query: best value per (protocol, gene) for the expression matrix."""
        rows = self._conn.execute(
            """SELECT ev.paper_id, ev.protocol_id, ev.gene_symbol, ev.value,
                      ev.unit, ev.source_type, ev.source_detail, ev.confidence,
                      ev.condition_label,
                      p.pmc_id, pr.protocol_arm
               FROM expression_values ev
               JOIN papers p ON p.id = ev.paper_id
               LEFT JOIN protocols pr ON pr.id = ev.protocol_id
               WHERE ev.id IN (
                   SELECT id FROM (
                       SELECT id, ROW_NUMBER() OVER (
                           PARTITION BY COALESCE(protocol_id, paper_id), gene_symbol
                           ORDER BY confidence DESC
                       ) AS rn
                       FROM expression_values
                   ) WHERE rn = 1
               )
               ORDER BY ev.paper_id, ev.protocol_id, ev.gene_symbol"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stage_expression_data(self) -> list[dict]:
        """Query joining expression values with stage mappings."""
        rows = self._conn.execute(
            """SELECT ev.*, m.stage_name, m.stage_number, m.condition_label AS stage_condition,
                      s.gsm_id, p.pmc_id
               FROM expression_values ev
               JOIN papers p ON p.id = ev.paper_id
               LEFT JOIN geo_sample_stage_mappings m ON m.protocol_id = ev.protocol_id
               LEFT JOIN geo_samples s ON s.id = m.geo_sample_id
               WHERE m.id IS NOT NULL
               ORDER BY ev.paper_id, ev.protocol_id, m.stage_number"""
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Pipeline dashboard: counts by category, status, etc."""
        cur = self._conn.cursor()

        total = cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

        # By category
        cat_rows = cur.execute(
            "SELECT triage_category, COUNT(*) as cnt FROM papers GROUP BY triage_category"
        ).fetchall()
        by_category = {r["triage_category"]: r["cnt"] for r in cat_rows}

        # By extraction status
        status_rows = cur.execute(
            """SELECT extraction_status, COUNT(*) as cnt FROM papers
               WHERE triage_category IN (?, ?, ?, ?)
               GROUP BY extraction_status""",
            (*EXTRACTABLE_CATEGORIES, REVIEW_CATEGORY),
        ).fetchall()
        by_status = {r["extraction_status"]: r["cnt"] for r in status_rows}

        # Text preparation
        with_text = cur.execute(
            "SELECT COUNT(*) FROM papers WHERE parsed_text_path IS NOT NULL"
        ).fetchone()[0]

        # Protocols
        protocol_count = cur.execute("SELECT COUNT(*) FROM protocols").fetchone()[0]

        # Corpus cache
        cache_count = cur.execute("SELECT COUNT(*) FROM corpus_cache").fetchone()[0]

        # GEO stats
        geo_papers = cur.execute(
            "SELECT COUNT(*) FROM papers WHERE geo_status = 'linked'"
        ).fetchone()[0]
        geo_series = cur.execute(
            "SELECT COUNT(*) FROM geo_accessions WHERE context = 'own_data'"
        ).fetchone()[0]
        geo_samples_total = cur.execute(
            "SELECT COUNT(*) FROM geo_samples"
        ).fetchone()[0]
        geo_mapped_samples = cur.execute(
            "SELECT COUNT(DISTINCT geo_sample_id) FROM geo_sample_stage_mappings"
        ).fetchone()[0]

        # RNA-seq stats
        rnaseq_checked = cur.execute(
            "SELECT COUNT(*) FROM papers WHERE rnaseq_status IS NOT NULL"
        ).fetchone()[0]
        rnaseq_papers = cur.execute(
            "SELECT COUNT(*) FROM papers WHERE rnaseq_status = 'has_rnaseq'"
        ).fetchone()[0]
        expression_values_count = cur.execute(
            "SELECT COUNT(*) FROM expression_values"
        ).fetchone()[0]
        expression_genes = cur.execute(
            "SELECT COUNT(DISTINCT gene_symbol) FROM expression_values"
        ).fetchone()[0]
        expression_papers = cur.execute(
            "SELECT COUNT(DISTINCT paper_id) FROM expression_values"
        ).fetchone()[0]

        return {
            "total_papers": total,
            "by_category": by_category,
            "extraction_status": by_status,
            "papers_with_text": with_text,
            "protocols_extracted": protocol_count,
            "corpus_cache_entries": cache_count,
            "geo_papers_linked": geo_papers,
            "geo_series_own_data": geo_series,
            "geo_samples_total": geo_samples_total,
            "geo_mapped_samples": geo_mapped_samples,
            "rnaseq_checked": rnaseq_checked,
            "rnaseq_papers": rnaseq_papers,
            "expression_values_count": expression_values_count,
            "expression_genes": expression_genes,
            "expression_papers": expression_papers,
        }

    def get_triage_result(self, paper_id: int) -> dict | None:
        """Get triage result for a paper."""
        row = self._conn.execute(
            "SELECT * FROM triage_results WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for k in ("base_protocols", "key_cell_types"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
