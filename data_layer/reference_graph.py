"""Citation DAG builder and topological sort for extraction ordering.

Parses <ref-list> from each PMC XML to find in-corpus citations,
builds a networkx DiGraph, and computes topological sort so that
foundational papers (e.g., Si-Tayeb 2010, Hay 2008) are extracted first.

Usage:
    from data_layer.reference_graph import build_reference_graph
    order = build_reference_graph(db)
    # order is a list of paper_ids in processing priority
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_reference_graph(db) -> list[int]:
    """Build citation DAG from PMC XMLs and return topological sort order.

    Args:
        db: PipelineDB instance with papers already imported.

    Returns:
        List of paper IDs in topological order (process first to last).
        Also writes processing_priority back to the papers table.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("networkx not installed. pip install networkx")
        raise

    from data_layer.xml_to_text import extract_ref_list

    # Build lookup indexes from all papers in DB
    all_papers = db._conn.execute(
        "SELECT id, pmc_id, doi, pmid FROM papers"
    ).fetchall()

    doi_to_id: dict[str, int] = {}
    pmid_to_id: dict[str, int] = {}
    pmc_to_id: dict[str, int] = {}

    for p in all_papers:
        pid = p["id"]
        if p["doi"]:
            doi_to_id[p["doi"]] = pid
        if p["pmid"]:
            pmid_to_id[p["pmid"]] = pid
        if p["pmc_id"]:
            pmc_to_id[p["pmc_id"]] = pid

    # Get extractable papers
    extractable = db._conn.execute(
        """SELECT id, pmc_id, xml_path FROM papers
           WHERE triage_category IN ('primary_protocol', 'disease_model',
                                     'methods_tool', 'review')
           AND xml_path IS NOT NULL"""
    ).fetchall()

    extractable_ids = {p["id"] for p in extractable}

    # Build DAG
    G = nx.DiGraph()
    for p in extractable:
        G.add_node(p["id"])

    edges_added = 0
    refs_total = 0

    for paper in extractable:
        paper_id = paper["id"]
        xml_path = paper["xml_path"]

        if not xml_path or not Path(xml_path).exists():
            continue

        refs = extract_ref_list(xml_path)
        refs_total += len(refs)

        for ref in refs:
            # Resolve reference to an in-corpus paper
            ref_paper_id = None
            if ref["doi"] and ref["doi"] in doi_to_id:
                ref_paper_id = doi_to_id[ref["doi"]]
            elif ref["pmid"] and ref["pmid"] in pmid_to_id:
                ref_paper_id = pmid_to_id[ref["pmid"]]
            elif ref["pmc_id"] and ref["pmc_id"] in pmc_to_id:
                ref_paper_id = pmc_to_id[ref["pmc_id"]]

            if ref_paper_id and ref_paper_id != paper_id:
                # Store reference in DB
                db.add_reference(
                    paper_id,
                    doi=ref["doi"],
                    pmc_id=ref["pmc_id"],
                )

                # Add edge: ref_paper → paper (ref should be processed first)
                if ref_paper_id in extractable_ids:
                    G.add_edge(ref_paper_id, paper_id)
                    edges_added += 1

    logger.info(
        "Reference graph: %d nodes, %d in-corpus edges (from %d total refs)",
        len(G.nodes), edges_added, refs_total,
    )

    # Handle cycles by removing weakest edges
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        # Remove the last edge in the cycle (arbitrary but consistent)
        edge_to_remove = cycle[-1][:2]
        G.remove_edge(*edge_to_remove)
        logger.info("Broke cycle by removing edge %s -> %s",
                     edge_to_remove[0], edge_to_remove[1])

    # Topological sort
    topo_order = list(nx.topological_sort(G))

    # Add any extractable papers not in the graph (no citation links)
    in_order = set(topo_order)
    for p in extractable:
        if p["id"] not in in_order:
            topo_order.append(p["id"])

    # Write processing_priority back to DB
    for priority, paper_id in enumerate(topo_order):
        db.update_paper(paper_id, processing_priority=priority)

    logger.info("Processing order set: %d papers", len(topo_order))
    return topo_order
