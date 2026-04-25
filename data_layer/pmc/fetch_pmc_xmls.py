"""Fetch PMC XMLs for hepatocyte differentiation papers using three queries
(iPSC/ESC hepatocyte, liver organoid, direct reprogramming), union results
by PMC ID, and store them in data/ with 5-character prefix sharding.

Layout:
    data/PMC76/PMC7612819.xml
    data/PMC41/PMC4132456.xml

Usage:
    python -m data_layer.pmc.fetch_pmc_xmls
"""

import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is on sys.path so `data_layer` resolves as a package
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tqdm import tqdm

from data_layer.pmc.pmc_client import PMCClient, _batches

# NOTE: PMC E-utilities silently treat [Title/Abstract] as [Title]-only.
# Work around by expanding each term to (term[Title] OR term[Abstract]).

# Q1: iPSC/ESC -> Hepatocyte differentiation (main query)
QUERY_HEPATOCYTE = (
    '((iPSC[Title] OR iPSC[Abstract]) '
    'OR ("induced pluripotent"[Title] OR "induced pluripotent"[Abstract]) '
    'OR (hiPSC[Title] OR hiPSC[Abstract]) '
    'OR ("iPS cell"[Title] OR "iPS cell"[Abstract]) '
    'OR ("embryonic stem"[Title] OR "embryonic stem"[Abstract]) '
    'OR (hESC[Title] OR hESC[Abstract]) '
    'OR ("pluripotent stem cell"[Title] OR "pluripotent stem cell"[Abstract])) '
    'AND ((hepatocyte[Title] OR hepatocyte[Abstract]) '
    'OR (hepatic[Title] OR hepatic[Abstract]) '
    'OR ("hepatocyte-like"[Title] OR "hepatocyte-like"[Abstract]) '
    'OR (HLC[Title] OR HLC[Abstract]) '
    'OR (hepatoblast[Title] OR hepatoblast[Abstract])) '
    'AND "open access"[filter]'
)

# Q2: iPSC/ESC -> Liver organoid / liver bud
QUERY_ORGANOID = (
    '((iPSC[Title] OR iPSC[Abstract]) '
    'OR ("induced pluripotent"[Title] OR "induced pluripotent"[Abstract]) '
    'OR (hiPSC[Title] OR hiPSC[Abstract]) '
    'OR ("embryonic stem"[Title] OR "embryonic stem"[Abstract]) '
    'OR (hESC[Title] OR hESC[Abstract]) '
    'OR ("pluripotent stem"[Title] OR "pluripotent stem"[Abstract])) '
    'AND (("liver organoid"[Title] OR "liver organoid"[Abstract]) '
    'OR ("hepatic organoid"[Title] OR "hepatic organoid"[Abstract]) '
    'OR ("liver bud"[Title] OR "liver bud"[Abstract])) '
    'AND "open access"[filter]'
)

# Q3: Direct reprogramming / transdifferentiation -> Hepatocyte
QUERY_REPROGRAMMING = (
    '(("direct reprogramming"[Title] OR "direct reprogramming"[Abstract]) '
    'OR ("direct conversion"[Title] OR "direct conversion"[Abstract]) '
    'OR (transdifferentiation[Title] OR transdifferentiation[Abstract])) '
    'AND ((hepatocyte[Title] OR hepatocyte[Abstract]) '
    'OR (hepatic[Title] OR hepatic[Abstract]) '
    'OR ("hepatocyte-like"[Title] OR "hepatocyte-like"[Abstract])) '
    'AND "open access"[filter]'
)

QUERIES = [
    ("Q1-Hepatocyte", QUERY_HEPATOCYTE),
    ("Q2-Organoid", QUERY_ORGANOID),
    ("Q3-Reprogramming", QUERY_REPROGRAMMING),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "db"
BATCH_SIZE = 25  # articles per efetch call
MAX_RETRIES = 3
NUM_WORKERS = 8  # parallel download threads (6 API keys × ~1.3 workers/key)


def shard_path(pmc_id: str) -> Path:
    """PMC7612819 -> data_layer/data/PMC76/PMC7612819.xml"""
    tag = f"PMC{pmc_id}" if not pmc_id.startswith("PMC") else pmc_id
    prefix = tag[:5]
    return DATA_DIR / prefix / f"{tag}.xml"


def extract_articles(raw_xml: str) -> dict[str, str]:
    """Split a <pmc-articleset> blob into {pmc_id: article_xml} pairs."""
    articles = {}
    # The XML from efetch has a DOCTYPE that ET chokes on — strip it
    cleaned = re.sub(r'<!DOCTYPE[^>]*>', '', raw_xml)
    cleaned = re.sub(r'<\?xml[^>]*\?>', '', cleaned)

    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError:
        # Try wrapping fragments
        try:
            root = ET.fromstring(f"<root>{cleaned}</root>")
        except ET.ParseError:
            return articles

    for article in root.iter("article"):
        # Find PMC ID in article-meta
        pmc_id = None
        for aid in article.findall(".//article-id"):
            if aid.get("pub-id-type") == "pmcid":
                pmc_id = aid.text
                break
            if aid.get("pub-id-type") == "pmcaid":
                pmc_id = aid.text

        if not pmc_id:
            continue

        # Strip the PMC prefix if present for consistent storage
        pmc_id = pmc_id.replace("PMC", "").split(".")[0]
        article_xml = ET.tostring(article, encoding="unicode", xml_declaration=False)
        articles[pmc_id] = article_xml

    return articles


def _fetch_batch(client: PMCClient, batch: list[str], progress: tqdm) -> list[str]:
    """Fetch a single batch of PMC IDs. Returns list of IDs that failed."""
    to_fetch = [pid for pid in batch if not shard_path(pid).exists()]
    if not to_fetch:
        progress.update(len(batch))
        return []

    failed = []
    raw_xml = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_xml = client._efetch_by_id(to_fetch, fmt="xml")
            break
        except RuntimeError as e:
            if attempt == MAX_RETRIES:
                failed.extend(to_fetch)
                progress.update(len(batch))
                tqdm.write(f"FAILED batch after {MAX_RETRIES} retries: {e}")
                return failed
            wait = 2 ** attempt
            tqdm.write(f"Retry {attempt}/{MAX_RETRIES} in {wait}s: {e}")
            time.sleep(wait)

    if raw_xml is None:
        return failed

    articles = extract_articles(raw_xml)

    for pid in to_fetch:
        xml_str = articles.get(pid)
        if xml_str is None:
            failed.append(pid)
            continue
        out = shard_path(pid)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(xml_str, encoding="utf-8")

    progress.update(len(batch))
    return failed


def fetch_and_store(client: PMCClient, pmc_ids: list[str], progress: tqdm,
                    workers: int = NUM_WORKERS) -> list[str]:
    """Fetch XMLs in parallel batches and write each article to its sharded path.
    Returns list of IDs that failed."""
    batches = list(_batches(pmc_ids, BATCH_SIZE))
    failed = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_batch, client, batch, progress): batch
            for batch in batches
        }
        for future in as_completed(futures):
            failed.extend(future.result())

    return failed


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    client = PMCClient()

    # Run each query and union results by PMC ID
    seen: set[str] = set()
    all_ids: list[str] = []

    print("=" * 60)
    print("Running 3 PMC queries (union by PMC ID)")
    print("=" * 60)

    for label, query in QUERIES:
        print(f"\n[{label}] Searching PMC...")
        ids = client.search(query)
        new_ids = [pid for pid in ids if pid not in seen]
        seen.update(new_ids)
        all_ids.extend(new_ids)
        print(f"[{label}] Hits: {len(ids)}, net-new: {len(new_ids)}")

    print(f"\nTotal unique PMC IDs: {len(all_ids)}")

    if not all_ids:
        print("Nothing to fetch.")
        return

    already = sum(1 for pid in all_ids if shard_path(pid).exists())
    remaining = len(all_ids) - already
    print(f"Already downloaded: {already}, remaining: {remaining}")
    print(f"Using {NUM_WORKERS} download workers")

    with tqdm(total=len(all_ids), desc="Fetching XMLs", unit="article") as pbar:
        pbar.update(already)
        failed = fetch_and_store(client, all_ids, pbar)

    if failed:
        failed_path = DATA_DIR / "failed_ids.txt"
        failed_path.write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} articles failed — IDs written to {failed_path}")
    else:
        print("All articles fetched successfully.")


if __name__ == "__main__":
    main()
