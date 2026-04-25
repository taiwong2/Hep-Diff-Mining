import os
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

load_dotenv()

EDIRECT_PATH = os.path.expanduser("~/edirect")
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class PMCResult:
    pmc_id: str
    pmid: str | None = None
    title: str | None = None
    abstract: str | None = None
    journal: str | None = None
    pub_date: str | None = None
    doi: str | None = None
    authors: list[str] = field(default_factory=list)


class ApiKeyRotator:
    def __init__(self, keys: list[str]):
        self._keys = keys
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
            return key

    def __len__(self):
        return len(self._keys)


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self._interval = 1.0 / requests_per_second
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


class PMCClient:
    """Client for searching PMC via NCBI EDirect command-line tools.

    EDirect pipeline model:
        esearch -db pmc -query "..." | efetch -format uid
        esearch -db pmc -query "..." | efetch -format docsum
    esearch posts results to the NCBI history server and outputs an
    ENTREZ_DIRECT XML block. That block is piped into efetch which
    retrieves the actual records.
    """

    def __init__(self, email: str | None = None, api_keys: list[str] | None = None):
        self.email = email or os.getenv("NCBI_EMAIL", "")
        raw_keys = api_keys or [
            k.strip()
            for k in os.getenv("NCBI_API_KEYS", "").split(",")
            if k.strip()
        ]
        self._rotator = ApiKeyRotator(raw_keys) if raw_keys else None
        rate = 7.0 * len(raw_keys) if raw_keys else 2.0
        self._limiter = RateLimiter(rate)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, retmax: int | None = None) -> list[str]:
        """Search PMC and return a list of PMC IDs.

        Args:
            query: PubMed/PMC query string.
            retmax: Maximum number of IDs to return. None means all results.
        """
        # esearch posts to history server, efetch -format uid pulls IDs
        edirect_xml = self._esearch(query)
        count = self._parse_count(edirect_xml)
        if count == 0:
            return []

        limit = retmax if retmax is not None else count
        ids = self._efetch_from_pipe(edirect_xml, fmt="uid", start=1, stop=min(limit, count))
        return [line.strip() for line in ids.splitlines() if line.strip()]

    def fetch_summaries(self, pmc_ids: list[str]) -> list[PMCResult]:
        """Fetch document summaries for a list of PMC IDs."""
        results = []
        for batch in _batches(pmc_ids, 200):
            xml = self._efetch_by_id(batch, fmt="docsum")
            results.extend(self._parse_docsummaries(xml))
        return results

    def fetch_xml(self, pmc_ids: list[str]) -> str:
        """Fetch full PMC XML for the given IDs."""
        chunks = []
        for batch in _batches(pmc_ids, 50):
            chunks.append(self._efetch_by_id(batch, fmt="xml"))
        return "\n".join(chunks)

    def elink(self, db_from: str, db_to: str, ids: list[str]) -> str:
        """Run elink to find linked records between NCBI databases.

        Example: elink('pubmed', 'gds', ['12345']) finds GDS records
        linked to PubMed ID 12345.

        Returns raw ENTREZ_DIRECT XML output for piping to efetch.
        """
        self._limiter.acquire()
        id_str = ",".join(ids)
        cmd = [
            os.path.join(EDIRECT_PATH, "elink"),
            "-db", db_from,
            "-id", id_str,
            "-target", db_to,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=self._build_env(),
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"elink failed: {result.stderr.strip()}")
        return result.stdout

    def elink_https(self, db_from: str, db_to: str, ids: list[str]) -> list[str]:
        """HTTPS elink: returns list of linked record IDs."""
        self._limiter.acquire()
        params = {
            "dbfrom": db_from,
            "db": db_to,
            "id": ",".join(ids),
            "retmode": "xml",
            "cmd": "neighbor",
        }
        if self.email:
            params["email"] = self.email
        if self._rotator:
            params["api_key"] = self._rotator.next()
        resp = requests.get(f"{EUTILS_BASE}/elink.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        linked_ids = []
        for link in root.iter("Link"):
            id_elem = link.find("Id")
            if id_elem is not None and id_elem.text:
                linked_ids.append(id_elem.text.strip())
        return linked_ids

    def esummary_https(self, db: str, ids: list[str]) -> str:
        """HTTPS esummary: returns raw XML response."""
        self._limiter.acquire()
        params = {
            "db": db,
            "id": ",".join(ids),
            "retmode": "xml",
            "version": "2.0",
        }
        if self.email:
            params["email"] = self.email
        if self._rotator:
            params["api_key"] = self._rotator.next()
        resp = requests.get(f"{EUTILS_BASE}/esummary.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        # Strip XML/DOCTYPE declarations so callers can safely wrap in <root>
        text = re.sub(r'<\?xml[^?]*\?>\s*', '', resp.text)
        text = re.sub(r'<!DOCTYPE[^>]*>\s*', '', text)
        return text

    def search_and_summarize(self, query: str, retmax: int | None = None) -> list[PMCResult]:
        """Search then fetch summaries for all results."""
        ids = self.search(query, retmax=retmax)
        if not ids:
            return []
        return self.fetch_summaries(ids)

    # ------------------------------------------------------------------
    # EDirect subprocess helpers
    # ------------------------------------------------------------------

    def _build_env(self) -> dict[str, str]:
        """Build a subprocess environment with the next API key injected."""
        env = os.environ.copy()
        env["PATH"] = EDIRECT_PATH + ":" + env.get("PATH", "")
        if self.email:
            env["NCBI_EMAIL"] = self.email
        if self._rotator:
            env["NCBI_API_KEY"] = self._rotator.next()
        return env

    def _esearch(self, query: str) -> str:
        """Run esearch and return the ENTREZ_DIRECT XML block."""
        self._limiter.acquire()
        cmd = [
            os.path.join(EDIRECT_PATH, "esearch"),
            "-db", "pmc",
            "-query", query,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=self._build_env(),
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"esearch failed: {result.stderr.strip()}")
        return result.stdout

    def _efetch_from_pipe(self, edirect_xml: str, fmt: str,
                          start: int | None = None, stop: int | None = None) -> str:
        """Run efetch by piping an ENTREZ_DIRECT XML block into stdin."""
        self._limiter.acquire()
        cmd = [os.path.join(EDIRECT_PATH, "efetch"), "-format", fmt]
        if start is not None:
            cmd.extend(["-start", str(start)])
        if stop is not None:
            cmd.extend(["-stop", str(stop)])
        result = subprocess.run(
            cmd,
            input=edirect_xml,
            capture_output=True,
            text=True,
            env=self._build_env(),
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"efetch failed: {result.stderr.strip()}")
        return result.stdout

    def _efetch_by_id(self, pmc_ids: list[str], fmt: str, db: str = "pmc") -> str:
        """Run efetch directly with -id flag for known IDs."""
        self._limiter.acquire()
        id_str = ",".join(pmc_ids)
        cmd = [
            os.path.join(EDIRECT_PATH, "efetch"),
            "-db", db,
            "-id", id_str,
            "-format", fmt,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=self._build_env(),
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"efetch failed: {result.stderr.strip()}")
        return result.stdout

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_count(edirect_xml: str) -> int:
        """Extract the <Count> from an ENTREZ_DIRECT XML block."""
        root = ET.fromstring(edirect_xml)
        return int(root.findtext("Count", "0"))

    @staticmethod
    def _parse_docsummaries(xml: str) -> list[PMCResult]:
        results = []
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            root = ET.fromstring(f"<root>{xml}</root>")

        for doc in root.iter("DocumentSummary"):
            pmc_id = doc.findtext("Id", "")

            # ArticleIds use child elements <IdType> and <Value>
            article_ids = {}
            for aid in doc.findall(".//ArticleId"):
                id_type = aid.findtext("IdType", "")
                value = aid.findtext("Value", "")
                if id_type and value:
                    article_ids[id_type] = value

            authors = []
            for author in doc.findall(".//Author"):
                name = author.findtext("Name", "")
                if name:
                    authors.append(name)

            results.append(PMCResult(
                pmc_id=pmc_id,
                pmid=article_ids.get("pmid"),
                title=doc.findtext("Title", None),
                abstract=None,
                journal=doc.findtext("FullJournalName", None) or doc.findtext("Source", None),
                pub_date=doc.findtext("PubDate", None) or doc.findtext("EPubDate", None),
                doi=article_ids.get("doi"),
                authors=authors,
            ))
        return results


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
