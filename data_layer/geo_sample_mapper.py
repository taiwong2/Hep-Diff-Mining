"""Map GEO samples to protocol stages/time points.

Tier 1: Regex/heuristic matching (fast, no LLM cost)
Tier 2: LLM fallback for ambiguous cases

Usage:
    from data_layer.geo_sample_mapper import map_samples_to_stages
    mappings = map_samples_to_stages(samples, protocol_stages)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "llm" / "agents" / "prompts" / "geo_sample_mapping.txt"

# Day extraction patterns — delimiters include hyphen, underscore, whitespace, start/end
DAY_PATTERNS = [
    re.compile(r'(?:^|[\s_\-])[Dd]ay[\s_\-]*(\d+)'),      # Day 5, day5, Day_5, Day-5
    re.compile(r'(?:^|[\s_\-])[Dd](\d+)(?:[\s_\-]|$)'),   # D5, d5, D5_, -D14
    re.compile(r'(?:^|[\s_\-])(\d+)\s*[Dd](?:[\s_\-]|$)'), # 5d, 5D
    re.compile(r'\b(\d+)\s*days?\b'),                        # 5 days, 5 day
]

# Stage alias mapping
STAGE_ALIASES = {
    # Pre-treatment / undifferentiated
    "ipsc": "pre_treatment",
    "esc": "pre_treatment",
    "hesc": "pre_treatment",
    "undifferentiated": "pre_treatment",
    "pluripotent": "pre_treatment",
    "stem cell": "pre_treatment",
    "d0": "pre_treatment",
    "day0": "pre_treatment",

    # Definitive endoderm
    "de": "definitive_endoderm",
    "definitive endoderm": "definitive_endoderm",
    "definitive_endoderm": "definitive_endoderm",
    "endoderm": "definitive_endoderm",

    # Hepatic endoderm / specification
    "he": "hepatic_endoderm",
    "hepatic endoderm": "hepatic_endoderm",
    "hepatic specification": "hepatic_endoderm",
    "hepatic_endoderm": "hepatic_endoderm",

    # Hepatoblast
    "hb": "hepatoblast",
    "hepatoblast": "hepatoblast",
    "hepatic progenitor": "hepatoblast",
    "hepatic_progenitor": "hepatoblast",
    "hp": "hepatoblast",

    # Mature hepatocyte
    "hlc": "mature_hepatocyte",
    "ihep": "mature_hepatocyte",
    "hepatocyte-like": "mature_hepatocyte",
    "hepatocyte like": "mature_hepatocyte",
    "mature hepatocyte": "mature_hepatocyte",
    "maturation": "mature_hepatocyte",
    "hepatocyte": "mature_hepatocyte",
}

# Map stage alias labels to likely protocol stage names (for matching against protocol stages)
ALIAS_TO_STAGE_KEYWORDS = {
    "pre_treatment": ["pre_treatment", "pre-treatment", "undifferentiated", "ipsc", "esc", "seeding"],
    "definitive_endoderm": ["definitive_endoderm", "definitive endoderm", "endoderm", "de "],
    "hepatic_endoderm": ["hepatic_endoderm", "hepatic endoderm", "hepatic specification", "specification"],
    "hepatoblast": ["hepatoblast", "hepatic progenitor", "progenitor", "expansion"],
    "mature_hepatocyte": ["maturation", "hepatocyte", "hlc", "ihep", "mature"],
}


@dataclass
class SampleStageMapping:
    """Mapping of a GEO sample to a protocol stage."""
    gsm_id: str
    stage_name: str = ""
    stage_number: int | None = None
    time_point_day: int | None = None
    condition_label: str = ""
    mapping_confidence: float = 0.0
    mapping_method: str = ""  # tier1_regex | tier2_llm


@dataclass
class StageRange:
    """A protocol stage with its cumulative day range."""
    stage_number: int
    stage_name: str
    start_day: int
    end_day: int
    duration_days: int


def _build_stage_ranges(stages: list[dict]) -> list[StageRange]:
    """Build cumulative day ranges from protocol stage data."""
    ranges: list[StageRange] = []
    cumulative_day = 0

    for i, stage in enumerate(stages):
        stage_name = stage.get("stage_name", stage.get("name", f"stage_{i+1}"))
        duration = stage.get("duration_days") or stage.get("duration") or 0

        # Try to parse duration from string like "3 days" or "3-5 days"
        if isinstance(duration, str):
            dur_match = re.search(r'(\d+)', str(duration))
            duration = int(dur_match.group(1)) if dur_match else 0

        start = cumulative_day
        end = cumulative_day + int(duration)
        ranges.append(StageRange(
            stage_number=i + 1,
            stage_name=stage_name,
            start_day=start,
            end_day=end,
            duration_days=int(duration),
        ))
        cumulative_day = end

    return ranges


def _extract_day(text: str) -> int | None:
    """Extract a day number from sample title/description."""
    for pattern in DAY_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def _extract_stage_alias(text: str) -> str | None:
    """Extract a stage alias from sample title/description."""
    text_lower = text.lower()
    # Check longer phrases first to avoid partial matches
    sorted_aliases = sorted(STAGE_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        if alias in text_lower:
            return STAGE_ALIASES[alias]
    return None


def _match_alias_to_stage(alias_label: str, stage_ranges: list[StageRange]) -> StageRange | None:
    """Match a stage alias label to the best protocol stage."""
    keywords = ALIAS_TO_STAGE_KEYWORDS.get(alias_label, [])
    for sr in stage_ranges:
        name_lower = sr.stage_name.lower()
        for kw in keywords:
            if kw in name_lower:
                return sr

    # Fallback: for pre_treatment, use stage 0 (before first stage)
    if alias_label == "pre_treatment" and stage_ranges:
        return StageRange(
            stage_number=0,
            stage_name="pre_treatment",
            start_day=0,
            end_day=0,
            duration_days=0,
        )

    # For mature_hepatocyte, use last stage
    if alias_label == "mature_hepatocyte" and stage_ranges:
        return stage_ranges[-1]

    return None


def _day_to_stage(day: int, stage_ranges: list[StageRange]) -> StageRange | None:
    """Find which stage a given day falls into."""
    for sr in stage_ranges:
        if sr.start_day <= day <= sr.end_day:
            return sr
    # If day exceeds all stages, assign to last stage
    if stage_ranges and day > stage_ranges[-1].end_day:
        return stage_ranges[-1]
    return None


# ------------------------------------------------------------------
# Tier 1 — Regex/heuristic mapping
# ------------------------------------------------------------------

def tier1_map_sample(
    gsm_id: str,
    sample_title: str,
    source_name: str,
    stage_ranges: list[StageRange],
) -> SampleStageMapping | None:
    """Try to map a sample using regex and heuristics.

    Returns a mapping or None if ambiguous.
    """
    combined_text = f"{sample_title} {source_name}"

    day = _extract_day(combined_text)
    alias = _extract_stage_alias(combined_text)

    if day is not None and alias is not None:
        # Both day and alias found — high confidence
        stage = _day_to_stage(day, stage_ranges)
        if stage:
            return SampleStageMapping(
                gsm_id=gsm_id,
                stage_name=stage.stage_name,
                stage_number=stage.stage_number,
                time_point_day=day,
                condition_label=alias,
                mapping_confidence=0.9,
                mapping_method="tier1_regex",
            )

    if day is not None:
        # Day only
        stage = _day_to_stage(day, stage_ranges)
        if stage:
            return SampleStageMapping(
                gsm_id=gsm_id,
                stage_name=stage.stage_name,
                stage_number=stage.stage_number,
                time_point_day=day,
                condition_label=stage.stage_name,
                mapping_confidence=0.7,
                mapping_method="tier1_regex",
            )

    if alias is not None:
        # Alias only
        matched_stage = _match_alias_to_stage(alias, stage_ranges)
        if matched_stage:
            return SampleStageMapping(
                gsm_id=gsm_id,
                stage_name=matched_stage.stage_name,
                stage_number=matched_stage.stage_number,
                time_point_day=matched_stage.start_day,
                condition_label=alias,
                mapping_confidence=0.7,
                mapping_method="tier1_regex",
            )

    return None


def tier1_map_all(
    samples: list[dict],
    stage_ranges: list[StageRange],
) -> tuple[list[SampleStageMapping], list[dict]]:
    """Map all samples using Tier 1. Returns (mapped, unmapped_samples)."""
    mapped: list[SampleStageMapping] = []
    unmapped: list[dict] = []

    for sample in samples:
        result = tier1_map_sample(
            gsm_id=sample["gsm_id"],
            sample_title=sample.get("sample_title", ""),
            source_name=sample.get("source_name", ""),
            stage_ranges=stage_ranges,
        )
        if result:
            mapped.append(result)
        else:
            unmapped.append(sample)

    return mapped, unmapped


# ------------------------------------------------------------------
# Tier 2 — LLM fallback
# ------------------------------------------------------------------

async def tier2_map_samples(
    client,
    samples: list[dict],
    stage_ranges: list[StageRange],
) -> list[SampleStageMapping]:
    """Use LLM to map ambiguous samples to protocol stages."""
    if not samples:
        return []

    system_prompt = PROMPT_PATH.read_text()

    # Build protocol summary for the LLM
    stage_summary = "Protocol stages:\n"
    for sr in stage_ranges:
        stage_summary += (
            f"  Stage {sr.stage_number}: {sr.stage_name} "
            f"(Day {sr.start_day}-{sr.end_day}, {sr.duration_days} days)\n"
        )

    # Build sample list
    sample_text = "GEO samples to map:\n"
    for s in samples:
        sample_text += f"  {s['gsm_id']}: title=\"{s.get('sample_title', '')}\""
        if s.get("source_name"):
            sample_text += f", source=\"{s['source_name']}\""
        if s.get("characteristics"):
            chars = s["characteristics"]
            if isinstance(chars, str):
                try:
                    chars = json.loads(chars)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(chars, dict):
                chars_str = ", ".join(f"{k}={v}" for k, v in chars.items())
                sample_text += f", characteristics=[{chars_str}]"
        sample_text += "\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{stage_summary}\n{sample_text}"},
    ]

    resp = await client.complete(
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=4000,
    )

    if resp is None:
        logger.warning("LLM returned None for sample mapping (possible timeout)")
        return []

    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON for sample mapping")
        return []

    mappings: list[SampleStageMapping] = []
    for m in data.get("mappings", []):
        mappings.append(SampleStageMapping(
            gsm_id=m["gsm_id"],
            stage_name=_stage_name_for_number(m.get("stage_number"), stage_ranges),
            stage_number=m.get("stage_number"),
            time_point_day=m.get("time_point_day"),
            condition_label=m.get("condition_label", ""),
            mapping_confidence=m.get("confidence", 0.6),
            mapping_method="tier2_llm",
        ))

    return mappings


def _stage_name_for_number(stage_num: int | None, stage_ranges: list[StageRange]) -> str:
    """Look up stage name by number."""
    if stage_num is None:
        return ""
    for sr in stage_ranges:
        if sr.stage_number == stage_num:
            return sr.stage_name
    return ""


# ------------------------------------------------------------------
# Full mapping pipeline
# ------------------------------------------------------------------

def map_samples_to_stages(
    samples: list[dict],
    protocol_stages: list[dict],
    client=None,
) -> list[SampleStageMapping]:
    """Map GEO samples to protocol stages using Tier 1 + optional Tier 2.

    Args:
        samples: List of dicts with gsm_id, sample_title, source_name, characteristics
        protocol_stages: List of stage dicts from the protocol record
        client: Optional OpenRouterClient for Tier 2 LLM fallback

    Returns list of SampleStageMapping objects.
    """
    stage_ranges = _build_stage_ranges(protocol_stages)
    if not stage_ranges:
        logger.warning("No stage ranges could be built from protocol")
        return []

    # Tier 1
    mapped, unmapped = tier1_map_all(samples, stage_ranges)
    logger.info("Tier 1 mapped %d/%d samples", len(mapped), len(samples))

    # Tier 2 for unmapped samples (if client provided)
    if unmapped and client:
        logger.info("Running Tier 2 LLM mapping for %d unmapped samples", len(unmapped))
        tier2_results = asyncio.run(tier2_map_samples(client, unmapped, stage_ranges))
        mapped.extend(tier2_results)
        logger.info("Tier 2 mapped %d additional samples", len(tier2_results))

    return mapped


async def map_samples_to_stages_async(
    samples: list[dict],
    protocol_stages: list[dict],
    client=None,
) -> list[SampleStageMapping]:
    """Async version of map_samples_to_stages."""
    stage_ranges = _build_stage_ranges(protocol_stages)
    if not stage_ranges:
        logger.warning("No stage ranges could be built from protocol")
        return []

    mapped, unmapped = tier1_map_all(samples, stage_ranges)
    logger.info("Tier 1 mapped %d/%d samples", len(mapped), len(samples))

    if unmapped and client:
        logger.info("Running Tier 2 LLM mapping for %d unmapped samples", len(unmapped))
        tier2_results = await tier2_map_samples(client, unmapped, stage_ranges)
        mapped.extend(tier2_results)
        logger.info("Tier 2 mapped %d additional samples", len(tier2_results))

    return mapped


# ------------------------------------------------------------------
# Batch runner
# ------------------------------------------------------------------

def map_all_papers(db, client=None, limit: int | None = None) -> int:
    """Run sample-to-stage mapping for all eligible papers. Returns count mapped.

    Uses a single event loop for all Tier 2 LLM calls to avoid
    asyncio.run() creating/closing loops repeatedly.
    """
    return asyncio.run(_map_all_papers_async(db, client=client, limit=limit))


async def _map_all_papers_async(db, client=None, limit: int | None = None) -> int:
    """Async implementation of map_all_papers."""
    papers = db.get_papers_needing_geo_mapping()
    if limit:
        papers = papers[:limit]

    if not papers:
        logger.info("No papers need GEO sample mapping")
        return 0

    logger.info("Running sample mapping on %d papers", len(papers))
    total_mapped = 0

    for i, paper in enumerate(papers):
        pmc_id = paper["pmc_id"]
        paper_id = paper["id"]

        # Get protocols for this paper
        protocols = db.get_protocols_for_paper(paper_id)
        if not protocols:
            continue

        # Get GEO accessions with own_data context
        accessions = db.get_geo_accessions(paper_id)
        own_data_accs = [a for a in accessions if a.get("context") in ("own_data", "ambiguous")]

        for acc in own_data_accs:
            acc_id = acc["id"]
            samples = db.get_geo_samples(acc_id)
            if not samples:
                continue

            # Map samples against each protocol
            for proto in protocols:
                stages = proto.get("stages") or []
                if not stages:
                    continue

                mappings = await map_samples_to_stages_async(
                    samples=samples,
                    protocol_stages=stages,
                    client=client,
                )

                for m in mappings:
                    # Find the geo_sample_id for this GSM
                    sample_row = next(
                        (s for s in samples if s["gsm_id"] == m.gsm_id), None
                    )
                    if not sample_row:
                        continue

                    db.store_sample_stage_mapping({
                        "geo_sample_id": sample_row["id"],
                        "protocol_id": proto["id"],
                        "stage_name": m.stage_name,
                        "stage_number": m.stage_number,
                        "time_point_day": m.time_point_day,
                        "condition_label": m.condition_label,
                        "mapping_confidence": m.mapping_confidence,
                        "mapping_method": m.mapping_method,
                    })
                    total_mapped += 1

        logger.info("[%d/%d] %s: mapped samples for %d protocol(s)",
                    i + 1, len(papers), pmc_id, len(protocols))

    logger.info("Sample mapping complete: %d total mappings", total_mapped)
    return total_mapped
