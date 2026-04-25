"""Agentic protocol extractor: multi-turn tool-calling LLM extraction.

Three-pass architecture:
- Pass 1: Structure identification (no tools) — identify protocol arms
- Pass 2: Detailed extraction (with tools) — extract full protocol per arm
- Pass 3: Supplement extraction (no tools, conditional) — augment from supplements

Usage:
    python -m llm.agents.agentic_extractor                    # full run
    python -m llm.agents.agentic_extractor --limit 5          # test
    python -m llm.agents.agentic_extractor --single PMC10114490
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from llm.openrouter.client import OpenRouterClient, APIError
from data_layer.database import PipelineDB
from data_layer.grounding import ground_protocol
from tools import ALL_TOOL_SCHEMAS, TOOL_DISPATCH

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "openai/gpt-4o-mini"
MAX_TOOL_CALLS_PER_PAPER = 5
MAX_CONVERSATION_TURNS = 10

PROMPT_DIR = Path(__file__).parent / "prompts"
PASS1_PROMPT = PROMPT_DIR / "extraction_pass1.txt"
PASS2_PROMPT = PROMPT_DIR / "extraction_pass2.txt"
PASS3_PROMPT = PROMPT_DIR / "extraction_pass3.txt"

DEFAULT_OUTPUT = Path("data/results/extraction_results.jsonl")


def _load_prompt(path: Path) -> str:
    return path.read_text().strip()


# ------------------------------------------------------------------
# Pass 1: Structure identification
# ------------------------------------------------------------------

async def run_pass1(
    client: OpenRouterClient,
    paper_text: str,
    title: str,
) -> dict | None:
    """Identify protocol structure: arms, stages, outcome locations."""
    system_prompt = _load_prompt(PASS1_PROMPT)
    user_message = f"Paper title: {title}\n\n---\n\n{paper_text}"

    try:
        resp = await client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=EXTRACTION_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp["choices"][0]["message"]["content"]
        return json.loads(content)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Pass 1 parse error: %s", e)
        return None
    except APIError as e:
        logger.warning("Pass 1 API error: %s", e)
        return None


# ------------------------------------------------------------------
# Pass 2: Detailed extraction with tool calling
# ------------------------------------------------------------------

async def run_pass2(
    client: OpenRouterClient,
    db: PipelineDB,
    paper_text: str,
    title: str,
    arm: dict,
    pass1_result: dict,
) -> tuple[dict | None, list[dict], int]:
    """Extract full protocol for one arm with tool calling.

    Returns (protocol_dict, incomplete_flags, tokens_used).
    """
    system_prompt = _load_prompt(PASS2_PROMPT)

    arm_desc = arm.get("arm_description", "main protocol")
    arm_id = arm.get("arm_id", "arm_1")
    base_ref = arm.get("base_protocol_referenced")

    user_content = (
        f"Paper title: {title}\n"
        f"Protocol arm to extract: {arm_id} — {arm_desc}\n"
        f"Paper structure (from Pass 1): {json.dumps(pass1_result, indent=2)}\n"
        f"\n---\n\n{paper_text}"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    tool_calls_made = 0
    incomplete_flags: list[dict] = []
    tokens_used = 0

    for turn in range(MAX_CONVERSATION_TURNS):
        try:
            resp = await client.complete(
                messages=messages,
                model=EXTRACTION_MODEL,
                temperature=0,
                tools=ALL_TOOL_SCHEMAS if tool_calls_made < MAX_TOOL_CALLS_PER_PAPER else None,
                response_format={"type": "json_object"} if tool_calls_made >= MAX_TOOL_CALLS_PER_PAPER else None,
            )
        except APIError as e:
            logger.warning("Pass 2 API error on turn %d: %s", turn, e)
            break

        # Track tokens
        usage = resp.get("usage", {})
        tokens_used += usage.get("total_tokens", 0)

        choice = resp["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "")

        # Check for tool calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls and tool_calls_made < MAX_TOOL_CALLS_PER_PAPER:
            # Add assistant message with tool calls
            messages.append(message)

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                tool_calls_made += 1
                logger.info("  Tool call %d: %s(%s)", tool_calls_made,
                            func_name, json.dumps(func_args)[:100])

                # Execute tool
                if func_name in TOOL_DISPATCH:
                    result_str = TOOL_DISPATCH[func_name](db, func_args)
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {func_name}"})

                # Collect incomplete flags
                if func_name == "flag_incomplete":
                    try:
                        flag_result = json.loads(result_str)
                        if flag_result.get("flag"):
                            incomplete_flags.append(flag_result["flag"])
                    except json.JSONDecodeError:
                        pass

                # Add tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

            continue

        # No tool calls — check if we got the final response
        content = message.get("content", "")
        if content:
            try:
                protocol = json.loads(content)
                protocol["incomplete_flags"] = incomplete_flags
                protocol["protocol_arm"] = arm_id
                return (protocol, incomplete_flags, tokens_used)
            except json.JSONDecodeError:
                # Try to extract JSON from the content
                protocol = _extract_json_from_text(content)
                if protocol:
                    protocol["incomplete_flags"] = incomplete_flags
                    protocol["protocol_arm"] = arm_id
                    return (protocol, incomplete_flags, tokens_used)
                logger.warning("Pass 2 final response not valid JSON")

        break

    return (None, incomplete_flags, tokens_used)


# ------------------------------------------------------------------
# Pass 3: Supplement extraction
# ------------------------------------------------------------------

async def run_pass3(
    client: OpenRouterClient,
    supplement_text: str,
    pass2_protocol: dict,
    title: str,
) -> dict | None:
    """Extract additional data from supplementary materials."""
    system_prompt = _load_prompt(PASS3_PROMPT)

    # Summarize the existing protocol for context
    protocol_summary = json.dumps(pass2_protocol, indent=2)
    if len(protocol_summary) > 4000:
        # Truncate to key fields
        summary = {
            "protocol_arm": pass2_protocol.get("protocol_arm"),
            "stages": [
                {"stage_name": s.get("stage_name"), "duration_days": s.get("duration_days")}
                for s in pass2_protocol.get("stages", [])
                if isinstance(s, dict)
            ],
            "endpoint_markers": [
                m.get("marker_name") for m in
                (pass2_protocol.get("endpoint_assessment", {}) or {}).get("markers", [])
                if isinstance(m, dict)
            ],
        }
        protocol_summary = json.dumps(summary, indent=2)

    user_content = (
        f"Paper title: {title}\n\n"
        f"Already extracted protocol (Pass 2):\n{protocol_summary}\n\n"
        f"---\n\nSupplementary Materials:\n\n{supplement_text}"
    )

    try:
        resp = await client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model=EXTRACTION_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp["choices"][0]["message"]["content"]
        return json.loads(content)
    except (json.JSONDecodeError, KeyError, IndexError, APIError) as e:
        logger.warning("Pass 3 error: %s", e)
        return None


# ------------------------------------------------------------------
# Merge Pass 3 results into protocol
# ------------------------------------------------------------------

def merge_pass3(protocol: dict, pass3_result: dict) -> dict:
    """Merge Pass 3 supplement data into the Pass 2 protocol."""
    if not pass3_result or pass3_result.get("no_additional_data"):
        return protocol

    updates = pass3_result.get("updates_to_existing") or {}

    # Merge endpoint markers
    if "endpoint_assessment" in updates:
        ea = protocol.get("endpoint_assessment") or {}
        new_markers = (updates["endpoint_assessment"] or {}).get("markers") or []
        existing_markers = ea.get("markers") or []
        existing_names = {m.get("marker_name") for m in existing_markers if isinstance(m, dict)}
        for m in new_markers:
            if isinstance(m, dict) and m.get("marker_name") not in existing_names:
                existing_markers.append(m)
        ea["markers"] = existing_markers

        new_assays = (updates["endpoint_assessment"] or {}).get("functional_assays") or []
        existing_assays = ea.get("functional_assays") or []
        existing_assay_names = {a.get("assay_name") for a in existing_assays if isinstance(a, dict)}
        for a in new_assays:
            if isinstance(a, dict) and a.get("assay_name") not in existing_assay_names:
                existing_assays.append(a)
        ea["functional_assays"] = existing_assays
        protocol["endpoint_assessment"] = ea

    # Merge stage updates
    for stage_update in (updates.get("stage_updates") or []):
        if not isinstance(stage_update, dict):
            continue
        stage_name = stage_update.get("stage_name")
        for stage in (protocol.get("stages") or []):
            if isinstance(stage, dict) and stage.get("stage_name") == stage_name:
                existing = stage.get("stage_markers") or []
                existing_names = {m.get("marker_name") for m in existing if isinstance(m, dict)}
                for m in (stage_update.get("additional_markers") or []):
                    if isinstance(m, dict) and m.get("marker_name") not in existing_names:
                        existing.append(m)
                stage["stage_markers"] = existing

    # Note supplement extraction
    notes = protocol.get("extraction_notes", "") or ""
    pass3_notes = pass3_result.get("extraction_notes", "")
    if pass3_notes:
        protocol["extraction_notes"] = f"{notes} | Pass 3: {pass3_notes}".strip(" | ")

    protocol["pass_number"] = 3
    return protocol


# ------------------------------------------------------------------
# Single paper extraction
# ------------------------------------------------------------------

async def extract_paper(
    client: OpenRouterClient,
    db: PipelineDB,
    paper: dict,
) -> list[dict]:
    """Run full 3-pass extraction on a single paper.

    Returns list of extracted protocol dicts (one per arm).
    """
    paper_id = paper["id"]
    pmc_id = paper["pmc_id"]
    title = paper.get("title", "")

    # Read parsed text
    text_path = paper.get("parsed_text_path")
    if not text_path or not Path(text_path).exists():
        logger.warning("No parsed text for %s", pmc_id)
        return []

    paper_text = Path(text_path).read_text()

    # Pre-load supplement text for grounding (used after Pass 2 and Pass 3)
    supp_text_path = paper.get("supplement_text_path")
    supplement_text_for_grounding: str | None = None
    if supp_text_path and Path(supp_text_path).exists():
        _supp = Path(supp_text_path).read_text()
        if _supp.strip():
            supplement_text_for_grounding = _supp

    start_time = time.time()
    total_tokens = 0

    # --- Pass 1: Structure identification ---
    logger.info("[%s] Pass 1: identifying structure...", pmc_id)
    pass1 = await run_pass1(client, paper_text, title)
    if not pass1:
        logger.warning("[%s] Pass 1 failed", pmc_id)
        db.log_processing(paper_id, "pass1", "failed",
                          error_message="Pass 1 returned None")
        return []

    arms = pass1.get("protocol_arms", [])
    if not arms:
        # Default to single arm
        arms = [{"arm_id": "arm_1", "arm_description": "main protocol",
                 "is_optimized": True}]

    logger.info("[%s] Pass 1: found %d arm(s)", pmc_id, len(arms))

    # --- Pass 2: Detailed extraction per arm ---
    protocols: list[dict] = []
    for arm in arms:
        arm_id = arm.get("arm_id", "arm_1")
        logger.info("[%s] Pass 2: extracting %s...", pmc_id, arm_id)

        protocol, flags, tokens = await run_pass2(
            client, db, paper_text, title, arm, pass1,
        )
        total_tokens += tokens

        if protocol:
            protocol["is_optimized"] = arm.get("is_optimized", False)

            # Ground against source text (+ supplement if available)
            protocol, removals = ground_protocol(protocol, paper_text, supplement_text_for_grounding)
            if removals:
                removed_terms = [r["term"] for r in removals]
                logger.info("[%s] Pass 2 %s: grounding removed %d item(s): %s",
                            pmc_id, arm_id, len(removals), ", ".join(removed_terms))
                notes = protocol.get("extraction_notes", "") or ""
                note = f"Grounding removed: {', '.join(removed_terms)}"
                protocol["extraction_notes"] = f"{notes} | {note}".strip(" | ")

            protocols.append(protocol)
            logger.info("[%s] Pass 2 %s: confidence=%.2f, flags=%d",
                        pmc_id, arm_id,
                        protocol.get("extraction_confidence", 0),
                        len(flags))
        else:
            logger.warning("[%s] Pass 2 %s: extraction failed", pmc_id, arm_id)
            db.log_processing(paper_id, f"pass2_{arm_id}", "failed",
                              error_message="Pass 2 returned None")

    # --- Pass 3: Supplement extraction (conditional) ---
    triage = db.get_triage_result(paper_id)
    has_supp_signal = triage and triage.get("supplement_signal")

    if supplement_text_for_grounding and has_supp_signal:
        for i, protocol in enumerate(protocols):
            logger.info("[%s] Pass 3: supplement extraction for arm %d...",
                        pmc_id, i + 1)
            pass3 = await run_pass3(client, supplement_text_for_grounding, protocol, title)
            if pass3:
                merged = merge_pass3(protocol, pass3)
                # Re-ground merged protocol against paper + supplement text
                merged, removals = ground_protocol(merged, paper_text, supplement_text_for_grounding)
                if removals:
                    removed_terms = [r["term"] for r in removals]
                    logger.info("[%s] Pass 3 arm %d: grounding removed %d item(s): %s",
                                pmc_id, i + 1, len(removals), ", ".join(removed_terms))
                    notes = merged.get("extraction_notes", "") or ""
                    note = f"Pass 3 grounding removed: {', '.join(removed_terms)}"
                    merged["extraction_notes"] = f"{notes} | {note}".strip(" | ")
                protocols[i] = merged

    # Record timing
    duration = time.time() - start_time
    db.log_processing(paper_id, "extraction", "completed",
                      tokens_used=total_tokens, duration_secs=duration)

    return protocols


# ------------------------------------------------------------------
# Batch extraction
# ------------------------------------------------------------------

async def run_extraction(
    db: PipelineDB,
    output_file: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
    single: str | None = None,
) -> None:
    """Run extraction on all eligible papers."""
    if single:
        papers = []
        p = db.get_paper(pmc_id=single)
        if p:
            papers = [p]
        else:
            print(f"Paper {single} not found in DB")
            return
    else:
        papers = db.get_papers_for_extraction()

    if limit:
        papers = papers[:limit]

    if not papers:
        print("No papers ready for extraction.")
        return

    print(f"Extracting protocols from {len(papers)} papers...")

    async with OpenRouterClient(max_concurrent=3) as client:
        for i, paper in enumerate(papers):
            pmc_id = paper["pmc_id"]
            paper_id = paper["id"]

            print(f"[{i+1}/{len(papers)}] {pmc_id}: {paper.get('title', '')[:60]}...")

            db.set_extraction_status(paper_id, "in_progress")

            try:
                protocols = await extract_paper(client, db, paper)

                if protocols:
                    for protocol in protocols:
                        db.store_protocol(paper_id, protocol)

                        # Also write to JSONL
                        record = {
                            "pmc_id": pmc_id,
                            "doi": paper.get("doi"),
                            "title": paper.get("title"),
                            "protocol_arm": protocol.get("protocol_arm"),
                            "extraction_confidence": protocol.get("extraction_confidence"),
                            "stages_count": len(protocol.get("stages") or []),
                            "incomplete_flags": protocol.get("incomplete_flags", []),
                        }
                        _append_jsonl(output_file, record)

                    db.set_extraction_status(paper_id, "completed")
                    print(f"  → {len(protocols)} protocol(s) extracted")
                else:
                    db.set_extraction_status(paper_id, "failed")
                    print(f"  → extraction failed")

            except Exception as e:
                logger.exception("Error extracting %s", pmc_id)
                db.set_extraction_status(paper_id, "failed")
                db.log_processing(paper_id, "extraction", "failed",
                                  error_message=str(e))
                print(f"  → error: {e}")


def _extract_json_from_text(text: str) -> dict | None:
    """Try to extract JSON from text that may have markdown fencing."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    import re
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find the outermost { }
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return None


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract hepatocyte differentiation protocols via agentic LLM",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to extract")
    parser.add_argument("--single", type=str, default=None,
                        help="Extract single paper by PMC ID")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="JSONL output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db = PipelineDB()
    asyncio.run(run_extraction(db, output_file=args.output,
                               limit=args.limit, single=args.single))
    db.close()


if __name__ == "__main__":
    main()
