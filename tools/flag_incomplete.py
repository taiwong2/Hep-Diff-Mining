"""Tool: flag_incomplete — structured incompleteness flagging.

When the LLM cannot fully extract a protocol field, it calls this tool
to record the reason. Flags are collected and stored on the protocol record.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

VALID_REASONS = {
    "not_reported",
    "behind_paywall_reference",
    "ambiguous_text",
    "in_supplement_not_available",
    "kit_composition_unknown",
    "qualitative_only",
}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "flag_incomplete",
        "description": (
            "Flag a protocol field that could not be fully extracted. "
            "Use this when information is missing, ambiguous, or only available "
            "in an unfetchable reference. The flag will be recorded on the "
            "protocol for manual review."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": (
                        "The protocol field that is incomplete. Examples: "
                        "'de_stage_growth_factors', 'maturation_duration', "
                        "'seeding_density', 'cyp3a4_activity', 'base_protocol_details'"
                    ),
                },
                "reason": {
                    "type": "string",
                    "enum": list(VALID_REASONS),
                    "description": "Why the field could not be extracted.",
                },
                "details": {
                    "type": "string",
                    "description": (
                        "Additional context: which reference is behind a paywall, "
                        "what makes the text ambiguous, etc."
                    ),
                },
            },
            "required": ["field", "reason"],
        },
    },
}


def execute(db, args: dict) -> str:
    """Execute flag_incomplete tool. Returns JSON acknowledgment."""
    field_name = args.get("field", "").strip()
    reason = args.get("reason", "").strip()
    details = args.get("details", "").strip()

    if not field_name:
        return json.dumps({"error": "Missing 'field' parameter"})

    if reason not in VALID_REASONS:
        return json.dumps({
            "error": f"Invalid reason '{reason}'. Valid reasons: {sorted(VALID_REASONS)}"
        })

    flag = {
        "field": field_name,
        "reason": reason,
        "details": details,
    }

    return json.dumps({
        "status": "flagged",
        "flag": flag,
        "message": f"Recorded: '{field_name}' is incomplete ({reason})."
                   + (f" Details: {details}" if details else ""),
    })
