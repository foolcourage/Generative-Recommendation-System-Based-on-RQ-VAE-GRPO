"""Strict parsing and validation for structured recommendation output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class ParseResult:
    valid: bool
    item_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    payload: dict[str, Any] | None = None


_ITEM_ID_PREFIX = re.compile(r"^(?:item[_\s-]*id|asin|id)\s*(?:[:=]\s*|\s+)", re.IGNORECASE)


def normalize_item_id(value: Any) -> str:
    """Normalize common label-prefixed IDs without changing the identifier.

    Candidate metadata is commonly rendered as ``id=B001...``. Small models
    sometimes copy that label into the JSON string even though the intended
    identifier is the ASIN. Accepting this deterministic presentation variant
    prevents false out-of-candidate penalties while JSON/schema checks remain
    strict.
    """
    text = str(value).strip()
    return _ITEM_ID_PREFIX.sub("", text, count=1).strip()


def parse_recommendation_json(
    text: str,
    expected_k: int | None = 10,
    allowed_item_ids: Iterable[str] | None = None,
    strict_exact_json: bool = True,
) -> ParseResult:
    """Parse model output and enforce the project JSON contract.

    `strict_exact_json=True` rejects markdown fences and leading/trailing prose. This
    is used by GRPO rewards so the learned behavior matches production parsing.
    """
    errors: list[str] = []
    stripped = (text or "").strip()
    if not stripped:
        return ParseResult(valid=False, errors=["empty_output"])

    if strict_exact_json and (not stripped.startswith("{") or not stripped.endswith("}")):
        return ParseResult(valid=False, errors=["not_exact_json_object"])

    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return ParseResult(valid=False, errors=["invalid_json"])

    if not isinstance(payload, dict) or set(payload) != {"recommendations"}:
        return ParseResult(valid=False, errors=["invalid_top_level_schema"], payload=payload if isinstance(payload, dict) else None)

    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list):
        return ParseResult(valid=False, errors=["recommendations_not_list"], payload=payload)

    if expected_k is not None and len(recommendations) != expected_k:
        errors.append("wrong_list_length")

    item_ids: list[str] = []
    for entry in recommendations:
        if not isinstance(entry, str) or not entry.strip():
            errors.append("invalid_item_id")
            continue
        normalized = normalize_item_id(entry)
        if not normalized:
            errors.append("invalid_item_id")
            continue
        item_ids.append(normalized)
    if len(item_ids) != len(set(item_ids)):
        errors.append("duplicate_item_ids")

    if allowed_item_ids is not None:
        allowed = {normalize_item_id(item_id) for item_id in allowed_item_ids}
        if any(item_id not in allowed for item_id in item_ids):
            errors.append("out_of_candidate_item")

    return ParseResult(valid=not errors, item_ids=item_ids, errors=sorted(set(errors)), payload=payload)


def candidate_ids_from_record(record: dict[str, Any]) -> list[str]:
    """Read candidate IDs from either raw evaluation records or VERL extra_info."""
    candidates = record.get("candidates", [])
    return [normalize_item_id(item.get("item_id", item)) if isinstance(item, dict) else normalize_item_id(item) for item in candidates]
