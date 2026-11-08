"""Unified prompt template shared by SID, VERL rollouts, evaluation and demo."""

from __future__ import annotations

import json
from typing import Any, Iterable

SYSTEM_PROMPT = """You are an e-commerce sequential recommendation model.
Infer the user's next-item preference from the chronological interaction history and rank only items from the candidate set.
Return exactly one valid JSON object and no markdown or explanation.
The JSON schema is: {{"recommendations":["ITEM_ID_1","ITEM_ID_2",...]}}
Rules: output exactly {top_k} distinct {identifier_name} in descending preference order; never invent an item_id.
Use each bare identifier only (for example "B001ABC123" or "S030712"), never include labels such as "id=" or "item_id=" inside a JSON string."""


def _clean(value: Any, max_chars: int) -> str:
    """Normalize metadata to one compact prompt line."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " > ".join(str(part) for part in value if part)
    text = " ".join(str(value).replace("\n", " ").split())
    return text[:max_chars]


def format_item(item: dict[str, Any], index: int, title_max_chars: int = 72) -> str:
    """Render one item with stable, compact metadata."""
    item_id = _clean(item.get("item_id", item.get("asin", "")), 64)
    title = _clean(item.get("title", "Unknown title"), title_max_chars)
    category = _clean(item.get("category", item.get("categories", "Unknown")), 48)
    return f"{index}. item_id: {item_id} | title: {title or 'Unknown title'} | category: {category or 'Unknown'}"


def build_user_prompt(
    history: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    top_k: int = 10,
    title_max_chars: int = 72,
    identifier_name: str = "item IDs",
) -> str:
    """Build the single user turn used in both training stages."""
    history_lines = [format_item(item, idx, title_max_chars) for idx, item in enumerate(history, start=1)]
    candidate_lines = [format_item(item, idx, title_max_chars) for idx, item in enumerate(candidates, start=1)]
    return (
        "The interaction history is ordered from oldest to newest.\n"
        "[HISTORY]\n"
        + "\n".join(history_lines)
        + "\n[/HISTORY]\n"
        "[CANDIDATES]\n"
        + "\n".join(candidate_lines)
        + "\n[/CANDIDATES]\n"
        f"Recommend the top {top_k} candidate {identifier_name} most likely to be interacted with next."
    )


def build_prompt_messages(
    history: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    top_k: int = 10,
    title_max_chars: int = 72,
    identifier_name: str = "item IDs",
) -> list[dict[str, str]]:
    """Return chat messages without the assistant response."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(top_k=top_k, identifier_name=identifier_name),
        },
        {
            "role": "user",
            "content": build_user_prompt(
                history,
                candidates,
                top_k=top_k,
                title_max_chars=title_max_chars,
                identifier_name=identifier_name,
            ),
        },
    ]


def build_target_json(ranked_item_ids: Iterable[str], top_k: int = 10) -> str:
    """Create canonical compact supervision JSON."""
    unique_ids: list[str] = []
    seen: set[str] = set()
    for item_id in ranked_item_ids:
        item_id = str(item_id)
        if item_id and item_id not in seen:
            unique_ids.append(item_id)
            seen.add(item_id)
        if len(unique_ids) == top_k:
            break
    payload = {"recommendations": unique_ids}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_sft_messages(prompt_messages: list[dict[str, str]], target_json: str) -> list[dict[str, str]]:
    """Attach the assistant target to the unified prompt."""
    return [*prompt_messages, {"role": "assistant", "content": target_json}]
