import json

from llm4rec.prompts import build_prompt_messages, build_target_json
from llm4rec.schema import parse_recommendation_json


def test_prompt_and_target_contract():
    history = [{"item_id": "A", "title": "Old item", "category": "Beauty"}]
    candidates = [
        {"item_id": f"I{i}", "title": f"Item {i}", "category": "Beauty"} for i in range(10)
    ]
    messages = build_prompt_messages(history, candidates, top_k=10)
    assert messages[0]["role"] == "system"
    assert "exactly 10 distinct item IDs" in messages[0]["content"]
    target = build_target_json([f"I{i}" for i in range(10)], top_k=10)
    assert json.loads(target)["recommendations"][0] == "I0"
    assert parse_recommendation_json(target, 10, [f"I{i}" for i in range(10)]).valid


def test_markdown_is_rejected():
    result = parse_recommendation_json('```json\n{"recommendations":[]}\n```', expected_k=0)
    assert not result.valid


def test_common_item_id_prefix_is_normalized():
    text = '{"recommendations":["id=I0","item_id: I1","ASIN I2"]}'
    result = parse_recommendation_json(text, expected_k=3, allowed_item_ids=["I0", "I1", "I2"])
    assert result.valid
    assert result.item_ids == ["I0", "I1", "I2"]
