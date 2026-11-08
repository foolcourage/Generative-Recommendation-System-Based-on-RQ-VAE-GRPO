import json

from llm4rec.prompts import build_target_json
from llm4rec.training.rewards import compute_score


def truth():
    return json.dumps({"target_item_id": "I0", "candidate_ids": [f"I{i}" for i in range(12)], "top_k": 10})


def test_good_ranking_has_max_reward():
    score = compute_score(build_target_json([f"I{i}" for i in range(10)]), truth())
    assert score["hr"] == 1.0
    assert score["ndcg"] == 1.0
    assert score["score"] == 1.0


def test_invalid_json_is_negative():
    score = compute_score("I0, I1", truth())
    assert score["score"] == -1.0


def test_late_hit_has_continuous_reward():
    ids = [f"I{i}" for i in range(1, 10)] + ["I0"]
    score = compute_score(build_target_json(ids), truth())
    assert score["hr"] == 1.0
    assert 0.0 < score["ndcg"] < 1.0
    assert 0.0 < score["score"] < 1.0


def test_label_prefixed_candidate_id_is_not_falsely_penalized():
    answer = build_target_json([f"id=I{i}" for i in range(10)])
    score = compute_score(answer, truth())
    assert score["hr"] == 1.0
    assert score["unknown_item_cost"] == 0.0
    assert score["format_reward"] > 0.0


def test_semantic_prefix_reward_is_continuous():
    candidates = [
        "S010203",
        "S010204",
        "S010506",
        *[f"S{index:02d}0000" for index in range(7, 14)],
    ]
    semantic_truth = json.dumps(
        {
            "target_item_id": candidates[0],
            "candidate_ids": candidates,
            "top_k": 10,
            "identifier_mode": "semantic_id",
            "semantic_prefix_weight": 0.05,
        }
    )
    answer = build_target_json([candidates[1], *candidates[2:10]])
    score = compute_score(answer, semantic_truth)
    assert score["hr"] == 0.0
    assert score["semantic_prefix_reward"] > 0.0
