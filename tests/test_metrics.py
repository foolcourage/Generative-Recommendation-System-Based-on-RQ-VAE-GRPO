import math

from llm4rec.evaluation.metrics import hit_rate_at_k, ndcg_at_k, recall_at_k


def test_single_target_metrics():
    ranked = ["A", "B", "C"]
    assert recall_at_k(ranked, "B", 3) == 1.0
    assert hit_rate_at_k(ranked, "B", 3) == 1.0
    assert math.isclose(ndcg_at_k(ranked, "B", 3), 1 / math.log2(3))
    assert ndcg_at_k(ranked, "Z", 3) == 0.0

