"""Exposure-aware recommendation metrics.

Ranking metrics use only news IDs that appeared in a logged MIND impression.
Full-catalog retrieval is evaluated separately against held-out clicked IDs;
catalog items absent from an impression are never assigned negative labels.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable, Mapping, Optional, Sequence, Tuple


def _validated_ks(ks: Sequence[int]) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(ks))
    if not values or any(not isinstance(k, int) or k < 1 for k in values):
        raise ValueError("ks must contain positive integers")
    return values


def _auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _ranked_labels(labels: Sequence[int], scores: Sequence[float]) -> list[int]:
    # Python's stable sort retains the logged candidate order for score ties.
    return [label for _, label in sorted(zip(scores, labels), key=lambda pair: -pair[0])]


def _mrr(ranked_labels: Sequence[int]) -> float:
    for rank, label in enumerate(ranked_labels, 1):
        if label:
            return 1.0 / rank
    return 0.0


def _ndcg(ranked_labels: Sequence[int], k: int) -> float:
    relevant = sum(ranked_labels)
    if relevant == 0:
        return 0.0
    dcg = sum(label / math.log2(rank + 1) for rank, label in enumerate(ranked_labels[:k], 1))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, relevant) + 1))
    return dcg / ideal


def ranking_metrics(
    records: Iterable[Tuple[str, int, float]], ks: Sequence[int] = (5, 10)
) -> dict[str, float | int | None]:
    """Macro-average AUC, MRR and nDCG over logged impression groups.

    Each record is ``(impression_id, observed_click_label, model_score)``.
    AUC excludes groups with only one label class and reports their count.
    MRR/nDCG assign zero to impressions without a click, retaining them in
    the denominator. Score ties preserve the logged candidate order.
    """

    cutoffs = _validated_ks(ks)
    grouped: OrderedDict[str, list[tuple[int, float]]] = OrderedDict()
    for impression_id, label, score in records:
        if label not in (0, 1):
            raise ValueError(f"click label must be 0 or 1, got {label!r}")
        if not math.isfinite(score):
            raise ValueError(f"score must be finite, got {score!r}")
        grouped.setdefault(impression_id, []).append((label, score))

    auc_values: list[float] = []
    mrr_sum = 0.0
    ndcg_sums = {k: 0.0 for k in cutoffs}
    positive_impressions = 0
    for pairs in grouped.values():
        labels = [label for label, _ in pairs]
        scores = [score for _, score in pairs]
        auc = _auc(labels, scores)
        if auc is not None:
            auc_values.append(auc)
        ranked = _ranked_labels(labels, scores)
        mrr_sum += _mrr(ranked)
        positive_impressions += int(any(labels))
        for k in cutoffs:
            ndcg_sums[k] += _ndcg(ranked, k)

    count = len(grouped)
    result: dict[str, float | int | None] = {
        "impressions": count,
        "positive_impressions": positive_impressions,
        "auc_impressions": len(auc_values),
        "auc": sum(auc_values) / len(auc_values) if auc_values else None,
        "mrr": mrr_sum / count if count else None,
    }
    result.update({f"ndcg@{k}": ndcg_sums[k] / count if count else None for k in cutoffs})
    return result


def recall_at_k(
    predictions: Mapping[str, Sequence[str]],
    clicked: Mapping[str, Iterable[str]],
    ks: Sequence[int] = (10, 50, 100),
) -> dict[str, float | int | None]:
    """Observed-click full-catalog Recall@K for held-out impressions.

    ``predictions`` contains ordered catalog IDs returned by a retriever.
    ``clicked`` contains only positively clicked IDs from held-out logged
    impressions. Unexposed catalog items have unknown relevance. Queries
    without observed clicks are omitted because recall is undefined for them.
    """

    cutoffs = _validated_ks(ks)
    totals = {k: 0.0 for k in cutoffs}
    queries = 0
    for impression_id, truth in clicked.items():
        relevant = set(truth)
        if not relevant:
            continue
        ranked = list(dict.fromkeys(predictions.get(impression_id, ())))
        queries += 1
        for k in cutoffs:
            totals[k] += len(relevant.intersection(ranked[:k])) / len(relevant)
    result: dict[str, float | int | None] = {"queries": queries}
    result.update({f"recall@{k}": totals[k] / queries if queries else None for k in cutoffs})
    return result
