"""Click ranking on *logged* MIND impression candidates only.

The standard-library pairwise model is a runnable baseline. LightGBM is an
optional server-side backend. Neither backend invents labels for unexposed
catalog items.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .content import ContentIndex, tokenize
from .data import Impression, NewsArticle


FEATURE_NAMES = (
    "content_similarity",
    "category_affinity",
    "title_overlap",
    "log_history_length",
    "generation_reciprocal_rank",
    "generation_present",
)


@dataclass(frozen=True)
class RankRow:
    impression_id: str
    news_id: str
    label: int
    features: Tuple[float, ...]


def build_rank_rows(
    impressions: Iterable[Impression],
    news: Mapping[str, NewsArticle],
    index: ContentIndex,
    generation: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[RankRow]:
    """Build features from pre-impression histories and candidate metadata.

    `generation` is an impression -> ranked news ID map. Generation scores are
    converted to reciprocal ranks so different requests share a stable scale.
    """
    rows: List[RankRow] = []
    generation = generation or {}
    for impression in impressions:
        history = [news_id for news_id in impression.history if news_id in news]
        profile = index.profile(history)
        category_counts = Counter(news[item].category for item in history)
        history_tokens = set()
        for item in history[-20:]:
            history_tokens.update(tokenize(news[item].title))
        generated_rank = {item: rank for rank, item in enumerate(generation.get(impression.impression_id, ()), 1)}
        for candidate in impression.candidates:
            article = news.get(candidate.news_id)
            if article is None:
                raise KeyError(f"missing metadata for exposed news {candidate.news_id}")
            vector = index.vectors.get(candidate.news_id, {})
            similarity = sum(value * vector.get(term, 0.0) for term, value in profile.items())
            affinity = category_counts[article.category] / len(history) if history else 0.0
            title_tokens = set(tokenize(article.title))
            overlap = len(title_tokens & history_tokens) / len(title_tokens) if title_tokens else 0.0
            rank = generated_rank.get(candidate.news_id)
            rows.append(
                RankRow(
                    impression.impression_id,
                    candidate.news_id,
                    candidate.label,
                    (
                        similarity,
                        affinity,
                        overlap,
                        math.log1p(len(history)),
                        1.0 / rank if rank else 0.0,
                        1.0 if rank else 0.0,
                    ),
                )
            )
    return rows


class PairwiseRanker:
    """Small deterministic pairwise logistic ranker for CPU smoke runs."""

    def __init__(self, weights: Optional[Sequence[float]] = None) -> None:
        self.weights = list(weights) if weights is not None else [0.0] * len(FEATURE_NAMES)
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError("wrong number of ranking weights")

    def fit(
        self,
        rows: Sequence[RankRow],
        epochs: int = 10,
        learning_rate: float = 0.1,
        l2: float = 1e-4,
        max_pairs_per_impression: int = 64,
        seed: int = 42,
    ) -> "PairwiseRanker":
        if epochs < 1 or learning_rate <= 0 or max_pairs_per_impression < 1:
            raise ValueError("invalid training hyperparameters")
        groups: Dict[str, List[RankRow]] = defaultdict(list)
        for row in rows:
            groups[row.impression_id].append(row)
        rng = random.Random(seed)
        grouped_pairs: List[List[Tuple[Tuple[float, ...], Tuple[float, ...]]]] = []
        for group in groups.values():
            positives = [row for row in group if row.label == 1]
            negatives = [row for row in group if row.label == 0]
            if not positives or not negatives:
                continue
            pairs = [(p.features, n.features) for p in positives for n in negatives]
            if len(pairs) > max_pairs_per_impression:
                pairs = rng.sample(pairs, max_pairs_per_impression)
            grouped_pairs.append(pairs)
        if not grouped_pairs:
            raise ValueError("ranker needs at least one impression with both clicked and unclicked candidates")
        for _ in range(epochs):
            rng.shuffle(grouped_pairs)
            for pairs in grouped_pairs:
                rng.shuffle(pairs)
                for positive, negative in pairs:
                    diff = [p - n for p, n in zip(positive, negative)]
                    margin = sum(weight * value for weight, value in zip(self.weights, diff))
                    gradient_scale = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, margin))))
                    for i, value in enumerate(diff):
                        self.weights[i] += learning_rate * (gradient_scale * value - l2 * self.weights[i])
        return self

    def predict(self, rows: Sequence[RankRow]) -> List[float]:
        return [sum(weight * value for weight, value in zip(self.weights, row.features)) for row in rows]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"feature_names": FEATURE_NAMES, "weights": self.weights}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseRanker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("model feature schema differs from this version")
        return cls(payload["weights"])


def fit_lightgbm(rows: Sequence[RankRow], seed: int = 42):
    """Optional LambdaRank backend for a multi-GPU/server environment."""
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as error:
        raise RuntimeError("Install the optional 'lightgbm' package to use this backend") from error
    groups: Dict[str, List[RankRow]] = defaultdict(list)
    for row in rows:
        groups[row.impression_id].append(row)
    ordered = [row for group in groups.values() for row in group]
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05, random_state=seed, verbosity=-1)
    model.fit([row.features for row in ordered], [row.label for row in ordered], group=[len(group) for group in groups.values()])
    return model

