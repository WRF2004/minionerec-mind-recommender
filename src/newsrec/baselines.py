"""Training-period-only popularity and item co-occurrence retrieval baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import math
from typing import Dict, Iterable, List, Sequence

from .content import CandidateScore
from .data import Impression


class PopularityIndex:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def fit(self, impressions: Iterable[Impression]) -> "PopularityIndex":
        self.counts = Counter(candidate.news_id for impression in impressions for candidate in impression.candidates if candidate.label == 1)
        return self

    def search(self, history: Sequence[str], k: int) -> List[CandidateScore]:
        seen = set(history)
        best = heapq.nlargest(k, ((news_id, count) for news_id, count in self.counts.items() if news_id not in seen), key=lambda pair: (pair[1], pair[0]))
        return [CandidateScore(news_id, float(count), "popularity") for news_id, count in best]


class ItemCFIndex:
    """Co-click evidence from earlier histories to training-period positives."""

    def __init__(self, max_history: int = 20) -> None:
        self.max_history = max_history
        self.neighbors: Dict[str, Dict[str, float]] = {}

    def fit(self, impressions: Iterable[Impression]) -> "ItemCFIndex":
        neighbors: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for impression in impressions:
            history = impression.history[-self.max_history:]
            positives = {candidate.news_id for candidate in impression.candidates if candidate.label == 1}
            if not history or not positives:
                continue
            normalization = 1.0 / math.sqrt(len(history))
            for age, source_id in enumerate(reversed(history)):
                for target_id in positives:
                    if source_id != target_id:
                        neighbors[source_id][target_id] += normalization / math.sqrt(age + 1)
        self.neighbors = {key: dict(value) for key, value in neighbors.items()}
        return self

    def search(self, history: Sequence[str], k: int) -> List[CandidateScore]:
        seen = set(history)
        scores: Dict[str, float] = defaultdict(float)
        for age, source_id in enumerate(reversed(history[-self.max_history:])):
            for target_id, weight in self.neighbors.get(source_id, {}).items():
                if target_id not in seen:
                    scores[target_id] += weight / math.sqrt(age + 1)
        best = heapq.nlargest(k, scores.items(), key=lambda pair: (pair[1], pair[0]))
        return [CandidateScore(news_id, score, "itemcf") for news_id, score in best]
