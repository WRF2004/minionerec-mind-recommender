"""A small, dependency-free content retrieval baseline for MIND news.

The index is intentionally simple: sparse TF-IDF over titles and abstracts.
It provides a reproducible full-catalog baseline and a candidate source for
fusion. Production scale systems should replace this with an ANN service.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import math
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it of on or that the "
    "this to was were will with after before about into over under says said".split()
)


@dataclass(frozen=True)
class CandidateScore:
    news_id: str
    score: float
    source: str


def tokenize(text: str) -> List[str]:
    return [token for token in _TOKEN.findall((text or "").lower()) if token not in _STOP]


def _article_text(article: object) -> str:
    if isinstance(article, Mapping):
        title = article.get("title", "")
        abstract = article.get("abstract", "")
    else:
        title = getattr(article, "title", "")
        abstract = getattr(article, "abstract", "")
    return f"{title or ''} {abstract or ''}"


def _unit_vector(weights: Mapping[str, float]) -> Dict[str, float]:
    norm = math.sqrt(sum(value * value for value in weights.values()))
    if not norm:
        return {}
    return {key: value / norm for key, value in weights.items()}


class ContentIndex:
    """Sparse title/abstract retriever with recency-weighted user profiles."""

    def __init__(self, recent_decay: float = 0.9) -> None:
        if not 0 < recent_decay <= 1:
            raise ValueError("recent_decay must be in (0, 1]")
        self.recent_decay = recent_decay
        self.vectors: Dict[str, Dict[str, float]] = {}
        self.postings: Dict[str, List[Tuple[str, float]]] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, news: Mapping[str, object]) -> "ContentIndex":
        terms_by_news = {news_id: Counter(tokenize(_article_text(article))) for news_id, article in news.items()}
        df: Counter[str] = Counter()
        for counts in terms_by_news.values():
            df.update(counts.keys())
        total = len(terms_by_news)
        self.idf = {term: math.log((total + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
        self.vectors = {}
        postings: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for news_id, counts in terms_by_news.items():
            vector = _unit_vector({term: (1.0 + math.log(freq)) * self.idf[term] for term, freq in counts.items()})
            self.vectors[news_id] = vector
            for term, weight in vector.items():
                postings[term].append((news_id, weight))
        self.postings = dict(postings)
        return self

    def profile(self, history: Sequence[str]) -> Dict[str, float]:
        profile: Dict[str, float] = defaultdict(float)
        for age, news_id in enumerate(reversed(history)):
            vector = self.vectors.get(news_id)
            if vector is None:
                continue
            weight = self.recent_decay ** age
            for term, value in vector.items():
                profile[term] += weight * value
        return _unit_vector(profile)

    def score(self, history: Sequence[str], news_id: str) -> float:
        profile = self.profile(history)
        return sum(weight * self.vectors.get(news_id, {}).get(term, 0.0) for term, weight in profile.items())

    def search(self, history: Sequence[str], k: int, exclude_seen: bool = True) -> List[CandidateScore]:
        if k <= 0:
            return []
        profile = self.profile(history)
        if not profile:
            return []
        scores: Dict[str, float] = defaultdict(float)
        for term, user_weight in profile.items():
            for news_id, item_weight in self.postings.get(term, ()):
                scores[news_id] += user_weight * item_weight
        if exclude_seen:
            for news_id in history:
                scores.pop(news_id, None)
        best = heapq.nlargest(k, scores.items(), key=lambda pair: (pair[1], pair[0]))
        return [CandidateScore(news_id, float(score), "content") for news_id, score in best]

