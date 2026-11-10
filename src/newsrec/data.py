"""MIND news and impression data, with chronology-preserving evaluation splits.

MIND's ``behaviors.tsv`` is an exposure log. Only candidates explicitly listed in
an impression have click labels; other catalog articles are *unlabeled*, not
negative examples. The user's history in each row predates that impression.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class NewsArticle:
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities: str
    abstract_entities: str


@dataclass(frozen=True)
class Candidate:
    news_id: str
    label: int


@dataclass(frozen=True)
class Impression:
    impression_id: str
    user_id: str
    timestamp: datetime
    history: tuple[str, ...]
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class RetrievalExample:
    """One clicked item, conditioned only on history before its impression."""

    impression_id: str
    user_id: str
    timestamp: datetime
    history: tuple[str, ...]
    positive_news_id: str


def _rows(path: str | Path) -> Iterator[tuple[int, list[str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if row and any(field.strip() for field in row):
                yield line_number, row


def _parse_timestamp(value: str) -> datetime:
    value = value.strip()
    for pattern in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"unsupported MIND behavior timestamp: {value!r}") from error
    # Aware ISO timestamps, if supplied by a derived dataset, are compared in UTC.
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _parse_news_row(row: list[str], path: str | Path, line_number: int) -> NewsArticle:
    if len(row) != 8:
        raise ValueError(f"{path}:{line_number}: expected 8 news columns, found {len(row)}")
    fields = [field.strip() for field in row]
    if not fields[0] or not fields[3]:
        raise ValueError(f"{path}:{line_number}: news ID and title are required")
    return NewsArticle(*fields)


def iter_news(path: str | Path) -> Iterator[NewsArticle]:
    """Yield articles from an official MIND ``news.tsv`` file."""

    for line_number, row in _rows(path):
        yield _parse_news_row(row, path, line_number)


def read_news(path: str | Path) -> dict[str, NewsArticle]:
    """Read a news catalog, rejecting conflicting duplicate identifiers."""

    catalog: dict[str, NewsArticle] = {}
    for article in iter_news(path):
        old = catalog.get(article.news_id)
        if old is not None and old != article:
            raise ValueError(f"{path}: conflicting metadata for news ID {article.news_id}")
        catalog[article.news_id] = article
    return catalog


def merge_news_catalogs(*catalogs: dict[str, NewsArticle]) -> dict[str, NewsArticle]:
    """Merge train and validation catalogs without silently changing metadata."""

    merged: dict[str, NewsArticle] = {}
    for catalog in catalogs:
        for news_id, article in catalog.items():
            old = merged.get(news_id)
            if old is not None and old != article:
                raise ValueError(f"conflicting metadata for news ID {news_id}")
            merged[news_id] = article
    return merged


def _parse_behavior_row(row: list[str], path: str | Path, line_number: int) -> Impression:
    if len(row) != 5:
        raise ValueError(f"{path}:{line_number}: expected 5 behavior columns, found {len(row)}")
    impression_id, user_id, time, history, impressions = (field.strip() for field in row)
    if not impression_id or not user_id:
        raise ValueError(f"{path}:{line_number}: impression ID and user ID are required")
    candidates: list[Candidate] = []
    for token in impressions.split():
        if "-" not in token:
            raise ValueError(
                f"{path}:{line_number}: unlabeled candidate {token!r}; "
                "MIND test impressions cannot be used for supervised evaluation"
            )
        news_id, label_text = token.rsplit("-", 1)
        if not news_id or label_text not in {"0", "1"}:
            raise ValueError(f"{path}:{line_number}: invalid candidate token {token!r}")
        candidates.append(Candidate(news_id, int(label_text)))
    if not candidates:
        raise ValueError(f"{path}:{line_number}: impression has no exposed candidates")
    try:
        timestamp = _parse_timestamp(time)
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: {error}") from error
    return Impression(
        impression_id=impression_id,
        user_id=user_id,
        timestamp=timestamp,
        history=tuple(history.split()),
        candidates=tuple(candidates),
    )


def iter_behaviors(path: str | Path) -> Iterator[Impression]:
    """Yield labeled impressions from an official MIND ``behaviors.tsv`` file."""

    for line_number, row in _rows(path):
        yield _parse_behavior_row(row, path, line_number)


def read_behaviors(path: str | Path) -> list[Impression]:
    return list(iter_behaviors(path))


def temporal_split(
    impressions: Iterable[Impression], val_fraction: float = 0.1
) -> tuple[list[Impression], list[Impression]]:
    """Split at a timestamp boundary, never placing equal times in both sets.

    Returns each partition in chronological order. The cutoff approximates
    ``val_fraction`` while keeping both partitions nonempty. At least two
    distinct timestamps are needed for a valid time split.
    """

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    ordered = sorted(impressions, key=lambda x: (x.timestamp, x.impression_id, x.user_id))
    if len(ordered) < 2:
        raise ValueError("at least two impressions are required for temporal split")
    boundaries = [i for i in range(1, len(ordered)) if ordered[i - 1].timestamp < ordered[i].timestamp]
    if not boundaries:
        raise ValueError("a temporal split requires at least two distinct timestamps")
    target_train_count = len(ordered) * (1 - val_fraction)
    split_at = min(boundaries, key=lambda i: (abs(i - target_train_count), -i))
    return ordered[:split_at], ordered[split_at:]


def build_retrieval_examples(
    impressions: Iterable[Impression],
    min_history: int = 1,
    exclude_seen_targets: bool = True,
) -> list[RetrievalExample]:
    """Build positive next-click targets from the *pre-impression* user history.

    Every clicked item in one impression gets the same pre-impression history;
    clicks within that impression have no known order. No synthetic negatives
    are created from the unexposed catalog.
    """

    if min_history < 0:
        raise ValueError("min_history must be nonnegative")
    examples: list[RetrievalExample] = []
    for impression in impressions:
        if len(impression.history) < min_history:
            continue
        seen = set(impression.history)
        for candidate in impression.candidates:
            if candidate.label == 1 and (not exclude_seen_targets or candidate.news_id not in seen):
                examples.append(
                    RetrievalExample(
                        impression_id=impression.impression_id,
                        user_id=impression.user_id,
                        timestamp=impression.timestamp,
                        history=impression.history,
                        positive_news_id=candidate.news_id,
                    )
                )
    return examples


def iter_exposed_candidates(impressions: Iterable[Impression]) -> Iterator[tuple[Impression, Candidate]]:
    """Yield only candidates with observed impression-level click labels."""

    for impression in impressions:
        for candidate in impression.candidates:
            yield impression, candidate
