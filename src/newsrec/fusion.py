"""Candidate fusion with a transparent, testable dynamic generation budget."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .content import CandidateScore


@dataclass(frozen=True)
class FusedCandidate:
    news_id: str
    score: float
    sources: Tuple[str, ...]
    generation_score: Optional[float]
    content_score: Optional[float]


def generation_confidence(candidates: Sequence[CandidateScore], top_n: int = 10) -> float:
    """Top-one softmax mass among top-n generation log scores, in [0, 1]."""
    scores = sorted((candidate.score for candidate in candidates), reverse=True)[:top_n]
    if not scores:
        return 0.0
    if len(scores) == 1:
        return 0.5
    largest = scores[0]
    exps = [math.exp(max(-60.0, score - largest)) for score in scores]
    return exps[0] / sum(exps)


def retrieval_agreement(generated: Sequence[CandidateScore], content: Sequence[CandidateScore], top_n: int = 10) -> float:
    """Overlap of two top-n lists; usable when upstream exports ranks only."""
    if not generated or not content:
        return 0.0
    gen_ids = {item.news_id for item in generated[:top_n]}
    con_ids = {item.news_id for item in content[:top_n]}
    return len(gen_ids & con_ids) / min(top_n, len(gen_ids), len(con_ids))


def generation_budget(k: int, history_length: int, confidence: float, fixed_fraction: Optional[float] = None) -> int:
    if k < 0 or history_length < 0:
        raise ValueError("k and history_length must be nonnegative")
    if fixed_fraction is not None:
        if not 0 <= fixed_fraction <= 1:
            raise ValueError("fixed_fraction must be in [0, 1]")
        fraction = fixed_fraction
    else:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        history_strength = min(history_length / 10.0, 1.0)
        fraction = 0.2 + 0.6 * confidence * history_strength
    return min(k, max(0, round(k * fraction)))


def fuse_candidates(
    generated: Sequence[CandidateScore],
    content: Sequence[CandidateScore],
    k: int,
    history_length: int,
    fixed_fraction: Optional[float] = None,
    rrf_constant: int = 60,
    confidence: Optional[float] = None,
) -> List[FusedCandidate]:
    """Allocate source quotas, deduplicate, then order selected items by RRF.

    Scores from the two retrievers need not share a scale. `fixed_fraction` is
    useful for a fixed-quota ablation; None selects the dynamic budget. If
    actual model confidence is unavailable, cross-retriever top-n agreement
    supplies a transparent proxy. The official evaluation export is rank-only.
    """
    if k <= 0:
        return []
    if rrf_constant < 0:
        raise ValueError("rrf_constant must be nonnegative")
    gen = list(dict((item.news_id, item) for item in generated).values())
    con = list(dict((item.news_id, item) for item in content).values())
    gen.sort(key=lambda item: (-item.score, item.news_id))
    con.sort(key=lambda item: (-item.score, item.news_id))
    evidence = retrieval_agreement(gen, con) if confidence is None else confidence
    budget = generation_budget(k, history_length, evidence, fixed_fraction)
    selected: Dict[str, None] = {}
    for item in gen[:budget]:
        selected[item.news_id] = None
    for item in con:
        if len(selected) >= k:
            break
        selected[item.news_id] = None
    for item in gen:
        if len(selected) >= k:
            break
        selected[item.news_id] = None
    gen_rank = {item.news_id: i + 1 for i, item in enumerate(gen)}
    con_rank = {item.news_id: i + 1 for i, item in enumerate(con)}
    gen_score = {item.news_id: item.score for item in gen}
    con_score = {item.news_id: item.score for item in con}
    result = []
    for news_id in selected:
        rank_g = gen_rank.get(news_id)
        rank_c = con_rank.get(news_id)
        score = (1.0 / (rrf_constant + rank_g) if rank_g is not None else 0.0) + (
            1.0 / (rrf_constant + rank_c) if rank_c is not None else 0.0
        )
        result.append(
            FusedCandidate(
                news_id=news_id,
                score=score,
                sources=tuple(source for source, rank in (("generation", rank_g), ("content", rank_c)) if rank is not None),
                generation_score=gen_score.get(news_id),
                content_score=con_score.get(news_id),
            )
        )
    return sorted(result, key=lambda item: (-item.score, item.news_id))
