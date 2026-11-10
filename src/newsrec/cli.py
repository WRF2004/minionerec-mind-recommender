"""Offline benchmark for content retrieval, MiniOneRec fusion and click ranking."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Dict, List, Mapping, Sequence

from .baselines import ItemCFIndex, PopularityIndex
from .content import CandidateScore, ContentIndex
from .data import Impression, merge_news_catalogs, read_behaviors, read_news, temporal_split
from .fusion import fuse_candidates
from .metrics import ranking_metrics, recall_at_k
from .minionerec_adapter import load_sid_index, load_upstream_predictions
from .ranking import FEATURE_NAMES, PairwiseRanker, build_rank_rows, fit_lightgbm


def _take(rows: Sequence[Impression], limit: int) -> Sequence[Impression]:
    return rows if limit == 0 else rows[:limit]


def _percentile(values: Sequence[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _load_generation(path: Path | None, upstream_data: Path | None, split: str):
    if path is None:
        return {}
    if upstream_data is None:
        raise ValueError("--upstream-data is required with a generation result JSON")
    item_ids = json.loads((upstream_data / "news_to_item.json").read_text(encoding="utf-8"))
    sid_index = load_sid_index(upstream_data / "MIND.index.json", len(item_ids))
    return load_upstream_predictions(path, upstream_data / split / "MIND.queries.jsonl", sid_index, item_ids)


def _retrieval_report(
    impressions: Sequence[Impression],
    index: ContentIndex,
    generation: Mapping[str, Sequence[CandidateScore]],
    k: int,
    popularity: Counter,
    popular_index: PopularityIndex,
    itemcf_index: ItemCFIndex,
):
    base_methods = ("popularity", "itemcf", "content")
    method_names = base_methods + ("generation", "fixed_fusion", "adaptive_fusion") if generation else base_methods
    predictions: Dict[str, Dict[str, List[str]]] = {name: {} for name in method_names}
    clicked: Dict[str, List[str]] = {}
    clicked_tail: Dict[str, List[str]] = {}
    latencies = []
    head_count = max(1, round(len(index.vectors) * 0.2))
    head_ids = set(sorted(index.vectors, key=lambda news_id: (-popularity[news_id], news_id))[:head_count])
    for impression in impressions:
        if not impression.history:
            continue
        history = set(impression.history)
        positive = [candidate.news_id for candidate in impression.candidates if candidate.label == 1 and candidate.news_id not in history]
        if not positive:
            continue
        clicked[impression.impression_id] = positive
        tail = [news_id for news_id in positive if news_id not in head_ids]
        if tail:
            clicked_tail[impression.impression_id] = tail
        start = time.perf_counter()
        popular = popular_index.search(impression.history, k)
        itemcf = itemcf_index.search(impression.history, k)
        content = index.search(impression.history, k)
        generated = [item for item in generation.get(impression.impression_id, ()) if item.news_id not in history][:k]
        predictions["popularity"][impression.impression_id] = [item.news_id for item in popular]
        predictions["itemcf"][impression.impression_id] = [item.news_id for item in itemcf]
        predictions["content"][impression.impression_id] = [item.news_id for item in content]
        if generation:
            fixed = fuse_candidates(generated, content, k, len(impression.history), fixed_fraction=0.5)
            adaptive = fuse_candidates(generated, content, k, len(impression.history))
            predictions["generation"][impression.impression_id] = [item.news_id for item in generated]
            predictions["fixed_fusion"][impression.impression_id] = [item.news_id for item in fixed]
            predictions["adaptive_fusion"][impression.impression_id] = [item.news_id for item in adaptive]
        latencies.append((time.perf_counter() - start) * 1000)
    cutoffs = sorted(set((min(10, k), k)))
    report = {name: recall_at_k(prediction, clicked, cutoffs) for name, prediction in predictions.items()}
    tail_report = {name: recall_at_k(prediction, clicked_tail, cutoffs) for name, prediction in predictions.items()}
    return {
        "all_clicked": report,
        "tail_clicked": tail_report,
        "generation_available": bool(generation),
        "latency_ms": {"median": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
        "tail_definition": "clicked items outside the top 20% of catalog IDs by retrieval-training click count; unseen items are tail",
    }


def benchmark(args) -> dict:
    train_news = read_news(args.train_dir / "news.tsv")
    news = merge_news_catalogs(train_news, read_news(args.dev_dir / "news.tsv")) if args.dev_dir else train_news
    raw_train = read_behaviors(args.train_dir / "behaviors.tsv")
    train, valid = temporal_split(raw_train, args.val_fraction)
    test = read_behaviors(args.dev_dir / "behaviors.tsv") if args.dev_dir else []
    valid = _take(valid, args.max_ranker_impressions)
    test = _take(test, args.max_eval_impressions)
    index = ContentIndex().fit(news)
    generation = {
        "valid": _load_generation(args.valid_generation_json, args.upstream_data, "valid"),
        "test": _load_generation(args.test_generation_json, args.upstream_data, "test"),
    }
    ranker_ranks = {key: [item.news_id for item in value] for key, value in generation["valid"].items()}
    ranker_rows = build_rank_rows(valid, news, index, ranker_ranks)
    model = PairwiseRanker().fit(ranker_rows, epochs=args.rank_epochs) if args.rank_backend == "pairwise" else fit_lightgbm(ranker_rows)
    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        if args.rank_backend == "pairwise":
            model.save(args.model_out)
        else:
            model.booster_.save_model(str(args.model_out))
    popular_index = PopularityIndex().fit(train)
    itemcf_index = ItemCFIndex().fit(train)
    popularity = popular_index.counts
    report = {
        "data": {"catalog_items": len(news), "retriever_train_impressions": len(train), "ranker_train_impressions": len(valid), "test_impressions": len(test)},
        "ranking_feature_names": list(FEATURE_NAMES),
        "ranking_backend": args.rank_backend,
        "ranking_weights": model.weights if args.rank_backend == "pairwise" else None,
        "results": {},
        "limitations": [
            "Ranking labels exist only for exposed candidates.",
            "Full-catalog recall uses observed clicked items as positives; unexposed items are unlabeled.",
            "The content catalog is transductive because MIND has no per-news publication timestamp.",
            "Latency is offline Python process time, not production serving P95.",
        ],
    }
    for split, impressions in (("valid", valid), ("test", test)):
        if not impressions:
            continue
        gen_ranks = {key: [item.news_id for item in value] for key, value in generation[split].items()}
        result = {"retrieval": _retrieval_report(impressions, index, generation[split], args.k, popularity, popular_index, itemcf_index)}
        if split == "test":
            rows = build_rank_rows(impressions, news, index, gen_ranks)
            scores = model.predict(rows) if args.rank_backend == "pairwise" else model.predict([row.features for row in rows])
            result["ranking"] = {
                "content_similarity": ranking_metrics((row.impression_id, row.label, row.features[0]) for row in rows),
                "pairwise": ranking_metrics((row.impression_id, row.label, score) for row, score in zip(rows, scores)),
            }
        report["results"][split] = result
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--dev-dir", type=Path)
    parser.add_argument("--upstream-data", type=Path, help="directory from prepare_minionerec.py with SID index")
    parser.add_argument("--valid-generation-json", type=Path)
    parser.add_argument("--test-generation-json", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-ranker-impressions", type=int, default=2000, help="0 = all")
    parser.add_argument("--max-eval-impressions", type=int, default=500, help="0 = all")
    parser.add_argument("-k", type=int, default=50)
    parser.add_argument("--rank-epochs", type=int, default=10)
    parser.add_argument("--rank-backend", choices=("pairwise", "lightgbm"), default="pairwise")
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.k < 1 or args.max_ranker_impressions < 0 or args.max_eval_impressions < 0:
        parser.error("k must be positive and impression limits must be nonnegative")
    result = benchmark(args)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
