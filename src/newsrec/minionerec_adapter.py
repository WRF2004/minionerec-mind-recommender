"""Convert MIND into the current MiniOneRec repository's data conventions.

The official MiniOneRec scripts expect integer item IDs, an `item.json` map,
an RQ-produced `index.json` map, and CSV columns containing Python list
literals. We keep the MIND ID mapping and per-query row order beside the
export so its rank-only evaluation JSON can be mapped back safely.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .content import CandidateScore
from .data import Impression, NewsArticle, RetrievalExample, build_retrieval_examples


CSV_FIELDS = (
    "user_id", "history_item_title", "item_title", "history_item_id", "item_id",
    "history_item_sid", "item_sid",
)
INTER_HEADER = "user_id:token\titem_id_list:token_seq\titem_id:token\n"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def export_catalog(news: Mapping[str, NewsArticle], directory: str | Path, dataset_name: str = "MIND") -> Dict[str, int]:
    """Write stable numeric mapping and metadata used by upstream RQ/SFT."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    item_ids = {news_id: i for i, news_id in enumerate(sorted(news))}
    metadata = {
        str(item_ids[news_id]): {
            "title": _clean(article.title),
            "description": _clean(article.abstract),
            "brand": "",
            "categories": article.category,
        }
        for news_id, article in news.items()
    }
    (directory / f"{dataset_name}.item.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    (directory / "news_to_item.json").write_text(json.dumps(item_ids, ensure_ascii=False), encoding="utf-8")
    return item_ids


def load_sid_index(path: str | Path, item_count: int) -> Dict[int, Tuple[str, ...]]:
    """Validate a complete, collision-free MiniOneRec semantic-ID index."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SID index must be a JSON object keyed by integer item ID")
    result: Dict[int, Tuple[str, ...]] = {}
    seen: Dict[Tuple[str, ...], int] = {}
    for raw_id, tokens in payload.items():
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid item ID in SID index: {raw_id!r}") from error
        if item_id < 0 or item_id >= item_count or str(item_id) != raw_id:
            raise ValueError(f"SID item ID outside catalog: {raw_id!r}")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(token, str) and re.fullmatch(r"<[a-z]_[0-9]+>", token) for token in tokens):
            raise ValueError(f"invalid semantic tokens for item {item_id}")
        sequence = tuple(tokens)
        if sequence in seen:
            raise ValueError(f"SID collision between item {seen[sequence]} and {item_id}; resolve before constrained decoding")
        seen[sequence] = item_id
        result[item_id] = sequence
    if set(result) != set(range(item_count)):
        raise ValueError(f"SID index covers {len(result)} of {item_count} catalog items")
    return result


def _eval_examples(impressions: Iterable[Impression]) -> List[RetrievalExample]:
    """Use one prompt per impression; keep all clicks for external metrics."""
    examples = build_retrieval_examples(impressions)
    result: List[RetrievalExample] = []
    seen = set()
    for example in examples:
        if example.impression_id not in seen:
            result.append(example)
            seen.add(example.impression_id)
    return result


def export_split(
    impressions: Iterable[Impression],
    news: Mapping[str, NewsArticle],
    item_ids: Mapping[str, int],
    directory: str | Path,
    split: str,
    sid_index: Optional[Mapping[int, Sequence[str]]] = None,
    dataset_name: str = "MIND",
    max_history: int = 50,
) -> int:
    """Write `.inter`, optional SID CSV, and evaluation row alignment JSONL."""
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    if max_history < 1:
        raise ValueError("max_history must be positive")
    directory = Path(directory)
    target = directory / split
    target.mkdir(parents=True, exist_ok=True)
    examples = build_retrieval_examples(impressions) if split == "train" else _eval_examples(impressions)
    inter_path = target / f"{dataset_name}.{split}.inter"
    csv_path = target / f"{dataset_name}.csv"
    alignment_path = target / f"{dataset_name}.queries.jsonl"
    with inter_path.open("w", encoding="utf-8", newline="") as inter, alignment_path.open("w", encoding="utf-8") as alignment:
        inter.write(INTER_HEADER)
        if sid_index is None:
            csv_handle = None
            writer = None
        else:
            csv_handle = csv_path.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
        try:
            written = 0
            for example in examples:
                history_ids = [item for item in example.history if item in item_ids][-max_history:]
                if not history_ids or example.positive_news_id not in item_ids:
                    continue
                target_id = item_ids[example.positive_news_id]
                numeric_history = [item_ids[item] for item in history_ids]
                inter.write(f"{example.user_id}\t{' '.join(map(str, numeric_history))}\t{target_id}\n")
                alignment.write(json.dumps({"impression_id": example.impression_id, "target_news_id": example.positive_news_id}, ensure_ascii=False) + "\n")
                if writer is not None:
                    assert sid_index is not None
                    sid_history = ["".join(sid_index[item_id]) for item_id in numeric_history]
                    writer.writerow({
                        "user_id": example.user_id,
                        "history_item_title": repr([_clean(news[item].title) for item in history_ids]),
                        "item_title": _clean(news[example.positive_news_id].title),
                        "history_item_id": repr(numeric_history),
                        "item_id": target_id,
                        "history_item_sid": repr(sid_history),
                        "item_sid": "".join(sid_index[target_id]),
                    })
                written += 1
        finally:
            if csv_handle is not None:
                csv_handle.close()
    return written


def export_item_info(news: Mapping[str, NewsArticle], item_ids: Mapping[str, int], sid_index: Mapping[int, Sequence[str]], directory: str | Path, dataset_name: str = "MIND") -> Path:
    target = Path(directory) / "info" / f"{dataset_name}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for news_id, item_id in sorted(item_ids.items(), key=lambda pair: pair[1]):
            handle.write(f"{''.join(sid_index[item_id])}\t{_clean(news[news_id].title)}\t{item_id}\n")
    return target


def load_upstream_predictions(
    result_json: str | Path,
    alignment_jsonl: str | Path,
    sid_index: Mapping[int, Sequence[str]],
    item_ids: Mapping[str, int],
) -> Dict[str, List[CandidateScore]]:
    """Map official `evaluate.py` rank-only JSON back to MIND impression IDs.

    Fails on a row count mismatch to prevent silent metric misalignment.
    Invalid generated SIDs are skipped. Returned scores are negative ranks,
    not model probabilities.
    """
    records = json.loads(Path(result_json).read_text(encoding="utf-8"))
    alignment = [json.loads(line) for line in Path(alignment_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != len(alignment):
        raise ValueError(f"prediction rows ({len(records)}) do not match alignment rows ({len(alignment)})")
    item_to_news = {item_id: news_id for news_id, item_id in item_ids.items()}
    sid_to_news = {"".join(tokens): item_to_news[item_id] for item_id, tokens in sid_index.items()}
    predictions: Dict[str, List[CandidateScore]] = {}
    for record, identity in zip(records, alignment):
        if not isinstance(record, dict) or not isinstance(record.get("predict"), list):
            raise ValueError("official result must contain a list in each record's predict field")
        target_sid = "".join(sid_index[item_ids[identity["target_news_id"]]])
        if "output" in record and str(record["output"]).strip() != target_sid:
            raise ValueError(f"prediction row does not match query {identity['impression_id']}")
        ranked = []
        seen = set()
        for raw_sid in record["predict"]:
            news_id = sid_to_news.get(str(raw_sid).strip())
            if news_id is None or news_id in seen:
                continue
            ranked.append(CandidateScore(news_id, -float(len(ranked)), "generation"))
            seen.add(news_id)
        predictions[identity["impression_id"]] = ranked
    return predictions

