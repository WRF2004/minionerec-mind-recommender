#!/usr/bin/env python3
"""Export MIND-small train/dev data for the official MiniOneRec codebase."""

import argparse
import json
from pathlib import Path

from newsrec.data import merge_news_catalogs, read_behaviors, read_news, temporal_split
from newsrec.minionerec_adapter import export_catalog, export_item_info, export_split, load_sid_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True, help="MIND-small train directory")
    parser.add_argument("--dev-dir", type=Path, help="MIND-small dev directory with labeled behaviors")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sid-index", type=Path, help="RQ-produced MIND.index.json; omit for pre-SID export")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    train_news = read_news(args.train_dir / "news.tsv")
    news = merge_news_catalogs(train_news, read_news(args.dev_dir / "news.tsv")) if args.dev_dir else train_news
    raw_train = read_behaviors(args.train_dir / "behaviors.tsv")
    train, valid = temporal_split(raw_train, args.val_fraction)
    test = read_behaviors(args.dev_dir / "behaviors.tsv") if args.dev_dir else []

    item_ids = export_catalog(news, args.out)
    sid_index = load_sid_index(args.sid_index, len(item_ids)) if args.sid_index else None
    if sid_index is not None:
        canonical_index = {str(item_id): list(sid_index[item_id]) for item_id in sorted(sid_index)}
        (args.out / "MIND.index.json").write_text(json.dumps(canonical_index, ensure_ascii=False), encoding="utf-8")
    counts = {
        "train": export_split(train, news, item_ids, args.out, "train", sid_index),
        "valid": export_split(valid, news, item_ids, args.out, "valid", sid_index),
        "test": export_split(test, news, item_ids, args.out, "test", sid_index),
    }
    if sid_index is not None:
        export_item_info(news, item_ids, sid_index, args.out)
    manifest = {
        "catalog_items": len(item_ids),
        "train_impressions": len(train),
        "valid_impressions": len(valid),
        "test_impressions": len(test),
        "retrieval_rows": counts,
        "sid_index_supplied": sid_index is not None,
        "train_end": train[-1].timestamp.isoformat(),
        "valid_start": valid[0].timestamp.isoformat(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
