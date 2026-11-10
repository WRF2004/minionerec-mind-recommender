#!/usr/bin/env python3
"""Encode exported MIND item text in stable numeric-ID order for RQ-VAE."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit("Install numpy and sentence-transformers for news embedding") from error
    items = json.loads(args.item_json.read_text(encoding="utf-8"))
    if set(items) != {str(i) for i in range(len(items))}:
        raise ValueError("item IDs must be contiguous from zero")
    texts = [f"{items[str(i)]['title']}. {items[str(i)]['description']}" for i in range(len(items))]
    model = SentenceTransformer(args.model)
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, embeddings.astype("float32"))
    print(f"saved {len(texts)} embeddings of dimension {embeddings.shape[1]} to {args.out}")


if __name__ == "__main__":
    main()
