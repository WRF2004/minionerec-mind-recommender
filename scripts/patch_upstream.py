#!/usr/bin/env python3
"""Apply the minimal MIND category adaptation to a MiniOneRec checkout.

The upstream scripts hard-code a category lookup. This patch adds news and
removes evaluate.py's hard-coded GPU 0 override. It is idempotent and fails if
the expected upstream source has changed.
"""

import argparse
from pathlib import Path


def patch_text(filename: str, text: str) -> str:
    if filename not in {"sft.py", "rl.py", "evaluate.py"}:
        raise ValueError(f"unexpected filename: {filename}")
    if '"MIND": "news articles"' not in text:
        needle = '"Books": "books"'
        if needle not in text:
            raise ValueError(f"{filename}: upstream category dictionary changed")
        text = text.replace(needle, needle + ', "MIND": "news articles"', 1)
    if filename == "evaluate.py":
        text = text.replace('    os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n', "")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_dir", type=Path)
    parser.add_argument("--check", action="store_true", help="verify patch can apply without writing")
    args = parser.parse_args()
    for name in ("sft.py", "rl.py", "evaluate.py"):
        path = args.upstream_dir / name
        original = path.read_text(encoding="utf-8")
        updated = patch_text(name, original)
        if not args.check and updated != original:
            path.write_text(updated, encoding="utf-8")
        print(f"{name}: {'already adapted' if updated == original else 'ready' if args.check else 'adapted'}")


if __name__ == "__main__":
    main()
