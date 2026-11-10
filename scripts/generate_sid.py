#!/usr/bin/env python3
"""Export collision-free MIND semantic IDs from a MiniOneRec RQ-VAE checkpoint."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


def resolve_collisions(codes):
    """Append a deterministic token for every duplicate RQ code."""
    groups = defaultdict(list)
    for item_id, code in enumerate(codes):
        groups[tuple(code)].append(item_id)
    result = [list(code) for code in codes]
    raw_codes = set(groups)
    used = {code for code, item_ids in groups.items() if len(item_ids) == 1}
    for code, item_ids in groups.items():
        if len(item_ids) < 2:
            continue
        suffix_level = chr(ord("a") + len(code))
        suffix = 0
        for item_id in item_ids:
            while code + (f"<{suffix_level}_{suffix}>",) in raw_codes or code + (f"<{suffix_level}_{suffix}>",) in used:
                suffix += 1
            result[item_id].append(f"<{suffix_level}_{suffix}>")
            used.add(tuple(result[item_id]))
            suffix += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit("Install the official MiniOneRec requirements to generate SIDs") from error
    sys.path.insert(0, str((args.upstream_dir / "rq").resolve()))
    from datasets import EmbDataset  # type: ignore
    from models.rqvae import RQVAE  # type: ignore

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    data = EmbDataset(cfg.data_path)
    kwargs = {name: getattr(cfg, name) for name in (
        "num_emb_list", "e_dim", "layers", "dropout_prob", "bn", "loss_type",
        "quant_loss_weight", "kmeans_init", "kmeans_iters", "sk_epsilons", "sk_iters",
    )}
    model = RQVAE(in_dim=data.dim, **kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    codes = []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=args.batch_size, shuffle=False):
            indices = model.get_indices(batch.to(device), use_sk=False).reshape(-1, len(cfg.num_emb_list)).cpu().tolist()
            for indices_for_item in indices:
                codes.append([f"<{chr(ord('a') + level)}_{int(index)}>" for level, index in enumerate(indices_for_item)])
    codes = resolve_collisions(codes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({str(i): code for i, code in enumerate(codes)}), encoding="utf-8")
    print(f"saved {len(codes)} unique semantic IDs to {args.out}")


if __name__ == "__main__":
    main()
