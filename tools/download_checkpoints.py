#!/usr/bin/env python3
"""Download public RELO checkpoints and encoder weights from Hugging Face Hub."""

from __future__ import annotations

import argparse
from typing import Iterable


DEFAULT_REPO_ID = "xche32/RELO"

CHECKPOINT_PATHS = {
    "t256": [
        "checkpoints/train/relo_warmup/relo_warmup_t256/RELO_WARMUP_ep0090.pth.tar",
        "checkpoints/train/relo/relo_t256/RELO_ep0090.pth.tar",
    ],
    "b256": [
        "checkpoints/train/relo_warmup/relo_warmup_b256/RELO_WARMUP_ep0090.pth.tar",
        "checkpoints/train/relo/relo_b256/RELO_ep0090.pth.tar",
    ],
    "l256": [
        "checkpoints/train/relo_warmup/relo_warmup_l256/RELO_WARMUP_ep0090.pth.tar",
        "checkpoints/train/relo/relo_l256/RELO_ep0090.pth.tar",
    ],
}

ENCODER_PATHS = {
    "t256": [
        "pretrained/itpn/fast_itpn_tiny_1600e_1k.pt",
    ],
    "b256": [
        "pretrained/itpn/fast_itpn_base_clipl_e1600.pt",
    ],
    "l256": [
        "pretrained/itpn/fast_itpn_large_1600e_1k.pt",
    ],
}


def resolve_patterns(variant: str) -> list[str]:
    if variant == "all":
        return [
            path
            for variant_name in ("t256", "b256", "l256")
            for path in ENCODER_PATHS[variant_name] + CHECKPOINT_PATHS[variant_name]
        ]
    return ENCODER_PATHS[variant] + CHECKPOINT_PATHS[variant]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repository ID. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--variant",
        choices=("t256", "b256", "l256", "all"),
        default="all",
        help="Checkpoint variant to download. Default: all",
    )
    parser.add_argument(
        "--local-dir",
        default=".",
        help="Destination directory. Default: current directory",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    patterns = resolve_patterns(args.variant)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with "
            "`python -m pip install huggingface_hub` or `bash install.sh`."
        ) from exc

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=args.local_dir,
        allow_patterns=patterns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
