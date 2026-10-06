"""Checkpoint paths."""

from pathlib import Path


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        for name in ("model_bf16.pt", "best.pt", "ckpt.pt"):
            if (path / name).exists():
                return path / name
        raise FileNotFoundError(f"no model_bf16.pt / best.pt / ckpt.pt in {path}")
    return path
