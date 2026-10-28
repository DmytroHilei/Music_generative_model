"""The engine's settings: config/serve/engine.toml, any key overridable (python -m musicar.serve --help)."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "serve" / "engine.toml"


@dataclass
class EngineConfig:
    checkpoint: str = "checkpoints/f1"
    base: list = field(default_factory=lambda: ["f1"])  # generate.py presets every request starts from
    name: str = "laptop"
    device: str = "auto"
    max_queue: int = 8
    idle_unload_min: float = 10.0  # 0 = keep the model loaded
    keep_jobs_min: float = 60.0
    heartbeat_s: float = 20.0
    min_free_mb: int = 3500
    batch_candidates: int = 4  # best-of candidates per batch (GPU memory)
    default_preset: str = "balanced"
    presets: dict = field(default_factory=lambda: {"balanced": {}})  # name -> decoding options on top of `base`
    auth: bool = True  # False: no token needed (local development only)
    tokens_db: str = "data/serve/tokens.sqlite"  # relative to the repository
    per_token_jobs: int = 2  # unfinished jobs per token at most
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000"])  # pages allowed to call the engine
    admin_key_file: str = "data/serve/admin.key"  # the page's server key for /v1/admin/invites; missing = route off

    @property
    def tokens_path(self):
        return ROOT / self.tokens_db

    @property
    def admin_key_path(self):
        return ROOT / self.admin_key_file

    @classmethod
    def load(cls, path=CONFIG, **overrides):
        """The TOML file's values, then `overrides` (None = keep the file's)."""
        values = tomllib.loads(Path(path).read_text())
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"{path}: unknown engine settings {', '.join(unknown)} "
                             f"(known: {', '.join(sorted(known))})")
        return cls(**values)
