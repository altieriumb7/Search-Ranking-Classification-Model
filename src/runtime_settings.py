from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_path(name: str, default: Path, project_root: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path


@dataclass(frozen=True)
class RuntimeSettings:
    demo_mode: bool
    allow_live_runs: bool
    require_session_api_key: bool
    benchmark_mode: str
    openai_api_key_present: bool
    default_config_path: Path
    reports_dir: Path

    @property
    def mode_label(self) -> str:
        return "demo" if self.demo_mode else "live"

    @property
    def live_runs_enabled(self) -> bool:
        return (not self.demo_mode) and self.allow_live_runs
