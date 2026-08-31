"""Configuration for evaluation logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "eval_results"


@dataclass
class EvalLogConfig:
    """Where and how to record evaluation metrics."""

    output_dir: Path = DEFAULT_LOG_DIR
    run_name: str = ""
    log_episode_events: bool = True
    log_planning: bool = True
    save_json: bool = True
    save_episode_csv: bool = True
    quiet: bool = False

    def resolve_run_name(self, env_key: str, seed: int) -> str:
        if self.run_name:
            return self.run_name
        return f"{env_key}_seed{seed}"
